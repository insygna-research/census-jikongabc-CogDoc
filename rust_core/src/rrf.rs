use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

// 将 Python 值(字符串/整数)统一转成字符串，用于拼接文档键
fn key_part_from_py(value: Bound<'_, PyAny>) -> String {
    if let Ok(value) = value.extract::<String>() {
        return value;
    }
    if let Ok(value) = value.extract::<i64>() {
        return value.to_string();
    }
    if let Ok(value) = value.extract::<u64>() {
        return value.to_string();
    }

    value
        .str()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

// 由 meta 的 source 与 chunk_index 拼出文档唯一键 "source::chunk_index"
fn extract_doc_key(doc: &Bound<'_, PyDict>) -> PyResult<String> {
    let Some(meta) = doc.get_item("meta")? else {
        return Ok("::".to_string()); // 无 meta：返回空键占位，不参与正常去重
    };
    let Ok(meta_dict) = meta.cast_into::<PyDict>() else {
        return Ok("::".to_string()); // meta 非字典：同样退化为空键
    };

    let source = meta_dict
        .get_item("source")?
        .map(key_part_from_py)
        .unwrap_or_default();
    let chunk_index = meta_dict
        .get_item("chunk_index")?
        .map(key_part_from_py)
        .unwrap_or_default();

    Ok(format!("{}::{}", source, chunk_index))
}

// 取出文档的 retrieval 子字典，不存在则新建并挂回文档
fn get_or_create_retrieval<'py>(doc: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    if let Some(retrieval) = doc.get_item("retrieval")?
        && let Ok(retrieval_dict) = retrieval.cast_into::<PyDict>()
    {
        return Ok(retrieval_dict);
    }

    let retrieval = PyDict::new(doc.py());
    doc.set_item("retrieval", &retrieval)?;
    Ok(retrieval)
}

// 将 source 文档的 retrieval 字段并入 target(同一文档补充另一路召回的得分)
fn merge_retrieval(target_doc: &Bound<'_, PyDict>, source_doc: &Bound<'_, PyDict>) -> PyResult<()> {
    let Some(source_retrieval) = source_doc.get_item("retrieval")? else {
        return Ok(());
    };
    let Ok(source_retrieval_dict) = source_retrieval.cast_into::<PyDict>() else {
        return Ok(());
    };

    let target_retrieval = get_or_create_retrieval(target_doc)?;
    for (key, value) in source_retrieval_dict.iter() {
        target_retrieval.set_item(key, value)?; // 同名字段以 source 为准覆盖写入
    }
    Ok(())
}

// 累加该文档的 RRF 得分；键已存在则合并 retrieval，否则新建条目
fn add_or_update_doc<'py>(
    score_map: &mut HashMap<String, (f64, Bound<'py, PyDict>)>,
    key: String,
    score: f64,
    doc: Bound<'py, PyDict>,
) -> PyResult<()> {
    if let Some(entry) = score_map.get_mut(&key) {
        entry.0 += score; // .0 为累计 RRF 得分，.1 为已登记文档
        merge_retrieval(&entry.1, &doc)?; // 把本路召回的得分字段并入已存文档
    } else {
        score_map.insert(key, (score, doc)); // 首次命中，直接登记
    }

    Ok(())
}

// RRF 融合：两路召回各按排名累加 1/(k+rank) 得分，去重合并后取前 top_n
pub fn rrf_fusion_core<'py>(
    vector_docs: Vec<Bound<'py, PyDict>>,
    bm25_docs: Vec<Bound<'py, PyDict>>,
    k: f64,
    top_n: usize,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let mut score_map: HashMap<String, (f64, Bound<'py, PyDict>)> = HashMap::new();

    // 向量召回：排名越靠前(idx 越小)得分越高
    for (idx, doc) in vector_docs.into_iter().enumerate() {
        let key = extract_doc_key(&doc)?;
        let score = 1.0 / (k + (idx + 1) as f64); // RRF 公式 1/(k+rank)，idx+1 将 0 基下标转 1 基排名
        add_or_update_doc(&mut score_map, key, score, doc)?;
    }

    // BM25 召回：同键文档得分叠加到向量召回之上
    for (idx, doc) in bm25_docs.into_iter().enumerate() {
        let key = extract_doc_key(&doc)?;
        let score = 1.0 / (k + (idx + 1) as f64); // 同一公式，与向量召回得分可直接相加

        add_or_update_doc(&mut score_map, key, score, doc)?;
    }

    // 按融合得分降序；得分相同时按文档键升序，保证结果稳定可复现
    let mut combined: Vec<(String, f64, Bound<'py, PyDict>)> = score_map
        .into_iter()
        .map(|(key, (score, doc))| (key, score, doc))
        .collect();
    combined.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let final_docs: Vec<Bound<'py, PyDict>> = combined
        .into_iter()
        .take(top_n)
        .map(|(_, rrf_score, doc)| {
            // 写回最终融合得分，并标记为混合召回渠道
            let retrieval = get_or_create_retrieval(&doc)?;
            retrieval.set_item("rrf_score", rrf_score)?;
            retrieval.set_item("search_channel", "hybrid")?;
            Ok(doc)
        })
        .collect::<PyResult<Vec<_>>>()?;

    Ok(final_docs)
}
