use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

// Python 字符串和整数都要转成可排序键片段。
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

// 优先使用标准化 chunk_id。
fn extract_doc_key(doc: &Bound<'_, PyDict>) -> PyResult<String> {
    let Some(meta) = doc.get_item("meta")? else {
        return Ok("::".to_string());
    };
    let Ok(meta_dict) = meta.cast_into::<PyDict>() else {
        return Ok("::".to_string());
    };

    if let Some(chunk_id) = meta_dict.get_item("chunk_id")? {
        let chunk_id = key_part_from_py(chunk_id);
        if !chunk_id.is_empty() {
            return Ok(chunk_id);
        }
    }

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

// fusion 输出必须带 retrieval 字典。
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

// 同一 chunk 的两路召回指标合并到同一文档。
fn merge_retrieval(target_doc: &Bound<'_, PyDict>, source_doc: &Bound<'_, PyDict>) -> PyResult<()> {
    let Some(source_retrieval) = source_doc.get_item("retrieval")? else {
        return Ok(());
    };
    let Ok(source_retrieval_dict) = source_retrieval.cast_into::<PyDict>() else {
        return Ok(());
    };

    let target_retrieval = get_or_create_retrieval(target_doc)?;
    for (key, value) in source_retrieval_dict.iter() {
        target_retrieval.set_item(key, value)?;
    }
    Ok(())
}

// 同一 chunk_id 累加 RRF 分数。
fn add_or_update_doc<'py>(
    score_map: &mut HashMap<String, (f64, Bound<'py, PyDict>)>,
    key: String,
    score: f64,
    doc: Bound<'py, PyDict>,
) -> PyResult<()> {
    if let Some(entry) = score_map.get_mut(&key) {
        entry.0 += score;
        merge_retrieval(&entry.1, &doc)?;
    } else {
        score_map.insert(key, (score, doc));
    }

    Ok(())
}

// RRF 融合后按分数降序、身份键升序输出。
pub fn rrf_fusion_core<'py>(
    vector_docs: Vec<Bound<'py, PyDict>>,
    bm25_docs: Vec<Bound<'py, PyDict>>,
    k: f64,
    top_n: usize,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let mut score_map: HashMap<String, (f64, Bound<'py, PyDict>)> = HashMap::new();

    for (idx, doc) in vector_docs.into_iter().enumerate() {
        let key = extract_doc_key(&doc)?;
        let score = 1.0 / (k + (idx + 1) as f64);
        add_or_update_doc(&mut score_map, key, score, doc)?;
    }

    for (idx, doc) in bm25_docs.into_iter().enumerate() {
        let key = extract_doc_key(&doc)?;
        let score = 1.0 / (k + (idx + 1) as f64);

        add_or_update_doc(&mut score_map, key, score, doc)?;
    }

    let mut combined: Vec<(String, f64, Bound<'py, PyDict>)> = score_map
        .into_iter()
        .map(|(key, (score, doc))| (key, score, doc))
        .collect();
    combined.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let final_docs: Vec<Bound<'py, PyDict>> = combined
        .into_iter()
        .take(top_n)
        .map(|(_, rrf_score, doc)| {
            let retrieval = get_or_create_retrieval(&doc)?;
            retrieval.set_item("rrf_score", rrf_score)?;
            retrieval.set_item("search_channel", "hybrid")?;
            Ok(doc)
        })
        .collect::<PyResult<Vec<_>>>()?;

    Ok(final_docs)
}
