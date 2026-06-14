use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

mod rrf;
mod scanner;

// 扫描 PDF 目录生成指纹清单，供 Python 端校验本地索引是否过期
#[pyfunction]
fn scan_pdf_manifest_native<'py>(
    py: Python<'py>,
    doc_id: String,
    doc_dir: String,
) -> PyResult<Bound<'py, PyDict>> {
    let scanned_files = match scanner::parallel_scan_manifest(&doc_dir) {
        Ok(files) => files,
        Err(e) => return Err(pyo3::exceptions::PyIOError::new_err(e.to_string())), // IO 错误转成 Python 异常
    };

    // 把每个文件指纹封装成 Python 字典
    let py_list = PyList::empty(py);
    for file in scanned_files {
        let single_file_dict = PyDict::new(py);
        single_file_dict.set_item("name", file.name)?;
        single_file_dict.set_item("size", file.size)?;
        single_file_dict.set_item("sha256", file.sha256)?;
        py_list.append(single_file_dict)?;
    }

    // 组装最终清单；doc_dir 为机器相关路径，Python 端比对时会忽略
    let final_manifest = PyDict::new(py);
    final_manifest.set_item("doc_id", doc_id)?;
    final_manifest.set_item("doc_dir", doc_dir)?;
    final_manifest.set_item("documents", py_list)?;

    Ok(final_manifest)
}

// 对向量召回与 BM25 召回结果做 RRF 融合，返回前 top_n 文档
#[pyfunction]
fn rrf_fusion_native<'py>(
    vector_docs: Vec<Bound<'py, PyDict>>,
    bm25_docs: Vec<Bound<'py, PyDict>>,
    k: f64,
    top_n: usize,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    rrf::rrf_fusion_core(vector_docs, bm25_docs, k, top_n)
}

// 模块入口：向 Python 注册导出的 native 函数
#[pymodule]
fn rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scan_pdf_manifest_native, m)?)?;
    m.add_function(wrap_pyfunction!(rrf_fusion_native, m)?)?;
    Ok(())
}
