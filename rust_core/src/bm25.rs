use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;

// 原生 BM25 与 rank_bm25.BM25Okapi 对齐，并在 Rust 内完成 top-k 排序。
#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct Bm25Index {
    k1: f64,
    b: f64,
    epsilon: f64,
    avgdl: f64,
    corpus_size: usize,
    doc_len: Vec<f64>,
    idf: HashMap<String, f64>,
    // term -> [(doc_id, term_freq)]，f=0 的文档对该词贡献恒为 0，无需存。
    postings: HashMap<String, Vec<(usize, f64)>>,
    // 保留分词语料：支撑增量重建与字节序列化复用，避免回 Python 侧重新分词或重建。
    corpus: Vec<Vec<String>>,
}

// 由分词语料构建索引；构造与增量重建共用，保证两条路径行为一致。
fn build_index(corpus: Vec<Vec<String>>, k1: f64, b: f64, epsilon: f64) -> Bm25Index {
    let corpus_size = corpus.len();
    let mut doc_len: Vec<f64> = Vec::with_capacity(corpus_size);
    let mut postings: HashMap<String, Vec<(usize, f64)>> = HashMap::new();
    let mut nd: HashMap<String, u32> = HashMap::new();
    let mut total_tokens: usize = 0;

    for (doc_id, doc) in corpus.iter().enumerate() {
        doc_len.push(doc.len() as f64);
        total_tokens += doc.len();

        let mut freqs: HashMap<&str, u32> = HashMap::new();
        for word in doc {
            *freqs.entry(word.as_str()).or_insert(0) += 1;
        }
        for (word, freq) in freqs {
            postings
                .entry(word.to_string())
                .or_default()
                .push((doc_id, freq as f64));
            *nd.entry(word.to_string()).or_insert(0) += 1;
        }
    }

    let avgdl = if corpus_size > 0 {
        total_tokens as f64 / corpus_size as f64
    } else {
        0.0
    };

    // IDF 与负值 epsilon 下限完全对齐 BM25Okapi._calc_idf。
    let mut idf: HashMap<String, f64> = HashMap::with_capacity(nd.len());
    let mut idf_sum = 0.0_f64;
    let mut negative_idfs: Vec<String> = Vec::new();
    for (word, freq) in &nd {
        let value = (corpus_size as f64 - *freq as f64 + 0.5).ln() - (*freq as f64 + 0.5).ln();
        idf.insert(word.clone(), value);
        idf_sum += value;
        if value < 0.0 {
            negative_idfs.push(word.clone());
        }
    }
    // average_idf 按唯一词数取平均，与 len(self.idf) 一致。
    let unique_terms = nd.len().max(1);
    let average_idf = idf_sum / unique_terms as f64;
    let eps = epsilon * average_idf;
    for word in negative_idfs {
        idf.insert(word, eps);
    }

    Bm25Index {
        k1,
        b,
        epsilon,
        avgdl,
        corpus_size,
        doc_len,
        idf,
        postings,
        corpus,
    }
}

#[pymethods]
impl Bm25Index {
    // 从分词语料创建 BM25 索引。
    #[new]
    #[pyo3(signature = (tokenized_corpus, k1 = 1.5, b = 0.75, epsilon = 0.25))]
    fn new(tokenized_corpus: Vec<Vec<String>>, k1: f64, b: f64, epsilon: f64) -> Self {
        build_index(tokenized_corpus, k1, b, epsilon)
    }

    // 序列化为字节，供 Python 端落盘；规避对 List[List[str]] 的 pickle 大列表开销。
    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let data = bincode::serialize(self).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &data))
    }

    // 从字节直接还原索引：跳过启动时的全量重建。
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        bincode::deserialize(data).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    // 增量重建：保留 keep_indices 指向的旧文档分词，追加新文档分词后整体重建。 BM25 分数依赖全局 IDF/avgdl，必须整体重建，但未变文档复用已有分词、不回 Python 重新切词。
    fn rebuild_from_kept(
        &self,
        keep_indices: Vec<usize>,
        new_tokenized: Vec<Vec<String>>,
    ) -> PyResult<Self> {
        let mut corpus: Vec<Vec<String>> =
            Vec::with_capacity(keep_indices.len() + new_tokenized.len());
        for idx in keep_indices {
            let row = self
                .corpus
                .get(idx)
                .ok_or_else(|| PyValueError::new_err("keep index out of range"))?;
            corpus.push(row.clone());
        }
        corpus.extend(new_tokenized);
        Ok(build_index(corpus, self.k1, self.b, self.epsilon))
    }

    // 同分按 doc_id 升序，固定 top-k 边界的稳定次序。
    fn score_topk(&self, query: Vec<String>, top_n: usize) -> Vec<(usize, f64)> {
        if self.corpus_size == 0 {
            return Vec::new();
        }

        // query 去重为计数：等价于 Python 对 query 列表逐 token 累加（重复词多次计分）。
        let mut query_counts: HashMap<&str, f64> = HashMap::new();
        for token in &query {
            *query_counts.entry(token.as_str()).or_insert(0.0) += 1.0;
        }

        let mut scores = vec![0.0_f64; self.corpus_size];
        for (token, count) in &query_counts {
            let idf_q = match self.idf.get(*token) {
                Some(value) => *value,
                None => continue,
            };
            if let Some(plist) = self.postings.get(*token) {
                for (doc_id, freq) in plist {
                    let denom = *freq
                        + self.k1 * (1.0 - self.b + self.b * self.doc_len[*doc_id] / self.avgdl);
                    scores[*doc_id] += count * idf_q * (*freq * (self.k1 + 1.0) / denom);
                }
            }
        }

        let mut order: Vec<usize> = (0..self.corpus_size).collect();
        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(Ordering::Equal)
                .then(a.cmp(&b))
        });
        order
            .into_iter()
            .take(top_n)
            .map(|doc_id| (doc_id, scores[doc_id]))
            .collect()
    }
}
