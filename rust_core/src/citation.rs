use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList};
use std::collections::{BTreeSet, HashMap};

const FALLBACK_MARKER: &str = "在所提供的参考资料中未找到与该问题相关的内容";

// 保存答案引用中的来源文件和页码。
#[derive(Debug, PartialEq, Eq)]
struct Citation {
    source: String,
    page: i64,
}

// 引用标签支持半角和全角左括号。
fn is_open_bracket(ch: char) -> bool {
    ch == '[' || ch == '［'
}

// 引用标签支持半角和全角右括号。
fn is_close_bracket(ch: char) -> bool {
    ch == ']' || ch == '］'
}

// 文件名与页码之间支持半角和全角冒号。
fn is_colon(ch: char) -> bool {
    ch == ':' || ch == '：'
}

// 页码前缀兼容中英文输入法下的 P。
fn is_page_prefix(ch: char) -> bool {
    matches!(ch, 'p' | 'P' | 'ｐ' | 'Ｐ')
}

// 页码数字兼容半角数字和全角数字。
fn decimal_digit_value(ch: char) -> Option<i64> {
    if ch.is_ascii_digit() {
        return Some((ch as u8 - b'0') as i64);
    }
    if ('０'..='９').contains(&ch) {
        return Some((ch as u32 - '０' as u32) as i64);
    }
    None
}

// 将页码文本归一为整数，meta 字符串允许符号位。
fn parse_page_number(page_part: &str, allow_sign: bool) -> Option<i64> {
    let mut page_part = page_part.trim();
    let mut sign = 1_i64;

    if allow_sign {
        if let Some(rest) = page_part.strip_prefix('-') {
            sign = -1;
            page_part = rest.trim_start();
        } else if let Some(rest) = page_part.strip_prefix('+') {
            page_part = rest.trim_start();
        }
    }

    if page_part.is_empty() {
        return None;
    }

    let mut page = 0_i64;
    for ch in page_part.chars() {
        let digit = decimal_digit_value(ch)?;
        page = page.checked_mul(10)?.checked_add(digit)?;
    }

    page.checked_mul(sign)
}

// 解析单个括号内的 source:Ppage 引用体。
fn parse_citation_body(body: &str) -> Option<Citation> {
    let (colon_idx, colon_ch) = body.char_indices().find(|(_, ch)| is_colon(*ch))?;
    let source = body[..colon_idx].trim();
    if source.is_empty() {
        return None;
    }

    let mut page_part = body[colon_idx + colon_ch.len_utf8()..].trim();
    if let Some(first) = page_part.chars().next()
        && is_page_prefix(first)
    {
        page_part = page_part[first.len_utf8()..].trim();
    }

    let page = parse_page_number(page_part, false)?;
    Some(Citation {
        source: source.to_string(),
        page,
    })
}

// 从答案中提取所有形如 [source:Ppage] 的引用标签。
fn extract_citations(answer: &str) -> Vec<Citation> {
    let mut citations = Vec::new();
    let mut search_from = 0;

    while let Some((open_offset, open_ch)) = answer[search_from..]
        .char_indices()
        .find(|(_, ch)| is_open_bracket(*ch))
    {
        let body_start = search_from + open_offset + open_ch.len_utf8();
        let Some((close_offset, _)) = answer[body_start..]
            .char_indices()
            .find(|(_, ch)| is_close_bracket(*ch))
        else {
            break;
        };

        let body_end = body_start + close_offset;
        if let Some(citation) = parse_citation_body(&answer[body_start..body_end]) {
            citations.push(citation);
        }
        search_from = body_end + answer[body_end..].chars().next().map_or(0, char::len_utf8);
    }

    citations
}

// Python source 只接受非空字符串。
fn py_to_source(value: Bound<'_, PyAny>) -> Option<String> {
    if let Ok(value) = value.extract::<String>()
        && !value.is_empty()
    {
        return Some(value);
    }
    None
}

// Python float 页码按 int(page) 语义截断。
fn truncate_f64_to_i64(value: f64) -> Option<i64> {
    if !value.is_finite() {
        return None;
    }

    let truncated = value.trunc();
    if truncated < i64::MIN as f64 || truncated > i64::MAX as f64 {
        return None;
    }
    Some(truncated as i64)
}

// Python 页码兼容 int、str、float 和实现 __float__ 的对象。
fn py_to_page(value: Bound<'_, PyAny>) -> Option<i64> {
    if let Ok(value) = value.extract::<i64>() {
        return Some(value);
    }
    if let Ok(value) = value.extract::<u64>() {
        return i64::try_from(value).ok();
    }
    if value.is_instance_of::<PyFloat>() {
        let value = value.extract::<f64>().ok()?;
        return truncate_f64_to_i64(value);
    }
    if let Ok(value) = value.extract::<String>() {
        return parse_page_number(&value, true);
    }
    if let Ok(value) = value.extract::<f64>() {
        return truncate_f64_to_i64(value);
    }
    None
}

// 构建本轮检索实际允许引用的 source-page 注册表。
fn build_allowed_registry(
    valid_docs: Vec<Bound<'_, PyDict>>,
) -> PyResult<HashMap<String, BTreeSet<i64>>> {
    let mut allowed_registry: HashMap<String, BTreeSet<i64>> = HashMap::new();

    for doc in valid_docs {
        let Some(meta) = doc.get_item("meta")? else {
            continue;
        };
        let Ok(meta) = meta.cast_into::<PyDict>() else {
            continue;
        };

        let Some(source) = meta.get_item("source")?.and_then(py_to_source) else {
            continue;
        };
        let Some(page) = meta.get_item("page")?.and_then(py_to_page) else {
            continue;
        };

        allowed_registry.entry(source).or_default().insert(page);
    }

    Ok(allowed_registry)
}

// Rust 端页码集合按排序后的 Python list 返回。
fn pages_to_py_list<'py>(py: Python<'py>, pages: &BTreeSet<i64>) -> PyResult<Bound<'py, PyList>> {
    let py_pages = PyList::empty(py);
    for page in pages {
        py_pages.append(*page)?;
    }
    Ok(py_pages)
}

// 返回 Python 门面生成 critique 所需的结构化校验结果。
pub fn validate_citations_core<'py>(
    py: Python<'py>,
    answer: String,
    valid_docs: Vec<Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    let invalid_sources = PyList::empty(py);
    let invalid_pages = PyList::empty(py);

    result.set_item("missing_citations", false)?;
    result.set_item("invalid_sources", &invalid_sources)?;
    result.set_item("invalid_pages", &invalid_pages)?;

    if answer.is_empty() {
        result.set_item("is_valid", true)?;
        return Ok(result);
    }

    let allowed_registry = build_allowed_registry(valid_docs)?;
    if allowed_registry.is_empty() || answer.contains(FALLBACK_MARKER) {
        result.set_item("is_valid", true)?;
        return Ok(result);
    }

    let citations = extract_citations(&answer);
    if citations.is_empty() {
        result.set_item("missing_citations", true)?;
        result.set_item("is_valid", false)?;
        return Ok(result);
    }

    for citation in citations {
        let Some(valid_pages) = allowed_registry.get(&citation.source) else {
            let item = PyDict::new(py);
            item.set_item("source", citation.source)?;
            item.set_item("page", citation.page)?;
            invalid_sources.append(item)?;
            continue;
        };

        if !valid_pages.contains(&citation.page) {
            let item = PyDict::new(py);
            item.set_item("source", citation.source)?;
            item.set_item("page", citation.page)?;
            item.set_item("valid_pages", pages_to_py_list(py, valid_pages)?)?;
            invalid_pages.append(item)?;
        }
    }

    result.set_item(
        "is_valid",
        invalid_sources.is_empty() && invalid_pages.is_empty(),
    )?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{Citation, extract_citations};

    // 验证半角引用标签可以被提取。
    #[test]
    fn extracts_half_width_citation() {
        assert_eq!(
            extract_citations("依据见 [ a.pdf : p6 ] 。"),
            vec![Citation {
                source: "a.pdf".to_string(),
                page: 6,
            }]
        );
    }

    // 验证全角引用标签可以被提取。
    #[test]
    fn extracts_full_width_citation() {
        assert_eq!(
            extract_citations("方法见原文［a.pdf：P5］。"),
            vec![Citation {
                source: "a.pdf".to_string(),
                page: 5,
            }]
        );
    }

    // 验证全角数字页码可以被提取。
    #[test]
    fn extracts_full_width_digit_citation() {
        assert_eq!(
            extract_citations("方法见原文［a.pdf：P５］。"),
            vec![Citation {
                source: "a.pdf".to_string(),
                page: 5,
            }]
        );
    }

    // 验证格式错误的引用标签会被忽略。
    #[test]
    fn ignores_malformed_citation_body() {
        assert!(extract_citations("不是引用[a.pdf:P五]。").is_empty());
    }
}
