use jieba_rs::Jieba;
use regex::Regex;
use std::sync::OnceLock;

// Jieba 词典加载一次即可复用；OnceLock 保证多线程下只初始化一遍。
static JIEBA: OnceLock<Jieba> = OnceLock::new();
// 英文/数字/技术符号片段，与 Python 端 r'[a-z0-9_\-\.]+' 完全一致。
static EN_RE: OnceLock<Regex> = OnceLock::new();

fn jieba() -> &'static Jieba {
    JIEBA.get_or_init(Jieba::new)
}

fn en_re() -> &'static Regex {
    EN_RE.get_or_init(|| Regex::new(r"[a-z0-9_\-.]+").unwrap())
}

// 与 Python 分词规则对齐，长度判定必须按字符数而不是字节。
pub fn tokenize_mixed_text_core(text: &str) -> Vec<String> {
    let lowered = text.to_lowercase();
    let mut tokens: Vec<String> = Vec::new();

    // 英文、数字、常见技术符号：保留字符数 > 1 的片段。
    for m in en_re().find_iter(&lowered) {
        let word = m.as_str();
        if word.chars().count() > 1 {
            tokens.push(word.to_string());
        }
    }

    // 英文片段先替换为空格，避免污染中文分词。
    let chinese_pure = en_re().replace_all(&lowered, " ");
    for token in jieba().cut(&chinese_pure, true) {
        let trimmed = token.word.trim();
        if !trimmed.is_empty() && trimmed.chars().count() > 1 {
            tokens.push(trimmed.to_string());
        }
    }

    tokens
}
