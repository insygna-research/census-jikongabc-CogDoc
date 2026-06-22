from run import parse_forced_mode


def test_parse_forced_mode_reads_explicit_prefixes():
    assert parse_forced_mode("/qa 这篇论文的方法是什么") == (
        "qa",
        "这篇论文的方法是什么",
    )
    assert parse_forced_mode("/summary 总结 a.pdf") == ("summary", "总结 a.pdf")
    assert parse_forced_mode("/compare 对比 a.pdf 和 b.pdf") == (
        "compare",
        "对比 a.pdf 和 b.pdf",
    )


def test_parse_forced_mode_preserves_query_without_prefix():
    assert parse_forced_mode("总结 a.pdf") == (None, "总结 a.pdf")


def test_parse_forced_mode_allows_empty_prefixed_query_for_cli_prompt():
    assert parse_forced_mode("/summary") == ("summary", "")
    assert parse_forced_mode("/compare   ") == ("compare", "")
