import sqlite3

from cogdoc.api.persistence import SqliteSessionStore
from cogdoc.api.session_store import SessionStore
from cogdoc.memory.manager import MemoryPolicy, update_mid_term


# 构造紧凑测试策略。
def _policy() -> MemoryPolicy:
    return MemoryPolicy(
        short_term_message_limit=4,
        short_term_char_limit=1000,
        mid_term_char_limit=1000,
        long_term_fact_limit=4,
        context_long_term_limit=4,
    )


# 验证短期淘汰内容进入中期记忆。
def test_short_term_compaction_archives_mid_term_memory():
    store = SessionStore(memory_policy=_policy())
    for index in range(4):
        turn = [
            {"role": "user", "content": f"问题{index}"},
            {"role": "assistant", "content": f"回答{index}"},
        ]
        store.record("kb", "s1", turn, turn)

    snapshot = store.get_memory_snapshot("kb", "s1")

    assert [item["content"] for item in snapshot["short_term"]] == [
        "问题2",
        "回答2",
        "问题3",
        "回答3",
    ]
    assert snapshot["mid_term"]["archived_messages"] == 4
    assert "用户: 问题0" in snapshot["mid_term"]["summary"]


# 验证普通请求不会晋升长期记忆。
def test_long_term_memory_requires_strong_signal():
    store = SessionStore(memory_policy=_policy())
    store.record("kb", "s1", [], [{"role": "user", "content": "今天用 Python 写脚本"}])

    assert store.get_memory_snapshot("kb", "s1")["long_term"] == []


# 验证显式目标和决策立即进入中期记忆。
def test_mid_term_goal_and_decision_are_recorded_immediately():
    store = SessionStore(memory_policy=_policy())
    turn = [
        {"role": "user", "content": "当前目标是完成检索模块，决定采用 Qdrant"},
        {"role": "assistant", "content": "收到"},
    ]
    store.record("kb", "s1", turn, turn)

    mid_term = store.get_memory_snapshot("kb", "s1")["mid_term"]

    assert mid_term["goals"] == ["完成检索模块"]
    assert mid_term["decisions"] == ["采用 Qdrant"]


# 验证长期记忆注入可独立关闭。
def test_long_term_context_injection_can_be_disabled():
    policy = MemoryPolicy(context_long_term_limit=0)
    store = SessionStore(memory_policy=policy)
    store.record("kb", "s1", [], [{"role": "user", "content": "请记住：默认使用中文"}])

    assert store.get_history("kb", "new-session") == []
    facts = store.get_memory_snapshot("kb", "s1")["long_term"]
    assert [fact["content"] for fact in facts] == ["默认使用中文"]


# 验证长期记忆跨会话复用并支持清除。
def test_long_term_memory_crosses_sessions_and_can_be_cleared():
    store = SessionStore(memory_policy=_policy())
    store.record("kb", "s1", [], [{"role": "user", "content": "请记住：我偏好 Rust"}])

    history = store.get_history("kb", "new-session")

    assert history[0]["role"] == "memory"
    assert "【长期记忆】" in history[0]["content"]
    assert "我偏好 Rust" in history[0]["content"]
    store.clear_long_term("kb")
    assert store.get_history("kb", "new-session") == []


# 验证内存版按重要性和新近性淘汰长期记忆。
def test_in_memory_long_term_eviction_uses_importance_and_recency():
    policy = MemoryPolicy(long_term_fact_limit=2, context_long_term_limit=2)
    store = SessionStore(memory_policy=policy)
    store.record("kb", "s1", [], [{"role": "user", "content": "请记住：高优先级事实"}])
    store.record("kb", "s1", [], [{"role": "user", "content": "我偏好旧方案"}])
    store.record("kb", "s1", [], [{"role": "user", "content": "我偏好新方案"}])

    facts = store.get_memory_snapshot("kb", "s1")["long_term"]

    assert [fact["content"] for fact in facts] == ["高优先级事实", "新方案"]


# 验证中期摘要连续淘汰后不超过字符预算。
def test_mid_term_summary_respects_character_budget():
    policy = MemoryPolicy(mid_term_char_limit=100, message_preview_chars=80)
    messages = [
        {"role": "user", "content": f"第{index}条" + "内容" * 20} for index in range(6)
    ]

    mid_term = update_mid_term({}, messages, policy)

    assert sum(len(item) for item in mid_term["summary"]) <= 100
    assert "第5条" in mid_term["summary"][-1]
    assert all("第0条" not in item for item in mid_term["summary"])


# 验证旧数据库自动迁移分层记忆字段。
def test_sqlite_store_migrates_legacy_session_table(tmp_path):
    db_path = tmp_path / "state.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE sessions (doc_id TEXT, session_id TEXT, memory TEXT, "
        "display TEXT, updated_at REAL, PRIMARY KEY (doc_id, session_id))"
    )
    connection.commit()
    connection.close()

    store = SqliteSessionStore(str(db_path), memory_policy=_policy())
    store.record("kb", "s1", [], [{"role": "user", "content": "以后都用中文回答"}])

    snapshot = store.get_memory_snapshot("kb", "s1")
    assert snapshot["long_term"][0]["content"] == "用中文回答"


# 验证 SQLite 分层记忆可跨实例恢复。
def test_sqlite_layered_memory_survives_restart(tmp_path):
    db_path = str(tmp_path / "state.db")
    store = SqliteSessionStore(db_path, memory_policy=_policy())
    for index in range(3):
        turn = [
            {"role": "user", "content": f"问题{index}"},
            {"role": "assistant", "content": f"回答{index}"},
        ]
        store.record("kb", "s1", turn, turn)
    store.record("kb", "s1", [], [{"role": "user", "content": "项目采用 PostgreSQL"}])

    reopened = SqliteSessionStore(db_path, memory_policy=_policy())
    snapshot = reopened.get_memory_snapshot("kb", "s1")

    assert len(snapshot["short_term"]) == 4
    assert snapshot["mid_term"]["archived_messages"] == 2
    assert snapshot["long_term"][0]["content"] == "PostgreSQL"


# 验证 SQLite 长期记忆索引与淘汰顺序。
def test_sqlite_long_term_memory_uses_order_index(tmp_path):
    policy = MemoryPolicy(long_term_fact_limit=2, context_long_term_limit=2)
    store = SqliteSessionStore(str(tmp_path / "state.db"), memory_policy=policy)
    store.record("kb", "s1", [], [{"role": "user", "content": "请记住：高优先级事实"}])
    store.record("kb", "s1", [], [{"role": "user", "content": "我偏好旧方案"}])
    store.record("kb", "s1", [], [{"role": "user", "content": "我偏好新方案"}])

    facts = store.get_memory_snapshot("kb", "s1")["long_term"]
    indexes = store._conn.execute("PRAGMA index_list(long_memories)").fetchall()

    assert [fact["content"] for fact in facts] == ["高优先级事实", "新方案"]
    assert "idx_long_memories_order" in {row[1] for row in indexes}


# 验证 SQLite 使用当前问题执行长期记忆召回。
def test_sqlite_query_aware_long_term_retrieval(tmp_path):
    policy = MemoryPolicy(
        context_long_term_limit=1,
        memory_semantic_enabled=False,
        memory_retrieval_mid_limit=0,
    )
    store = SqliteSessionStore(str(tmp_path / "state.db"), memory_policy=policy)
    store.record(
        "kb", "source", [], [{"role": "user", "content": "请记住：默认使用中文"}]
    )
    store.record(
        "kb",
        "source",
        [],
        [{"role": "user", "content": "我偏好 PostgreSQL 数据库"}],
    )

    context = store.get_history("kb", "target", "PostgreSQL 怎么配置")

    assert "PostgreSQL" in context[0]["content"]
    assert "默认使用中文" not in context[0]["content"]
