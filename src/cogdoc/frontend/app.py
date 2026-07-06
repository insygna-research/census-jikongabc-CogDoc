import os
import queue
import threading
import time
import uuid
from collections.abc import Mapping
import streamlit as st
from cogdoc.frontend.api_client import (
    CogDocAPIError,
    CogDocClient,
    format_api_error,
    response_payload,
)

DEFAULT_API_URL = os.getenv("COGDOC_API_URL", "http://localhost:8000")
MAIN_VIEWS = ["对话", "调试"]
STREAM_RERUN_INTERVAL_SECONDS = 0.8
STREAM_PREVIEW_HEAD_CHARS = 1200
STREAM_PREVIEW_TAIL_CHARS = 3600
SIDEBAR_CACHE_TTL_SECONDS = 2.0
SIDEBAR_STREAM_CACHE_TTL_SECONDS = 30.0
SIDEBAR_STALE_CACHE_GRACE_SECONDS = 120.0
TRACE_NODE_LABELS = {
    "runtime.setup": "运行准备",
    "intent_router": "意图路由",
    "rewrite_node": "问题改写",
    "verify_rewrite_node": "改写校验",
    "retrieve_node": "召回检索",
    "rerank_node": "重排",
    "generate_node": "答案生成",
    "citation_node": "引用校验",
    "qa_subgraph": "问答流程",
    "summary_subgraph": "摘要流程",
    "compare_subgraph": "对比流程",
}

st.set_page_config(page_title="CogDoc", layout="wide")


# 创建测试客户端。
def _client() -> CogDocClient:
    return CogDocClient(st.session_state.api_url)


# 处理响应错误。
def _response_error(response, fallback: str = "请求失败") -> str:
    return format_api_error(response_payload(response), response.status_code, fallback)


# 提取响应状态与载荷。
def _response_status_payload(response) -> tuple[int, object]:
    return response.status_code, response_payload(response)


# 完成 init状态 处理。
def _init_state() -> None:
    st.session_state.setdefault("api_url", DEFAULT_API_URL)
    # session_id 持久化进 URL，刷新后复用同一会话（后端多轮记忆得以续上）。
    if "session_id" not in st.session_state:
        st.session_state.session_id = st.query_params.get("sid") or uuid.uuid4().hex
        st.query_params["sid"] = st.session_state.session_id
    st.session_state.setdefault("kb_id", None)
    st.session_state.setdefault("msg_seq", 0)
    st.session_state.setdefault("messages_by_context", {})
    st.session_state.setdefault("restored_contexts", set())
    st.session_state.setdefault("pending_streams", {})
    st.session_state.setdefault("pending_retrieve_debugs", {})
    st.session_state.setdefault("api_cache", {})
    st.session_state.setdefault("main_views_by_context", {})
    st.session_state.setdefault("trace_cache", {})
    st.session_state.setdefault("active_trace_id", "")
    st.session_state.setdefault("trace_labels", {})
    st.session_state.setdefault("trace_options_by_id", {})
    st.session_state.setdefault("trace_session_items_by_context", {})
    st.session_state.setdefault("trace_session_loaded", set())
    st.session_state.setdefault("trace_session_error", {})
    st.session_state.setdefault("retrieve_debug_by_context", {})
    # 兼容旧状态：升级前只有一份全局 messages，迁移到当前 (kb, session) 桶里。
    if "messages" in st.session_state:
        if st.session_state.kb_id and st.session_state.messages:
            st.session_state.messages_by_context.setdefault(
                _context_key(st.session_state.kb_id, st.session_state.session_id),
                st.session_state.messages,
            )
        st.session_state.pop("messages", None)
    for legacy_key in ("restored_for", "answering", "pending_prompt", "pending_mode"):
        st.session_state.pop(legacy_key, None)
    # 本会话内已知的对话 id（按 kb），与后端列表合并，保证空/新对话也留得住、点得到。
    st.session_state.setdefault("known_sessions", {})


# 判断是否存在未完成流式请求。
def _has_pending_stream() -> bool:
    has_stream = any(
        not pending.get("done")
        for pending in st.session_state.pending_streams.values()
        if isinstance(pending, Mapping)
    )
    has_retrieve = any(
        not pending.get("done")
        for pending in st.session_state.pending_retrieve_debugs.values()
        if isinstance(pending, Mapping)
    )
    return has_stream or has_retrieve


# 返回侧栏缓存时长。
def _sidebar_cache_ttl() -> float:
    return (
        SIDEBAR_STREAM_CACHE_TTL_SECONDS
        if _has_pending_stream()
        else SIDEBAR_CACHE_TTL_SECONDS
    )


# 读取带 TTL 的 API 缓存。
def _cached_api_value(key: tuple, loader):
    cache = st.session_state.api_cache
    now = time.monotonic()
    entry = cache.get(key)
    if entry and now - entry["time"] <= _sidebar_cache_ttl():
        return entry["value"]
    try:
        value = loader()
    except Exception:
        if entry and now - entry["time"] <= SIDEBAR_STALE_CACHE_GRACE_SECONDS:
            return entry["value"]
        raise
    cache[key] = {"time": now, "value": value}
    return value


# 清理 API 缓存。
def _clear_api_cache(prefix: tuple | None = None) -> None:
    if prefix is None:
        st.session_state.api_cache.clear()
        return
    for key in list(st.session_state.api_cache):
        if key[: len(prefix)] == prefix:
            st.session_state.api_cache.pop(key, None)


# 处理context键。
def _context_key(kb_id: str, session_id: str | None = None) -> tuple[str, str]:
    return (kb_id, session_id or st.session_state.session_id)


# 处理消息FOR。
def _messages_for(kb_id: str, session_id: str | None = None) -> list[dict]:
    return st.session_state.messages_by_context.setdefault(
        _context_key(kb_id, session_id), []
    )


# 处理消息from历史。
def _message_from_history(turn: Mapping, fallback_query: str = "") -> dict:
    metadata = turn.get("metadata") if isinstance(turn.get("metadata"), Mapping) else {}
    trace_id = turn.get("trace_id") or metadata.get("trace_id")
    query = turn.get("query") or metadata.get("query") or fallback_query
    if trace_id and query:
        st.session_state.trace_labels[str(trace_id)] = str(query)
    msg = {
        "role": turn.get("role", "assistant"),
        "content": turn.get("content", ""),
        "id": _next_id(),
    }
    if trace_id:
        msg["final"] = {
            "trace_id": trace_id,
            "task_type": turn.get("task_type") or metadata.get("task_type") or "-",
            "is_valid": True,
        }
        msg["query"] = query
    return msg


# 处理消息from历史。
def _messages_from_history(turns: list[Mapping]) -> list[dict]:
    messages = []
    last_user_query = ""
    for turn in turns:
        if turn.get("role") == "user":
            last_user_query = str(turn.get("content") or "")
            messages.append(_message_from_history(turn))
        else:
            messages.append(_message_from_history(turn, fallback_query=last_user_query))
    return messages


# 恢复历史记录。
def _restore_history(kb_id: str) -> None:
    # kb 或 session 变化时重载该 kb 的历史；同一 (kb, session) 内不重复拉，保留实时追加的消息。
    marker = _context_key(kb_id)
    if marker in st.session_state.restored_contexts:
        return
    if st.session_state.messages_by_context.get(marker):
        st.session_state.restored_contexts.add(marker)
        return
    try:
        resp = _client().get_session_history(st.session_state.session_id, kb_id)
        turns = resp.json().get("messages", []) if resp.status_code == 200 else []
    except Exception:
        turns = []
    st.session_state.messages_by_context[marker] = _messages_from_history(turns)
    st.session_state.restored_contexts.add(marker)


# 构造label。
def _page_label(page) -> str:
    # page 可能为空（schema 默认 None），避免渲染成 "PNone"。
    return f" · P{page}" if page is not None else ""


# 切换会话。
def _switch_session(session_id: str) -> None:
    # 切换/新建对话：换 session_id（同步 URL）；消息按 (kb, session) 分桶，不清其他对话。
    st.session_state.session_id = session_id
    st.query_params["sid"] = session_id
    st.rerun()


# 完成 conversations 处理。
def _conversations(client: CogDocClient, kb_id: str) -> None:
    # 多对话列表：前端已知会话 ∪ 后端已存会话，新建/切换/删除，全部可点。
    st.subheader("对话")
    current = st.session_state.session_id
    known = st.session_state.known_sessions.setdefault(kb_id, [])
    if current not in known:
        known.insert(0, current)

    if st.button("➕ 新对话", use_container_width=True):
        new_id = uuid.uuid4().hex
        st.session_state.known_sessions.setdefault(kb_id, []).insert(0, new_id)
        _switch_session(new_id)

    try:
        status_code, payload = _cached_api_value(
            ("sessions", client.base_url, kb_id),
            lambda: _response_status_payload(client.list_sessions(kb_id)),
        )
        if status_code == 200:
            sessions = (
                payload.get("sessions", []) if isinstance(payload, Mapping) else []
            )
            backend = {
                s["session_id"]: s
                for s in sessions
                if isinstance(s, Mapping) and isinstance(s.get("session_id"), str)
            }
        else:
            st.warning(
                "读取会话列表失败: "
                f"{format_api_error(payload, status_code, '读取会话列表失败')}"
            )
            backend = {}
    except Exception as exc:
        st.warning(f"读取会话列表失败: {exc}")
        backend = {}

    # 已知列表打底（含空对话），再补上后端有、前端没记的（如刷新后从别处恢复的）。
    ordered = list(known)
    for sid in backend:
        if sid not in ordered:
            ordered.append(sid)

    for sid in ordered:
        title = backend.get(sid, {}).get("title") or "新对话"
        mark = "🟢 " if sid == current else ""
        row = st.columns([5, 1])
        if row[0].button(f"{mark}{title}", key=f"sess-{sid}", use_container_width=True):
            _switch_session(sid)
        if row[1].button("🗑", key=f"sessdel-{sid}"):
            client.delete_session(sid, kb_id)
            _clear_api_cache(("sessions", client.base_url, kb_id))
            if sid in known:
                known.remove(sid)
            if sid == current:
                _switch_session(uuid.uuid4().hex)
            st.rerun()


# 完成 send反馈 处理。
def _send_feedback(final: dict, query: str, feedback: str) -> None:
    # 凭该回答的 trace_id 提交赞/踩，关联 kb/query/answer 落到后端。
    trace_id = final.get("trace_id")
    if not trace_id:
        st.toast("缺少 trace_id，无法提交反馈")
        return
    resp = _client().submit_feedback(
        trace_id=trace_id,
        feedback=feedback,
        kb_id=st.session_state.kb_id,
        query=query,
        answer=final.get("answer", ""),
    )
    st.toast(
        "反馈已记录" if resp.status_code == 201 else f"反馈失败: {resp.status_code}"
    )


# 加载跟踪。
def _load_trace(trace_id: str, force: bool = False) -> None:
    trace_id = trace_id.strip()
    if not trace_id:
        return
    if force:
        st.session_state.trace_cache.pop(trace_id, None)
    elif trace_id in st.session_state.trace_cache:
        return
    try:
        resp = _client().get_trace(trace_id)
        if resp.status_code == 200:
            payload = response_payload(resp)
            st.session_state.trace_cache[trace_id] = payload
            if isinstance(payload, Mapping):
                query = (payload.get("config") or {}).get("query_preview")
                if query:
                    st.session_state.trace_labels[trace_id] = str(query)
        else:
            st.session_state.trace_cache[trace_id] = {
                "error": _response_error(resp, "读取 trace 失败")
            }
    except Exception as exc:
        st.session_state.trace_cache[trace_id] = {"error": str(exc)}


# 加载会话跟踪。
def _load_session_traces(kb_id: str | None, force: bool = False) -> None:
    if not kb_id:
        return
    marker = _context_key(kb_id)
    if not force and marker in st.session_state.trace_session_loaded:
        return
    try:
        resp = _client().list_traces(
            limit=30,
            kb_id=kb_id,
            session_id=st.session_state.session_id,
        )
        if resp.status_code != 200:
            st.session_state.trace_session_error[marker] = _response_error(
                resp, "读取当前会话 trace 失败"
            )
            st.session_state.trace_session_items_by_context[marker] = []
            st.session_state.trace_session_loaded.add(marker)
            return
        payload = response_payload(resp)
        traces = payload.get("traces", []) if isinstance(payload, Mapping) else []
        items = [
            trace
            for trace in traces
            if isinstance(trace, Mapping) and isinstance(trace.get("trace_id"), str)
        ]
        st.session_state.trace_session_items_by_context[marker] = items
        st.session_state.trace_session_error.pop(marker, None)
        st.session_state.trace_session_loaded.add(marker)
        for trace in items:
            query = str(trace.get("query_preview") or "").strip()
            if query:
                st.session_state.trace_labels[trace["trace_id"]] = query
    except Exception as exc:
        st.session_state.trace_session_error[marker] = str(exc)
        st.session_state.trace_session_items_by_context[marker] = []
        st.session_state.trace_session_loaded.add(marker)


# 处理跟踪optionlabel。
def _trace_option_label(trace_id: str) -> str:
    if not trace_id:
        return "选择最近 trace"
    option_trace = st.session_state.trace_options_by_id.get(trace_id)
    if option_trace:
        status = option_trace.get("status", "-")
        task = option_trace.get("task_type", "-")
        title = _trace_query_title(trace_id, option_trace)
        suffix = _format_duration(option_trace.get("duration_ms"))
        duration = f" · {suffix}" if suffix else ""
        return f"{title} · {task} · {status}{duration}"
    return _trace_query_title(trace_id, {})


# 处理跟踪查询title。
def _trace_query_title(trace_id: str, trace: Mapping) -> str:
    query = str(trace.get("query_preview") or "").strip()
    if not query:
        query = str(st.session_state.trace_labels.get(trace_id, "")).strip()
    return query or "未记录问题"


# 格式化耗时。
def _format_duration(duration_ms) -> str:
    if duration_ms is None:
        return ""
    try:
        value = float(duration_ms)
    except (TypeError, ValueError):
        return str(duration_ms)
    if value >= 1000:
        return f"{value / 1000:.1f}s"
    return f"{value:.0f} ms"


# 格式化页码范围。
def _page_range_label(page_start, page_end=None) -> str:
    if page_start is None and page_end is None:
        return ""
    if page_end is None or page_start == page_end:
        return f"P{page_start}"
    if page_start is None:
        return f"P{page_end}"
    return f"P{page_start}-{page_end}"


# 构建流式预览文本。
def _stream_preview(answer: str | None) -> str:
    answer = str(answer or "")
    if len(answer) <= STREAM_PREVIEW_HEAD_CHARS + STREAM_PREVIEW_TAIL_CHARS:
        return answer + "▌" if answer else "正在思考…"
    omitted = len(answer) - STREAM_PREVIEW_HEAD_CHARS - STREAM_PREVIEW_TAIL_CHARS
    return (
        answer[:STREAM_PREVIEW_HEAD_CHARS]
        + f"\n\n... 已生成 {len(answer)} 字，暂折叠中间 {omitted} 字，完成后显示全文 ...\n\n"
        + answer[-STREAM_PREVIEW_TAIL_CHARS:]
        + "▌"
    )


# 处理current跟踪items。
def _current_trace_items(kb_id: str | None) -> list[dict]:
    items = []
    seen = set()
    for msg in reversed(_messages_for(kb_id) if kb_id else []):
        final = msg.get("final") or {}
        trace_id = final.get("trace_id")
        if not trace_id or trace_id in seen:
            continue
        seen.add(trace_id)
        cached = st.session_state.trace_cache.get(str(trace_id))
        item = {"trace_id": str(trace_id)}
        if isinstance(cached, Mapping) and not isinstance(cached.get("error"), str):
            item.update(
                {
                    "query_preview": (cached.get("config") or {}).get("query_preview"),
                    "task_type": cached.get("task_type"),
                    "status": cached.get("status"),
                    "duration_ms": cached.get("duration_ms"),
                }
            )
        query = msg.get("query") or st.session_state.trace_labels.get(trace_id, "")
        if query and not item.get("query_preview"):
            item["query_preview"] = str(query)
        if final.get("task_type") and not item.get("task_type"):
            item["task_type"] = final.get("task_type")
        items.append(item)
    if kb_id:
        marker = _context_key(kb_id)
        for trace in st.session_state.trace_session_items_by_context.get(marker, []):
            trace_id = trace.get("trace_id")
            if not trace_id or trace_id in seen:
                continue
            seen.add(trace_id)
            items.append(dict(trace))
    return items


# 处理跟踪node键。
def _trace_node_key(node_name: str) -> str:
    tail = (node_name or "").rsplit(".", 1)[-1]
    if ":" in tail:
        return tail.split(":", 1)[0]
    return tail


# 处理跟踪steplabel。
def _trace_step_label(step: Mapping, idx: int) -> str:
    node_name = str(step.get("node_name") or f"step-{idx + 1}")
    node_key = _trace_node_key(node_name)
    title = TRACE_NODE_LABELS.get(node_key, node_key)
    duration = step.get("duration_ms")
    label = f"{idx + 1}. {title}"
    if node_key != title:
        label += f" · {node_key}"
    formatted = _format_duration(duration)
    if formatted:
        label += f" · {formatted}"
    return label


# 渲染跟踪step。
def _render_trace_step(step: Mapping, idx: int) -> None:
    with st.expander(_trace_step_label(step, idx)):
        if step.get("node_name"):
            st.caption(f"原始节点: {step.get('node_name')}")
        details = []
        if step.get("task_type"):
            details.append(f"任务: {step.get('task_type')}")
        if step.get("model"):
            details.append(f"模型: {step.get('model')}")
        if step.get("retrieval_top_k") is not None:
            details.append(f"top_k: {step.get('retrieval_top_k')}")
        if step.get("error_class"):
            details.append(f"错误: {step.get('error_class')}")
        if details:
            cols = st.columns(len(details))
            for col, text in zip(cols, details):
                col.caption(text)

        if step.get("router_reason"):
            st.markdown("**路由理由**")
            st.write(step["router_reason"])
        rewritten = step.get("rewritten_queries") or []
        if rewritten:
            st.markdown("**改写查询**")
            for query in rewritten:
                st.write(f"- {query}")
        elif (step.get("counts") or {}).get("rewritten_query_count"):
            st.caption("此旧 trace 只记录了改写数量，未保存具体改写查询。")
        if step.get("critique"):
            st.markdown("**校验反馈**")
            st.write(step["critique"])
        counts = step.get("counts") or {}
        if counts:
            st.markdown("**计数**")
            st.json(counts)
        evidence = step.get("evidence") or []
        if evidence:
            st.markdown("**证据预览**")
            for item in evidence:
                source = item.get("source", "")
                chunk_id = item.get("chunk_id", "")
                st.caption(f"{source}{_page_label(item.get('page'))} · `{chunk_id}`")
                st.write(item.get("text_preview", ""))


# 渲染跟踪调试。
def _render_trace_debug(trace: dict) -> None:
    trace_error = trace.get("error")
    if isinstance(trace_error, str):
        st.error(trace["error"])
        return
    summary = trace.get("summary") or {}
    meta = st.columns(4)
    meta[0].caption(f"状态: {trace.get('status') or '-'}")
    meta[1].caption(f"任务: {trace.get('task_type') or '-'}")
    meta[2].caption(f"耗时: {_format_duration(trace.get('duration_ms')) or '-'}")
    meta[3].caption(f"步骤: {summary.get('step_count', 0)}")
    if trace.get("config"):
        with st.expander("请求配置"):
            st.json(trace["config"])
    if trace_error:
        with st.expander("运行错误", expanded=True):
            st.json(trace_error)
    for idx, step in enumerate(trace.get("steps") or []):
        if isinstance(step, Mapping):
            _render_trace_step(step, idx)


# 渲染跟踪lookup。
def _render_trace_lookup(kb_id: str | None) -> None:
    st.subheader("Trace 调试")
    _load_session_traces(kb_id)
    left, right = st.columns([1, 2])
    with left:
        traces = _current_trace_items(kb_id)
        trace_ids = {trace["trace_id"] for trace in traces}
        if st.session_state.active_trace_id not in trace_ids:
            st.session_state.active_trace_id = ""
        marker = _context_key(kb_id) if kb_id else None
        if marker and st.session_state.trace_session_error.get(marker):
            st.warning(st.session_state.trace_session_error[marker])
        if not traces:
            st.info("当前对话还没有 trace。发送问题后这里会显示。")
        if traces:
            st.session_state.trace_options_by_id = {
                trace["trace_id"]: trace for trace in traces
            }
            status_options = ["全部"] + sorted(
                {str(trace.get("status") or "-") for trace in traces}
            )
            task_options = ["全部"] + sorted(
                {str(trace.get("task_type") or "-") for trace in traces}
            )
            status_filter = st.selectbox(
                "状态", status_options, key="trace-status-filter"
            )
            task_filter = st.selectbox("任务", task_options, key="trace-task-filter")
            filtered = [
                trace
                for trace in traces
                if (status_filter == "全部" or trace.get("status") == status_filter)
                and (task_filter == "全部" or trace.get("task_type") == task_filter)
            ]
            options = [""] + [trace["trace_id"] for trace in filtered]
            selected = st.selectbox(
                "当前对话请求",
                options,
                format_func=_trace_option_label,
                key="trace_recent_select",
            )
            if selected and st.button(
                "打开选中 trace", key="trace-open-selected", use_container_width=True
            ):
                st.session_state.active_trace_id = selected
                _load_trace(selected)

    with right:
        active = st.session_state.active_trace_id
        if not active:
            st.info("选择当前对话中的请求查看 trace。")
            return
        top = st.columns([3, 1])
        top[0].markdown(f"**{_trace_query_title(active, {})}**")
        top[0].caption(f"trace_id: {active}")
        if top[1].button("刷新 trace", key="trace-refresh", use_container_width=True):
            _load_session_traces(kb_id, force=True)
            _load_trace(active, force=True)
        if active in st.session_state.trace_cache:
            _render_trace_debug(st.session_state.trace_cache[active])


# 渲染 evidence。
def _render_evidence(final: dict, key: str, query: str = "") -> None:
    # 渲染一条回答的元信息 + 引用/证据面板 + 赞踩按钮（消费结构化字段）。
    meta = st.columns(3)
    meta[0].caption(f"任务: {final.get('task_type', '-')}")
    meta[1].caption(f"引用校验: {'通过' if final.get('is_valid') else '未通过'}")
    meta[2].caption(f"trace: {(final.get('trace_id') or '')[:8]}")

    citations = final.get("citations") or []
    evidence = final.get("evidence") or []
    if citations:
        with st.expander(f"📌 引用来源 ({len(citations)})"):
            for c in citations:
                st.write(
                    f"- **{c.get('source', '')}**{_page_label(c.get('page'))} · `{c.get('chunk_id', '')}`"
                )
    if evidence:
        with st.expander(f"🧩 证据片段 ({len(evidence)})"):
            for e in evidence:
                st.markdown(f"**{e.get('source', '')}**{_page_label(e.get('page'))}")
                st.caption(e.get("text_preview", ""))

    fb = st.columns([1, 1, 6])
    if fb[0].button("👍", key=f"up-{key}"):
        _send_feedback(final, query, "thumbs_up")
    if fb[1].button("👎", key=f"down-{key}"):
        _send_feedback(final, query, "thumbs_down")

    trace_id = final.get("trace_id")
    if trace_id:
        if query:
            st.session_state.trace_labels[trace_id] = query


# 渲染 source chunks 浏览。
def _render_source_browser(client: CogDocClient, kb_id: str) -> None:
    with st.expander("索引内容"):
        try:
            status_code, payload = _cached_api_value(
                ("sources", client.base_url, kb_id),
                lambda: _response_status_payload(client.list_sources(kb_id)),
            )
        except Exception as exc:
            st.warning(f"读取来源文件失败: {exc}")
            return
        if status_code != 200:
            st.warning(
                "读取来源文件失败: "
                f"{format_api_error(payload, status_code, '读取来源文件失败')}"
            )
            return
        sources = payload.get("sources", []) if isinstance(payload, Mapping) else []
        sources = [str(source) for source in sources if source]
        if not sources:
            st.caption("暂无已索引 source")
            return
        selected = st.selectbox("source", sources, key=f"source-browser-{kb_id}")
        if not st.checkbox("加载 chunk 预览", key=f"source-browser-load-{kb_id}"):
            return
        chunk_limit = st.selectbox(
            "显示数量", [10, 20, 50], index=1, key=f"source-browser-limit-{kb_id}"
        )
        try:
            status_code, payload = _cached_api_value(
                ("chunks", client.base_url, kb_id, selected, 0, chunk_limit),
                lambda: _response_status_payload(
                    client.list_source_chunks(kb_id, selected, limit=chunk_limit)
                ),
            )
        except Exception as exc:
            st.warning(f"读取 chunks 失败: {exc}")
            return
        if status_code != 200:
            st.warning(
                "读取 chunks 失败: "
                f"{format_api_error(payload, status_code, '读取 chunks 失败')}"
            )
            return
        chunks = payload.get("chunks", []) if isinstance(payload, Mapping) else []
        total = (
            payload.get("total_count", len(chunks))
            if isinstance(payload, Mapping)
            else len(chunks)
        )
        st.caption(f"{selected} · {total} chunks")
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                continue
            chunk_id = str(chunk.get("chunk_id") or "")
            page_start = chunk.get("page_start", chunk.get("page"))
            page_end = chunk.get("page_end", page_start)
            page_label = _page_range_label(page_start, page_end)
            prefix = f"{page_label} · " if page_label else ""
            st.caption(f"{prefix}`{chunk_id}`")
            if chunk.get("context_preview"):
                st.caption(str(chunk.get("context_preview")))
            st.write(str(chunk.get("text_preview") or ""))


# 格式化检索分数。
def _score_label(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


# 渲染检索命中。
def _render_retrieve_hit(hit: Mapping) -> None:
    rank = hit.get("rank", "-")
    source = hit.get("source") or "-"
    page = _page_range_label(hit.get("page_start", hit.get("page")), hit.get("page_end"))
    chunk_id = hit.get("chunk_id") or "-"
    score = _score_label(hit.get("rerank_score"))
    title = f"#{rank} · {source}"
    if page:
        title += f" · {page}"
    with st.expander(title):
        cols = st.columns(4)
        cols[0].caption(f"chunk: {chunk_id}")
        cols[1].caption(f"chunk_index: {hit.get('chunk_index')}")
        cols[2].caption(f"rerank_score: {score}")
        cols[3].caption(f"rewrite: {hit.get('rewrite_query') or '-'}")
        st.write(hit.get("text_preview") or "")
        retrieval = hit.get("retrieval") or {}
        if retrieval:
            st.markdown("**retrieval metadata**")
            st.json(retrieval)


# 检索调试后台worker。
def _retrieve_debug_worker(
    *,
    api_url: str,
    kb_id: str,
    query: str,
    top_k: int,
    rerank: bool,
    rerank_top_n: int | None,
    outbox: queue.Queue,
) -> None:
    try:
        client = CogDocClient(api_url)
        resp = client.retrieve(
            kb_id,
            query,
            top_k=top_k,
            rerank=rerank,
            rerank_top_n=rerank_top_n,
        )
        outbox.put(
            (
                "result",
                {"status_code": resp.status_code, "payload": response_payload(resp)},
            )
        )
    except Exception as exc:
        outbox.put(("result", {"status_code": None, "payload": {"message": str(exc)}}))
    finally:
        outbox.put(("done", {}))


# 启动检索调试。
def _start_retrieve_debug(
    kb_id: str,
    query: str,
    top_k: int,
    rerank: bool,
    rerank_top_n: int | None,
) -> None:
    marker = _context_key(kb_id)
    pending = st.session_state.pending_retrieve_debugs.get(marker)
    if pending and not pending.get("done"):
        return
    outbox: queue.Queue = queue.Queue()
    pending = {
        "query": query,
        "top_k": top_k,
        "rerank": rerank,
        "rerank_top_n": rerank_top_n,
        "started_at": time.monotonic(),
        "queue": outbox,
        "done": False,
    }
    worker = threading.Thread(
        target=_retrieve_debug_worker,
        kwargs={
            "api_url": st.session_state.api_url,
            "kb_id": kb_id,
            "query": query,
            "top_k": top_k,
            "rerank": rerank,
            "rerank_top_n": rerank_top_n,
            "outbox": outbox,
        },
        daemon=True,
    )
    pending["thread"] = worker
    st.session_state.pending_retrieve_debugs[marker] = pending
    worker.start()


# 消费检索调试事件。
def _drain_retrieve_debug_events() -> None:
    for marker, pending in list(st.session_state.pending_retrieve_debugs.items()):
        outbox = pending["queue"]
        while True:
            try:
                event, data = outbox.get_nowait()
            except queue.Empty:
                break
            if event == "result":
                st.session_state.retrieve_debug_by_context[marker] = data
            elif event == "done":
                pending["done"] = True
        if pending.get("done"):
            st.session_state.pending_retrieve_debugs.pop(marker, None)


# 渲染检索调试。
def _render_retrieve_debug(client: CogDocClient, kb_id: str | None) -> None:
    st.subheader("检索调试")
    if not kb_id:
        st.info("先选择知识库。")
        return
    marker = _context_key(kb_id)
    pending = st.session_state.pending_retrieve_debugs.get(marker)
    with st.form(f"retrieve-debug-{kb_id}-{st.session_state.session_id}"):
        query = st.text_area("检索问题", height=90, placeholder="输入要召回的查询…")
        controls = st.columns([1, 1, 1])
        top_k = controls[0].slider("top_k", min_value=1, max_value=50, value=8)
        rerank = controls[1].checkbox("重排", value=True)
        rerank_top_n = controls[2].number_input(
            "rerank_top_n",
            min_value=1,
            max_value=50,
            value=min(8, top_k),
            disabled=not rerank,
        )
        if rerank:
            st.caption(
                "重排会加载 bge-reranker-v2-m3；无可用 GPU 时后端默认跳过 CPU 重排，"
                "如强制开启可能明显卡顿。"
            )
        submitted = st.form_submit_button(
            "运行检索", use_container_width=True, disabled=bool(pending)
        )
    if submitted:
        if not query.strip():
            st.warning("请输入检索问题。")
        else:
            _start_retrieve_debug(
                kb_id,
                query.strip(),
                top_k,
                rerank,
                int(rerank_top_n) if rerank else None,
            )
            st.rerun()
    pending = st.session_state.pending_retrieve_debugs.get(marker)
    if pending:
        elapsed = time.monotonic() - pending.get("started_at", time.monotonic())
        st.info(
            f"正在检索：{pending.get('query')} · "
            f"top_k={pending.get('top_k')} · "
            f"rerank={pending.get('rerank')} · {elapsed:.1f}s"
        )
        time.sleep(STREAM_RERUN_INTERVAL_SECONDS)
        st.rerun()
    result = st.session_state.retrieve_debug_by_context.get(marker)
    if not result:
        st.info("运行一次检索后，这里会显示命中 chunk、分数和 retrieval 元数据。")
        return
    status_code = result.get("status_code")
    payload = result.get("payload")
    if status_code != 200:
        st.error(format_api_error(payload, status_code, "检索失败"))
        return
    if not isinstance(payload, Mapping):
        st.error(f"检索响应格式不符合预期: {payload}")
        return
    hits = payload.get("hits", [])
    header = st.columns(4)
    header[0].caption(f"query: {payload.get('query') or '-'}")
    header[1].caption(f"top_k: {payload.get('top_k')}")
    header[2].caption(f"rerank: {payload.get('rerank')}")
    header[3].caption(f"hits: {len(hits) if isinstance(hits, list) else 0}")
    if not hits:
        st.warning("没有召回到内容。")
        return
    if any(
        isinstance(hit, Mapping)
        and (hit.get("retrieval") or {}).get("rerank_skipped_reason")
        for hit in hits
    ):
        st.warning("后端检测到 reranker 会走 CPU，已跳过重排以避免卡死。")
    for hit in hits:
        if isinstance(hit, Mapping):
            _render_retrieve_hit(hit)


# 渲染调试区。
def _debug_area(kb_id: str | None) -> None:
    trace_tab, retrieve_tab = st.tabs(["Trace 调试", "检索调试"])
    with trace_tab:
        _render_trace_lookup(kb_id)
    with retrieve_tab:
        _render_retrieve_debug(_client(), kb_id)


# 完成 侧边栏 处理。
def _sidebar() -> None:
    # 侧栏：后端地址、模式开关、知识库选择/新建/上传入库/文档列表。
    with st.sidebar:
        st.title("CogDoc")
        st.session_state.api_url = st.text_input("后端地址", st.session_state.api_url)
        st.session_state.is_local = st.toggle("本地 Ollama 模式", value=False)
        client = _client()

        st.divider()
        st.subheader("知识库")
        try:
            kbs = _cached_api_value(
                ("kbs", client.base_url), client.list_knowledge_bases
            )
        except CogDocAPIError as exc:
            st.error(f"读取知识库失败: {exc}")
            return
        except Exception as exc:
            st.error(f"连不上后端: {exc}")
            return

        if not isinstance(kbs, list):
            st.error(f"意外的响应格式: {kbs}")
            return
        if not all(
            isinstance(kb, Mapping) and isinstance(kb.get("kb_id"), str) for kb in kbs
        ):
            st.error(f"知识库列表响应缺少 kb_id: {kbs}")
            return
        kb_ids = [kb["kb_id"] for kb in kbs]
        if kb_ids:
            # kb 选择也持久化进 URL，刷新后定位回原库（历史按 (kb, session) 存）。
            url_kb = st.query_params.get("kb")
            default_idx = kb_ids.index(url_kb) if url_kb in kb_ids else 0
            st.session_state.kb_id = st.selectbox(
                "选择知识库", kb_ids, index=default_idx
            )
            st.query_params["kb"] = st.session_state.kb_id
        else:
            st.session_state.kb_id = None
            st.info("还没有知识库，先在下面新建一个。")

        with st.form("create_kb", clear_on_submit=True):
            new_kb = st.text_input("新建知识库 ID")
            if st.form_submit_button("创建") and new_kb:
                resp = client.create_knowledge_base(new_kb)
                if resp.status_code == 201:
                    _clear_api_cache(("kbs", client.base_url))
                    st.success(f"已创建 {new_kb}")
                    st.rerun()
                else:
                    st.error(resp.json().get("message", resp.text))

        if not st.session_state.kb_id:
            return

        kb_id = st.session_state.kb_id

        st.divider()
        _conversations(client, kb_id)

        st.divider()
        st.subheader("文档")
        uploaded = st.file_uploader("上传 PDF", type=["pdf"])
        if st.button("上传并入库", disabled=uploaded is None):
            resp = client.upload_document(kb_id, uploaded.name, uploaded.getvalue())
            if resp.status_code != 202:
                st.error(resp.json().get("message", resp.text))
            else:
                _poll_job(client, resp.json()["job_id"])
                _clear_api_cache(("documents", client.base_url, kb_id))
                _clear_api_cache(("sources", client.base_url, kb_id))
                _clear_api_cache(("chunks", client.base_url, kb_id))
                _clear_api_cache(("kbs", client.base_url))
                st.rerun()

        try:
            status_code, docs = _cached_api_value(
                ("documents", client.base_url, kb_id),
                lambda: _response_status_payload(client.list_documents(kb_id)),
            )
            if status_code != 200:
                st.error(
                    "读取文档列表失败: "
                    f"{format_api_error(docs, status_code, '读取文档列表失败')}"
                )
                docs = []
        except Exception as exc:
            st.error(f"读取文档列表失败: {exc}")
            docs = []
        if docs and not isinstance(docs, list):
            st.error(f"文档列表响应格式不符合预期: {docs}")
            docs = []
        if isinstance(docs, list) and docs:
            st.caption(f"文档 ({len(docs)})")
            for doc in docs:
                if not isinstance(doc, Mapping) or not doc.get("name"):
                    st.error(f"文档列表项格式不符合预期: {doc}")
                    continue
                row = st.columns([5, 1])
                row[0].write(doc["name"])
                if row[1].button("🗑", key=f"del-{doc['name']}"):
                    client.delete_document(kb_id, doc["name"])
                    _clear_api_cache(("documents", client.base_url, kb_id))
                    _clear_api_cache(("sources", client.base_url, kb_id))
                    _clear_api_cache(("chunks", client.base_url, kb_id))
                    _clear_api_cache(("kbs", client.base_url))
                    st.rerun()

        _render_source_browser(client, kb_id)

        with st.expander("⚠️ 删除知识库"):
            st.caption("会删除该库全部文档与索引，不可恢复。")
            if st.button("确认删除此知识库", key="del_kb"):
                resp = client.delete_knowledge_base(kb_id)
                if resp.status_code == 204:
                    _clear_api_cache()
                    st.session_state.kb_id = None
                    st.query_params.pop("kb", None)
                    st.success("已删除")
                    st.rerun()
                else:
                    st.error(resp.json().get("message", resp.text))


# 轮询任务。
def _poll_job(client: CogDocClient, job_id: str) -> None:
    # 轮询入库 job 直到终态，期间在 st.status 里实时显示进度。
    with st.status("后台入库中…", expanded=True) as status:
        job = {}
        for _ in range(300):
            resp = client.get_job(job_id)
            if resp.status_code != 200:
                # job 端点出错（如任务过期）时响应没有 status 字段，直接报错退出。
                status.update(
                    label=f"查询入库任务失败：{resp.text[:200]}", state="error"
                )
                return
            job = resp.json()
            if job.get("status") in ("succeeded", "failed"):
                break
            time.sleep(0.2)
        if job.get("status") == "succeeded":
            status.update(
                label=f"入库完成：{job.get('document_count')} 篇 / {job.get('chunk_count')} chunks",
                state="complete",
            )
        else:
            status.update(
                label=f"入库失败：{job.get('message', '') or '超时未完成'}",
                state="error",
            )


# 流式处理对话worker。
def _stream_chat_worker(
    *,
    api_url: str,
    kb_id: str,
    session_id: str,
    prompt: str,
    mode: str,
    is_local: bool,
    stop_event: threading.Event,
    outbox: queue.Queue,
) -> None:
    # 后台线程只碰队列和 stop_event，不直接写 Streamlit 状态，避免跨线程 UI 状态竞争。
    try:
        client = CogDocClient(api_url)
        for event, data in client.stream_chat(
            kb_id,
            prompt,
            mode=mode,
            session_id=session_id,
            is_local=is_local,
            on_response=lambda response: outbox.put(
                ("response", {"response": response})
            ),
        ):
            if stop_event.is_set():
                break
            outbox.put((event, data))
    except Exception as exc:
        if not stop_event.is_set():
            outbox.put(("error", {"message": str(exc)}))
    finally:
        outbox.put(("done", {"cancelled": stop_event.is_set()}))


# 处理startstream。
def _start_stream(kb_id: str, prompt: str, mode: str) -> None:
    key = _context_key(kb_id)
    pending = st.session_state.pending_streams.get(key)
    if pending and not pending.get("done"):
        return

    user_msg_id = _next_id()
    _messages_for(kb_id).append({"role": "user", "content": prompt, "id": user_msg_id})

    outbox: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    pending = {
        "kb_id": kb_id,
        "session_id": st.session_state.session_id,
        "prompt": prompt,
        "mode": mode,
        "is_local": st.session_state.is_local,
        "user_msg_id": user_msg_id,
        "answer": "",
        "final": None,
        "error": None,
        "stage": "",
        "done": False,
        "cancelled": False,
        "queue": outbox,
        "stop_event": stop_event,
    }
    worker = threading.Thread(
        target=_stream_chat_worker,
        kwargs={
            "api_url": st.session_state.api_url,
            "kb_id": kb_id,
            "session_id": st.session_state.session_id,
            "prompt": prompt,
            "mode": mode,
            "is_local": st.session_state.is_local,
            "stop_event": stop_event,
            "outbox": outbox,
        },
        daemon=True,
    )
    pending["thread"] = worker
    st.session_state.pending_streams[key] = pending
    worker.start()


# 移除消息。
def _remove_message(kb_id: str, session_id: str, msg_id: int) -> None:
    messages = _messages_for(kb_id, session_id)
    st.session_state.messages_by_context[_context_key(kb_id, session_id)] = [
        msg for msg in messages if msg.get("id") != msg_id
    ]


# 处理cancelstream。
def _cancel_stream(key: tuple[str, str]) -> None:
    pending = st.session_state.pending_streams.get(key)
    if not pending:
        return
    pending["cancelled"] = True
    pending["done"] = True
    pending["stop_event"].set()
    response = pending.get("response")
    if response is not None:
        response.close()
    _remove_message(
        pending["kb_id"],
        pending["session_id"],
        pending["user_msg_id"],
    )


# 处理finishstream。
def _finish_stream(key: tuple[str, str], pending: dict) -> None:
    if pending.get("cancelled"):
        _remove_message(
            pending["kb_id"],
            pending["session_id"],
            pending["user_msg_id"],
        )
        st.session_state.pending_streams.pop(key, None)
        return

    error = pending.get("error")
    if error:
        _messages_for(pending["kb_id"], pending["session_id"]).append(
            {
                "role": "assistant",
                "content": f"[{error.get('error_code', 'ERROR')}] {error.get('message', '')}",
                "id": _next_id(),
            }
        )
        st.session_state.pending_streams.pop(key, None)
        return

    final = pending.get("final")
    answer = (final or {}).get("answer") or pending.get("answer", "")
    trace_id = (final or {}).get("trace_id")
    if trace_id and pending.get("prompt"):
        st.session_state.trace_labels[trace_id] = pending["prompt"]
    _messages_for(pending["kb_id"], pending["session_id"]).append(
        {
            "role": "assistant",
            "content": answer or "（无答案）",
            "final": final,
            "query": pending["prompt"],
            "id": _next_id(),
        }
    )
    st.session_state.pending_streams.pop(key, None)


# 处理drainstreamevents。
def _drain_stream_events() -> None:
    for key, pending in list(st.session_state.pending_streams.items()):
        outbox = pending["queue"]
        while True:
            try:
                event, data = outbox.get_nowait()
            except queue.Empty:
                break
            if pending.get("cancelled"):
                continue
            if event == "token":
                pending["answer"] += data.get("content", "")
            elif event == "start":
                pending["stage"] = "正在启动请求…"
            elif event == "node":
                stage = data.get("stage", "")
                pending["stage"] = f"正在处理：{stage}" if stage else ""
            elif event == "final":
                pending["final"] = data
            elif event == "error":
                pending["error"] = data
            elif event == "response":
                pending["response"] = data.get("response")
            elif event == "done":
                pending["cancelled"] = bool(data.get("cancelled"))
                pending["done"] = True
        if pending.get("done"):
            _finish_stream(key, pending)


# 完成 chatarea 处理。
def _chat_area() -> None:
    # 主对话区：按 (kb, session) 还原历史 + 渲染气泡；SSE 在后台线程中归属到发送时上下文。
    _drain_stream_events()
    _drain_retrieve_debug_events()
    kb_id = st.session_state.kb_id
    if kb_id:
        _restore_history(kb_id)
    current_key = _context_key(kb_id) if kb_id else None
    current_pending = (
        st.session_state.pending_streams.get(current_key) if current_key else None
    )
    answering = bool(current_pending)
    current_view = st.session_state.main_views_by_context.get(current_key, "对话")
    if current_view not in MAIN_VIEWS:
        current_view = "对话"
        st.session_state.main_views_by_context[current_key] = current_view
    view_key = (
        f"main-view-{kb_id}-{st.session_state.session_id}"
        if current_key
        else "main-view-empty"
    )
    view = st.radio(
        "主视图",
        MAIN_VIEWS,
        index=MAIN_VIEWS.index(current_view),
        horizontal=True,
        key=view_key,
        label_visibility="collapsed",
    )
    if current_key:
        st.session_state.main_views_by_context[current_key] = view

    if view == "对话":
        st.subheader(f"对话 · {kb_id or '未选择知识库'}")
        mode = st.radio(
            "模式",
            ["auto", "qa", "summary", "compare"],
            horizontal=True,
            key="chat_mode",
            disabled=answering,
        )

        messages = _messages_for(kb_id) if kb_id else []
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"] or "（无答案）")
                if msg.get("final") and not answering:
                    _render_evidence(
                        msg["final"], key=msg["id"], query=msg.get("query", "")
                    )

        if current_pending:
            with st.chat_message("assistant"):
                st.markdown(_stream_preview(current_pending.get("answer", "")))
                if current_pending.get("stage"):
                    st.caption(current_pending["stage"])
            if st.button("■ 终止问题", type="primary", use_container_width=True):
                _cancel_stream(current_key)
                st.rerun()
            time.sleep(STREAM_RERUN_INTERVAL_SECONDS)
            st.rerun()

        prompt = st.chat_input("问点什么…", disabled=not kb_id)
        if prompt:
            _start_stream(kb_id, prompt, mode)
            st.rerun()
    else:
        _debug_area(kb_id)


# 完成 nextid 处理。
def _next_id() -> int:
    st.session_state.msg_seq += 1
    return st.session_state.msg_seq


# 隐藏default默认界面。
def _hide_default_chrome() -> None:
    # 隐藏 Streamlit 右上角默认工具条（Deploy / 菜单 / 状态）与页脚，只留自家品牌。
    st.markdown(
        """
        <style>
        [data-testid="stToolbar"] {display: none;}
        [data-testid="stDecoration"] {display: none;}
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# 完成 品牌头部页头 处理。
def _brand_header() -> None:
    # 主区顶部品牌标题，替代被隐藏的默认头。
    st.markdown(
        "<h2 style='margin:0 0 0.1rem 0;'>🧠 CogDoc</h2>"
        "<p style='color:#888;margin:0 0 0.6rem 0;'>面向个人 / 企业的本地 RAG 知识库控制台</p>",
        unsafe_allow_html=True,
    )


# 启动入口。
def main() -> None:
    _init_state()
    _hide_default_chrome()
    _brand_header()
    _sidebar()
    _chat_area()


main()
