import os
import queue
import threading
import time
import uuid
from typing import Mapping
import streamlit as st
from cogdoc.frontend.api_client import (
    CogDocAPIError,
    CogDocClient,
    format_api_error,
    response_payload,
)

DEFAULT_API_URL = os.getenv("COGDOC_API_URL", "http://localhost:8000")

st.set_page_config(page_title="CogDoc", layout="wide")


# 创建测试客户端。
def _client() -> CogDocClient:
    return CogDocClient(st.session_state.api_url)


def _response_error(response, fallback: str = "请求失败") -> str:
    return format_api_error(response_payload(response), response.status_code, fallback)


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
    st.session_state.setdefault("trace_cache", {})
    st.session_state.setdefault("active_trace_id", "")
    st.session_state.setdefault("trace_list", [])
    st.session_state.setdefault("trace_list_error", "")
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


def _context_key(kb_id: str, session_id: str | None = None) -> tuple[str, str]:
    return (kb_id, session_id or st.session_state.session_id)


def _messages_for(kb_id: str, session_id: str | None = None) -> list[dict]:
    return st.session_state.messages_by_context.setdefault(
        _context_key(kb_id, session_id), []
    )


# 恢复历史记录。
def _restore_history(kb_id: str) -> None:
    # kb 或 session 变化时重载该 kb 的历史；同一 (kb, session) 内不重复拉，保留实时追加的消息。
    marker = _context_key(kb_id)
    if marker in st.session_state.restored_contexts:
        return
    try:
        resp = _client().get_session_history(st.session_state.session_id, kb_id)
        turns = resp.json().get("messages", []) if resp.status_code == 200 else []
    except Exception:
        turns = []
    st.session_state.messages_by_context[marker] = [
        {
            "role": turn.get("role", "assistant"),
            "content": turn.get("content", ""),
            "id": _next_id(),
        }
        for turn in turns
    ]
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
        resp = client.list_sessions(kb_id)
        if resp.status_code == 200:
            payload = response_payload(resp)
            sessions = payload.get("sessions", []) if isinstance(payload, Mapping) else []
            backend = {
                s["session_id"]: s
                for s in sessions
                if isinstance(s, Mapping) and isinstance(s.get("session_id"), str)
            }
        else:
            st.warning(f"读取会话列表失败: {_response_error(resp)}")
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
            st.session_state.trace_cache[trace_id] = response_payload(resp)
        else:
            st.session_state.trace_cache[trace_id] = {
                "error": _response_error(resp, "读取 trace 失败")
            }
    except Exception as exc:
        st.session_state.trace_cache[trace_id] = {"error": str(exc)}


def _load_trace_list() -> None:
    try:
        resp = _client().list_traces(limit=30)
        if resp.status_code != 200:
            st.session_state.trace_list_error = _response_error(
                resp, "读取 trace 列表失败"
            )
            st.session_state.trace_list = []
            return
        payload = response_payload(resp)
        traces = payload.get("traces", []) if isinstance(payload, Mapping) else []
        st.session_state.trace_list = [
            trace
            for trace in traces
            if isinstance(trace, Mapping) and isinstance(trace.get("trace_id"), str)
        ]
        st.session_state.trace_list_error = ""
    except Exception as exc:
        st.session_state.trace_list = []
        st.session_state.trace_list_error = str(exc)


def _trace_option_label(trace_id: str) -> str:
    if not trace_id:
        return "选择最近 trace"
    for trace in st.session_state.trace_list:
        if trace.get("trace_id") == trace_id:
            status = trace.get("status", "-")
            task = trace.get("task_type", "-")
            duration = trace.get("duration_ms")
            suffix = f" · {duration} ms" if duration is not None else ""
            return f"{trace_id[:8]} · {task} · {status}{suffix}"
    return trace_id


def _render_trace_step(step: Mapping, idx: int) -> None:
    node = step.get("node_name") or f"step-{idx + 1}"
    duration = step.get("duration_ms")
    label = f"{idx + 1}. {node}"
    if duration is not None:
        label += f" · {duration} ms"
    with st.expander(label):
        cols = st.columns(4)
        cols[0].caption(f"任务: {step.get('task_type') or '-'}")
        cols[1].caption(f"模型: {step.get('model') or '-'}")
        cols[2].caption(f"top_k: {step.get('retrieval_top_k') or '-'}")
        cols[3].caption(f"错误: {step.get('error_class') or '-'}")

        if step.get("router_reason"):
            st.markdown("**路由理由**")
            st.write(step["router_reason"])
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
                st.caption(
                    f"{source}{_page_label(item.get('page'))} · `{chunk_id}`"
                )
                st.write(item.get("text_preview", ""))


def _render_trace_debug(trace: dict) -> None:
    trace_error = trace.get("error")
    if isinstance(trace_error, str):
        st.error(trace["error"])
        return
    summary = trace.get("summary") or {}
    meta = st.columns(4)
    meta[0].caption(f"状态: {trace.get('status') or '-'}")
    meta[1].caption(f"任务: {trace.get('task_type') or '-'}")
    meta[2].caption(f"耗时: {trace.get('duration_ms') or '-'} ms")
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


def _render_trace_lookup() -> None:
    with st.expander("Trace 调试"):
        list_cols = st.columns([1, 1, 3])
        if list_cols[0].button("最近 trace", key="trace-list-load"):
            _load_trace_list()
        if list_cols[1].button("刷新列表", key="trace-list-refresh"):
            _load_trace_list()
        if st.session_state.trace_list_error:
            st.warning(f"读取 trace 列表失败: {st.session_state.trace_list_error}")
        if st.session_state.trace_list:
            options = [""] + [trace["trace_id"] for trace in st.session_state.trace_list]
            selected = st.selectbox(
                "最近请求",
                options,
                format_func=_trace_option_label,
                key="trace_recent_select",
            )
            if selected and st.button("打开选中 trace", key="trace-open-selected"):
                st.session_state.active_trace_id = selected
                _load_trace(selected)

        with st.form("trace_lookup"):
            trace_id = st.text_input(
                "trace_id", value=st.session_state.active_trace_id
            ).strip()
            submitted = st.form_submit_button("查询")
        if submitted and trace_id:
            st.session_state.active_trace_id = trace_id
            _load_trace(trace_id)
        active = st.session_state.active_trace_id
        if active and st.button("刷新 trace", key="trace-refresh"):
            _load_trace(active, force=True)
        if active and active in st.session_state.trace_cache:
            st.caption(f"trace_id: {active}")
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
        if st.button("查看 trace", key=f"trace-load-{key}"):
            st.session_state.active_trace_id = trace_id
            _load_trace(trace_id)
        if trace_id in st.session_state.trace_cache:
            with st.expander("Trace 调试", expanded=True):
                st.caption(f"trace_id: {trace_id}")
                _render_trace_debug(st.session_state.trace_cache[trace_id])


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
            kbs = client.list_knowledge_bases()
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
                st.rerun()

        try:
            resp = client.list_documents(kb_id)
            if resp.status_code != 200:
                st.error(f"读取文档列表失败: {_response_error(resp)}")
                docs = []
            else:
                docs = response_payload(resp)
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
                    st.rerun()

        with st.expander("⚠️ 删除知识库"):
            st.caption("会删除该库全部文档与索引，不可恢复。")
            if st.button("确认删除此知识库", key="del_kb"):
                resp = client.delete_knowledge_base(kb_id)
                if resp.status_code == 204:
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


def _remove_message(kb_id: str, session_id: str, msg_id: int) -> None:
    messages = _messages_for(kb_id, session_id)
    st.session_state.messages_by_context[_context_key(kb_id, session_id)] = [
        msg for msg in messages if msg.get("id") != msg_id
    ]


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
    kb_id = st.session_state.kb_id
    if kb_id:
        _restore_history(kb_id)
    st.subheader(f"对话 · {kb_id or '未选择知识库'}")
    current_key = _context_key(kb_id) if kb_id else None
    current_pending = (
        st.session_state.pending_streams.get(current_key) if current_key else None
    )
    answering = bool(current_pending)
    mode = st.radio(
        "模式",
        ["auto", "qa", "summary", "compare"],
        horizontal=True,
        key="chat_mode",
        disabled=answering,
    )
    _render_trace_lookup()

    messages = _messages_for(kb_id) if kb_id else []
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"] or "（无答案）")
            if msg.get("final"):
                _render_evidence(
                    msg["final"], key=msg["id"], query=msg.get("query", "")
                )

    if current_pending:
        with st.chat_message("assistant"):
            answer = current_pending.get("answer", "")
            st.markdown((answer + "▌") if answer else "正在思考…")
            if current_pending.get("stage"):
                st.caption(current_pending["stage"])
        if st.button("■ 终止问题", type="primary", use_container_width=True):
            _cancel_stream(current_key)
            st.rerun()
        time.sleep(0.1)
        st.rerun()

    prompt = st.chat_input("问点什么…", disabled=not kb_id)
    if prompt:
        _start_stream(kb_id, prompt, mode)
        st.rerun()


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
