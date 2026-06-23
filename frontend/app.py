import os
import time
import uuid
import streamlit as st
from frontend.api_client import CogDocClient

DEFAULT_API_URL = os.getenv("COGDOC_API_URL", "http://localhost:8000")

st.set_page_config(page_title="CogDoc", layout="wide")


def _client() -> CogDocClient:
    return CogDocClient(st.session_state.api_url)


def _init_state() -> None:
    st.session_state.setdefault("api_url", DEFAULT_API_URL)
    # session_id 持久化进 URL，刷新后复用同一会话（后端多轮记忆得以续上）。
    if "session_id" not in st.session_state:
        st.session_state.session_id = st.query_params.get("sid") or uuid.uuid4().hex
        st.query_params["sid"] = st.session_state.session_id
    st.session_state.setdefault("kb_id", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("msg_seq", 0)
    st.session_state.setdefault("restored_for", None)
    # 本会话内已知的对话 id（按 kb），与后端列表合并，保证空/新对话也留得住、点得到。
    st.session_state.setdefault("known_sessions", {})


def _restore_history(kb_id: str) -> None:
    # kb 或 session 变化时重载该 kb 的历史；同一 (kb, session) 内不重复拉，保留实时追加的消息。
    marker = (kb_id, st.session_state.session_id)
    if st.session_state.restored_for == marker:
        return
    try:
        resp = _client().get_session_history(st.session_state.session_id, kb_id)
        turns = resp.json().get("messages", []) if resp.status_code == 200 else []
    except Exception:
        turns = []
    st.session_state.messages = [
        {
            "role": turn.get("role", "assistant"),
            "content": turn.get("content", ""),
            "id": _next_id(),
        }
        for turn in turns
    ]
    st.session_state.restored_for = marker


def _page_label(page) -> str:
    # page 可能为空（schema 默认 None），避免渲染成 "PNone"。
    return f" · P{page}" if page is not None else ""


def _switch_session(session_id: str) -> None:
    # 切换/新建对话：换 session_id（同步 URL）、清空界面、触发重载历史。
    st.session_state.session_id = session_id
    st.query_params["sid"] = session_id
    st.session_state.messages = []
    st.session_state.restored_for = None
    st.rerun()


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
        backend = {
            s["session_id"]: s
            for s in client.list_sessions(kb_id).json().get("sessions", [])
        }
    except Exception:
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
        except Exception as exc:
            st.error(f"连不上后端: {exc}")
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
            docs = client.list_documents(kb_id).json()
        except Exception as exc:
            st.error(f"读取文档列表失败: {exc}")
            docs = []
        if isinstance(docs, list) and docs:
            st.caption(f"文档 ({len(docs)})")
            for doc in docs:
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


def _chat_area() -> None:
    # 主对话区：还原历史 + 渲染气泡 + SSE 流式提问，final 帧为权威答案。
    kb_id = st.session_state.kb_id
    if kb_id:
        _restore_history(kb_id)
    st.subheader(f"对话 · {kb_id or '未选择知识库'}")
    mode = st.radio("模式", ["auto", "qa", "summary", "compare"], horizontal=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"] or "（无答案）")
            if msg.get("final"):
                _render_evidence(
                    msg["final"], key=msg["id"], query=msg.get("query", "")
                )

    prompt = st.chat_input("问点什么…", disabled=not kb_id)
    if not prompt:
        return

    st.session_state.messages.append(
        {"role": "user", "content": prompt, "id": _next_id()}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        box = st.empty()
        answer, final, error = "", None, None
        try:
            for event, data in _client().stream_chat(
                kb_id,
                prompt,
                mode=mode,
                session_id=st.session_state.session_id,
                is_local=st.session_state.is_local,
            ):
                if event == "token":
                    answer += data.get("content", "")
                    box.markdown(answer + "▌")
                elif event == "final":
                    final = data
                elif event == "error":
                    error = data
        except Exception as exc:
            error = {"message": str(exc)}

        if error:
            box.error(
                f"[{error.get('error_code', 'ERROR')}] {error.get('message', '')}"
            )
            answer = ""
        else:
            answer = (final or {}).get("answer", answer)
            box.markdown(answer or "（无答案）")
            if final:
                _render_evidence(final, key=str(_next_id()), query=prompt)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "final": final,
            "query": prompt,
            "id": _next_id(),
        }
    )


def _next_id() -> int:
    st.session_state.msg_seq += 1
    return st.session_state.msg_seq


def main() -> None:
    _init_state()
    _sidebar()
    _chat_area()


main()
