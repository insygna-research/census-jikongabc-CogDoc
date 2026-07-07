import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from concurrent.futures import Executor, Future
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# 同步执行器用于避免冒烟检查中的嵌套线程等待。
class InlineExecutor(Executor):
    # 立即执行提交的函数并返回结果对象。
    def submit(self, fn, /, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    # 冒烟检查执行器没有后台线程需要回收。
    def shutdown(self, wait=True, *, cancel_futures=False):
        return None


# 设置隔离状态目录，避免污染真实数据目录。
def _configure_isolated_env(data_dir: Path) -> None:
    os.environ["COGDOC_DATA_DIR"] = str(data_dir)
    os.environ["COGDOC_TRACE_ENABLED"] = "false"
    os.environ["COGDOC_TRACE_DIR"] = str(data_dir / "traces")
    os.environ["COGDOC_LOG_TO_CONSOLE"] = "false"
    os.environ["COGDOC_API_KEYS"] = ""
    os.environ["RATE_LIMIT_BURST"] = "0"

    from cogdoc.config.settings import get_settings

    get_settings.cache_clear()


# 构造模拟对话结果，避免真实大模型调用。
def _fake_chat_result(answer: str, trace_id: str, task_type: str = "qa"):
    from cogdoc.service.chat_service import ChatResult

    return ChatResult(
        answer=answer,
        task_type=task_type,
        citations=[{"chunk_id": "chunk-1", "source": "a.pdf", "page": 1}],
        evidence=[
            {
                "chunk_id": "chunk-1",
                "chunk_index": 0,
                "source": "a.pdf",
                "page": 1,
                "text_preview": "smoke evidence",
            }
        ],
        critique="",
        is_valid=True,
        trace_id=trace_id,
        request_id=trace_id,
        steps=[],
        chat_messages=[
            {"role": "user", "content": "smoke", "timestamp": None},
            {"role": "assistant", "content": answer, "timestamp": None},
        ],
        raw_output={"answer": answer},
    )


# 构造注入模拟依赖的接口应用。
def _build_app(data_dir: Path):
    from cogdoc.api.app import create_app
    from cogdoc.api.feedback_store import FeedbackStore
    from cogdoc.api.ingest import IndexJobManager, KnowledgeBaseRegistry
    from cogdoc.api.session_store import SessionStore
    from cogdoc.service.chat_service import ChatEvent

    import cogdoc.api.routes.documents as document_routes

    # 冒烟检查只验证接口串联，不做真实索引清理。
    document_routes.delete_kb_index_transactional = lambda kb_id: None
    document_routes.mark_kb_deleted = lambda kb_id: None

    def fake_doc() -> dict:
        return {
            "text": "smoke evidence text about a document",
            "meta": {
                "chunk_id": "chunk-1",
                "chunk_index": 0,
                "source": "a.pdf",
                "page": 1,
                "page_start": 1,
                "page_end": 1,
            },
            "retrieval": {"channel": "smoke", "score": 1.0},
        }

    # 返回当前知识库的隔离源文件目录。
    def source_dir_for(kb_id: str) -> str:
        return str(data_dir / "kb" / kb_id / "sources")

    # 写入最小清单，让文档接口可读到模拟入库结果。
    def fake_ingest(kb_id: str, source_dir: str, on_commit=None):
        from cogdoc.tools.manifest import (
            save_index_manifest,
            stamp_chunk_identity_contract,
        )

        if on_commit is not None:
            on_commit(f"smoke-{int(time.time() * 1000)}")
        pdfs = sorted(Path(source_dir).glob("*.pdf"))
        save_index_manifest(
            stamp_chunk_identity_contract(
                {
                    "doc_id": kb_id,
                    "documents": [
                        {"name": p.name, "size": p.stat().st_size, "sha256": "smoke"}
                        for p in pdfs
                    ],
                }
            )
        )
        doc_count = len(pdfs)
        return SimpleNamespace(document_count=doc_count, chunk_count=doc_count * 3)

    # 同步对话运行器避免真实图和模型调用。
    def fake_chat_runner(doc_id, query, is_local, chat_history, forced_task):
        task_type = forced_task or "qa"
        return _fake_chat_result(
            f"smoke answer for {query}", "trace-smoke-sync", task_type
        )

    # 流式对话运行器产出三类事件帧。
    def fake_chat_stream(doc_id, query, is_local, chat_history, forced_task):
        task_type = forced_task or "qa"
        result = _fake_chat_result(
            f"streamed smoke answer for {query}", "trace-smoke-stream", task_type
        )
        yield ChatEvent(
            "request_started", {"trace_id": result.trace_id, "doc_id": doc_id}
        )
        yield ChatEvent("token", {"content": "streamed"})
        yield ChatEvent("final", {"result": result, "output": result.raw_output})

    registry = KnowledgeBaseRegistry(
        registry_path=str(data_dir / "kb" / "registry.json"),
        source_dir_for=source_dir_for,
    )
    jobs = IndexJobManager(
        ingest_fn=fake_ingest,
        source_dir_for=source_dir_for,
        kb_exists=registry.exists,
    )
    app = create_app(
        chat_runner=fake_chat_runner,
        chat_stream_runner=fake_chat_stream,
        session_store=SessionStore(),
        kb_registry=registry,
        index_jobs=jobs,
        feedback_store=FeedbackStore(
            feedback_path=str(data_dir / "feedback" / "feedback.jsonl"),
            bad_cases_path=str(data_dir / "feedback" / "bad_cases.jsonl"),
        ),
        api_keys=set(),
        offload_workers=1,
    )
    # 替换前关闭默认线程池，避免留下未使用的执行器。
    app.state.offload_executor.shutdown(wait=False)
    # 仅冒烟检查替换为同步执行器，生产代码仍使用有界线程池。
    app.state.offload_executor = InlineExecutor()
    app.state.source_list_reader = lambda kb_id: ["a.pdf"]
    app.state.source_chunks_reader = lambda kb_id, source: [fake_doc()]
    app.state.retrieve_runner = lambda body: [fake_doc()]
    return app


# 轮询入库任务直到终态。
async def _wait_job(client, job_id: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        resp = await client.get(f"/v1/index-jobs/{job_id}")
        _assert_status(resp, 200, f"get job {job_id}")
        last = resp.json()
        if last["status"] not in {"pending", "running"}:
            return last
        await asyncio.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s; last={last}")


# 校验响应状态码并输出失败上下文。
def _assert_status(resp, expected: int, label: str) -> None:
    if resp.status_code != expected:
        raise AssertionError(
            f"{label}: expected HTTP {expected}, got {resp.status_code}: {resp.text}"
        )


# 按详细开关打印冒烟检查步骤。
def _print_step(verbose: bool, name: str, payload=None) -> None:
    if not verbose:
        return
    if payload is None:
        print(f"[ok] {name}")
    else:
        print(f"[ok] {name}: {json.dumps(payload, ensure_ascii=False)}")


# 执行完整接口冒烟检查流程。
async def _run_smoke(data_dir: Path, timeout: float, verbose: bool) -> None:
    from httpx import ASGITransport, AsyncClient

    app = _build_app(data_dir)
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            kb_id = "smoke-kb"
            session_id = "smoke-session"

            created = await client.post("/v1/knowledge-bases", json={"kb_id": kb_id})
            _assert_status(created, 201, "create KB")
            _print_step(verbose, "create KB", created.json())

            listed = await client.get("/v1/knowledge-bases")
            _assert_status(listed, 200, "list KBs")
            assert any(item["kb_id"] == kb_id for item in listed.json())
            _print_step(verbose, "list KBs")

            uploaded = await client.post(
                f"/v1/knowledge-bases/{kb_id}/documents",
                files={
                    "file": ("a.pdf", b"%PDF-1.4 smoke\n%%EOF\n", "application/pdf")
                },
            )
            _assert_status(uploaded, 202, "upload document")
            upload_job = await _wait_job(client, uploaded.json()["job_id"], timeout)
            assert upload_job["status"] == "succeeded", upload_job
            assert upload_job["document_count"] == 1, upload_job
            _print_step(verbose, "upload document job", upload_job)

            docs = await client.get(f"/v1/knowledge-bases/{kb_id}/documents")
            _assert_status(docs, 200, "list documents")
            assert docs.json()[0]["name"] == "a.pdf", docs.text
            _print_step(verbose, "list documents", docs.json())

            sources = await client.get(f"/v1/knowledge-bases/{kb_id}/sources")
            _assert_status(sources, 200, "list sources")
            assert sources.json()["sources"] == ["a.pdf"], sources.text
            _print_step(verbose, "list sources", sources.json())

            chunks = await client.get(
                f"/v1/knowledge-bases/{kb_id}/sources/a.pdf/chunks"
            )
            _assert_status(chunks, 200, "source chunks")
            assert chunks.json()["chunks"][0]["chunk_id"] == "chunk-1", chunks.text
            _print_step(verbose, "source chunks", chunks.json())

            retrieve = await client.post(
                "/v1/retrieve",
                json={"query": "smoke retrieval", "doc_id": kb_id, "top_k": 3},
            )
            _assert_status(retrieve, 200, "retrieve")
            assert retrieve.json()["hits"][0]["source"] == "a.pdf", retrieve.text
            _print_step(verbose, "retrieve", retrieve.json())

            chat = await client.post(
                "/v1/chat",
                json={
                    "query": "smoke question",
                    "doc_id": kb_id,
                    "session_id": session_id,
                    "mode": "qa",
                },
            )
            _assert_status(chat, 200, "chat")
            chat_body = chat.json()
            assert chat_body["trace_id"] == "trace-smoke-sync", chat_body
            assert chat_body["citations"][0]["source"] == "a.pdf", chat_body
            _print_step(verbose, "chat", chat_body)

            summary = await client.post(
                "/v1/summary",
                json={
                    "query": "summarize a.pdf",
                    "doc_id": kb_id,
                    "session_id": session_id,
                },
            )
            _assert_status(summary, 200, "summary")
            assert summary.json()["task_type"] == "summary", summary.text
            _print_step(verbose, "summary", summary.json())

            compare = await client.post(
                "/v1/compare",
                json={
                    "query": "compare a.pdf and b.pdf",
                    "doc_id": kb_id,
                    "session_id": session_id,
                },
            )
            _assert_status(compare, 200, "compare")
            assert compare.json()["task_type"] == "compare", compare.text
            _print_step(verbose, "compare", compare.json())

            stream = await client.post(
                "/v1/chat/stream",
                json={
                    "query": "stream smoke",
                    "doc_id": kb_id,
                    "session_id": session_id,
                    "mode": "summary",
                },
            )
            _assert_status(stream, 200, "chat stream")
            assert "event: start" in stream.text, stream.text
            assert "event: token" in stream.text, stream.text
            assert "event: final" in stream.text, stream.text
            _print_step(verbose, "chat stream")

            from cogdoc.observability.trace import build_trace_payload, trace_path

            trace_payload = build_trace_payload(
                trace_id=chat_body["trace_id"],
                request_id=chat_body["request_id"],
                task_type=chat_body["task_type"],
                steps=[
                    {
                        "node_name": "smoke",
                        "duration_ms": 1.0,
                        "model": None,
                        "token": None,
                        "retrieval_top_k": 3,
                        "critique": None,
                        "error_class": None,
                        "counts": {"evidence_count": 1},
                        "evidence": chat_body["evidence"],
                    }
                ],
                config={
                    "doc_id": kb_id,
                    "session_id": session_id,
                    "query_preview": "smoke question",
                },
            )
            path = trace_path(chat_body["trace_id"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(trace_payload, ensure_ascii=False), encoding="utf-8"
            )

            traces = await client.get(
                "/v1/traces", params={"doc_id": kb_id, "session_id": session_id}
            )
            _assert_status(traces, 200, "list traces")
            assert traces.json()["traces"][0]["trace_id"] == chat_body["trace_id"], (
                traces.text
            )
            trace = await client.get(f"/v1/traces/{chat_body['trace_id']}")
            _assert_status(trace, 200, "get trace")
            assert trace.json()["summary"]["step_count"] == 1, trace.text
            _print_step(verbose, "trace", trace.json())

            history = await client.get(
                f"/v1/sessions/{session_id}/history", params={"doc_id": kb_id}
            )
            _assert_status(history, 200, "session history")
            assert history.json()["messages"], history.text
            _print_step(verbose, "session history", history.json())

            feedback = await client.post(
                "/v1/feedback",
                json={
                    "trace_id": chat_body["trace_id"],
                    "feedback": "thumbs_down",
                    "kb_id": kb_id,
                    "query": "smoke question",
                    "answer": chat_body["answer"],
                    "citations": chat_body["citations"],
                    "evidence": chat_body["evidence"],
                },
            )
            _assert_status(feedback, 201, "feedback")
            assert feedback.json()["is_bad_case"] is True, feedback.text
            _print_step(verbose, "feedback", feedback.json())

            deleted_doc = await client.delete(
                f"/v1/knowledge-bases/{kb_id}/documents/a.pdf"
            )
            _assert_status(deleted_doc, 202, "delete document")
            delete_job = await _wait_job(client, deleted_doc.json()["job_id"], timeout)
            assert delete_job["status"] == "succeeded", delete_job
            _print_step(verbose, "delete document job", delete_job)

            deleted_kb = await client.delete(f"/v1/knowledge-bases/{kb_id}")
            _assert_status(deleted_kb, 204, "delete KB")
            _print_step(verbose, "delete KB")
    finally:
        # 非阻塞收尾避免进程内接口模式等待空闲执行器。
        app.state.index_jobs.shutdown(wait=False)
        app.state.offload_executor.shutdown(wait=False)


# 解析命令行参数并启动冒烟检查。
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run an in-process CogDoc API smoke test without real LLM/index work. "
            "The harness replaces the app offload pool with an inline executor so "
            "the smoke stays deterministic under ASGITransport."
        )
    )
    parser.add_argument("--data-dir", help="Directory for isolated smoke state.")
    parser.add_argument(
        "--timeout", type=float, default=5.0, help="Job timeout seconds."
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Only print final result."
    )
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        _configure_isolated_env(data_dir)
        asyncio.run(_run_smoke(data_dir, args.timeout, not args.quiet))
        print(f"API smoke passed. data_dir={data_dir}")
        return 0

    with tempfile.TemporaryDirectory(prefix="cogdoc-smoke-") as tmp:
        data_dir = Path(tmp)
        _configure_isolated_env(data_dir)
        asyncio.run(_run_smoke(data_dir, args.timeout, not args.quiet))
        print("API smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
