import argparse
import atexit
import json
import os
import re
import signal
import sys
from uuid import uuid4

try:
    import readline
except ImportError:
    readline = None

from cogdoc.agents.conversation_memory import (
    CHAT_HISTORY_MESSAGE_LIMIT,
    extract_final_answer,
)
from cogdoc.api.ingest import KnowledgeBaseRegistry
from cogdoc.command_modes import parse_forced_mode
from cogdoc.config.settings import get_settings
from cogdoc.graph.subgraphs.qa import RetrieverFactory
from cogdoc.graph.workflow import UNKNOWN_RESPONSE
from cogdoc.observability.logger import configure_logging
from cogdoc.observability.trace import trace_dir, trace_path
from cogdoc.service.chat_service import ChatEvent, ChatResult, run_chat
from cogdoc.service.ingest_service import (
    build_kb_index_transactional,
    cancel_all_timers,
    drain_purge_queue,
    stamp_index_build_version,
)
from cogdoc.service.kb_state import KBState
from cogdoc.service.mutation_journal import shared_mutation_journal
from cogdoc.service.process_lock import (
    acquire_single_instance_lock,
    locking_supported,
    release_single_instance_lock,
    strict_single_process,
)
from cogdoc.tools.embedder import Embedder
from cogdoc.tools.manifest import (
    load_index_manifest,
    manifests_match,
    stamp_chunk_identity_contract,
)
from cogdoc.tools.reranker import BGEReranker, skipped_cpu_rerank_docs
from cogdoc.tools.rust_core_loader import ensure_rust_core

rust_core = None
DEBUG_COMMANDS = [
    "/trace",
    "/steps",
    "/rewrite",
    "/evidence",
    "/config",
    "/ls",
    "/local",
    "/cloud",
    "/qa",
    "/retrieve",
    "/summary",
    "/compare",
    "/help",
    "exit",
    "quit",
]
DEBUG_HELP_TEXT = """\
调试模式命令：
    /trace [trace_id]       查看最近一次或指定 trace 的摘要
    /steps                  查看最近一次 trace 的节点耗时
    /rewrite                查看最近一次问题改写结果
    /evidence               查看最近一次检索/重排证据摘要
    /config                 查看最近一次请求配置
    /ls                     查看当前知识库 PDF
    /local  /cloud          切换本地 Ollama / 云端 API
    /qa <问题>              调试模式下强制问答
    /retrieve <问题>        只执行召回与重排，不调用 LLM
    /summary <文件名>       调试模式下强制总结
    /compare <A> <B> ...    调试模式下强制对比
    /help                   显示本帮助
直接输入文本 = 继续问答，并在答案后显示本次 trace 摘要。\
"""
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
TRACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


# 释放运行时锁。
def _release_runtime_lock(lock_fh) -> None:
    # 仅在后台 Timer 确已排空时显式释放锁；否则留给进程退出由 OS 释放。
    if cancel_all_timers():
        release_single_instance_lock(lock_fh)


# 加载并返回 Rust 原生扩展模块。
def get_rust_core():
    global rust_core
    if rust_core is None:
        rust_core = ensure_rust_core("scan_pdf_manifest_native", "rrf_fusion_native")
    return rust_core


# 在中断信号期间安全输出提示。
def safe_print_on_interrupt(message: str) -> None:
    # 打印退出提示时临时忽略 SIGINT，避免 Ctrl+C 连按打断清理路径。
    previous_handler = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print(message)
    finally:
        signal.signal(signal.SIGINT, previous_handler)


# 定义DebugSession。
class DebugSession:
    # 独立 debug 控制台的 trace 命令和展示逻辑。
    def __init__(self):
        self.last_trace: dict | None = None

    # 格式化耗时。
    def format_duration(self, duration_ms) -> str:
        if duration_ms is None:
            return ""
        try:
            value = float(duration_ms)
        except (TypeError, ValueError):
            return str(duration_ms)
        if value >= 1000:
            return f"{value / 1000:.1f}s"
        return f"{value:.0f}ms"

    # 提取 trace 节点短名。
    def trace_node_key(self, node_name: str) -> str:
        tail = (node_name or "").rsplit(".", 1)[-1]
        return tail.split(":", 1)[0] if ":" in tail else tail

    # 构建 trace 步骤展示名。
    def trace_step_title(self, step: dict, idx: int) -> str:
        node_name = str(step.get("node_name") or f"step-{idx + 1}")
        node_key = self.trace_node_key(node_name)
        title = TRACE_NODE_LABELS.get(node_key, node_key)
        duration = self.format_duration(step.get("duration_ms"))
        suffix = f" · {duration}" if duration else ""
        return f"{idx + 1}. {title} · {node_key}{suffix}"

    # 从运行结果构造 debug 控制台可打印的 trace 载荷。
    def trace_from_result(
        self,
        result: ChatResult,
        query: str = "",
        doc_id: str = "",
        is_local: bool = False,
        forced_task: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        return {
            "trace_id": result.trace_id,
            "request_id": result.request_id,
            "task_type": result.task_type,
            "status": "ok" if result.is_valid else "blocked",
            "config": {
                "doc_id": doc_id,
                "session_id": session_id or "",
                "query_preview": " ".join((query or "").split())[:80],
                "is_local": is_local,
                "forced_task": forced_task,
            },
            "steps": list(result.steps or []),
            "trace_path": result.trace_path,
        }

    # 记录最近一次请求。
    def remember_result(
        self,
        result: ChatResult,
        query: str = "",
        doc_id: str = "",
        is_local: bool = False,
        forced_task: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        if result.trace_path:
            payload = self.load_trace_file(result.trace_path)
            if payload is not None:
                self.last_trace = payload
                return self.last_trace
        self.last_trace = self.trace_from_result(
            result,
            query=query,
            doc_id=doc_id,
            is_local=is_local,
            forced_task=forced_task,
            session_id=session_id,
        )
        return self.last_trace

    # 加载指定 trace 文件路径。
    def load_trace_file(self, path_text: str) -> dict | None:
        try:
            with open(path_text, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"⚠️ trace 文件读取失败: {exc}")
            return None
        payload.setdefault("trace_path", path_text)
        return payload

    # 加载指定 trace 文件。
    def load_trace_payload(self, trace_id: str) -> dict | None:
        if not TRACE_ID_PATTERN.fullmatch(trace_id):
            print(f"⚠️ trace_id 不合法: {trace_id}")
            return None
        path = trace_path(trace_id)
        if not path.exists() or not path.is_file():
            print(f"⚠️ trace 不存在: {trace_id}")
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"⚠️ trace 文件读取失败: {exc}")
            return None
        payload.setdefault("trace_path", str(path))
        return payload

    # 当前可用 trace。
    def current_trace(self) -> dict | None:
        if self.last_trace is None:
            print("（还没有 trace。先在 debug 控制台里问一个问题。）")
            return None
        return self.last_trace

    # 打印 trace 摘要。
    def print_trace_summary(self, trace: dict | None = None) -> None:
        trace = trace or self.current_trace()
        if trace is None:
            return
        steps = list(trace.get("steps") or [])
        summary = trace.get("summary") or {}
        config = trace.get("config") or {}
        duration = self.format_duration(trace.get("duration_ms"))
        duration_text = f" · {duration}" if duration else ""
        trace_id = str(trace.get("trace_id") or "-")[:8]
        print(
            f"\nTrace: {trace_id} · "
            f"{trace.get('task_type', '-')} · {trace.get('status', '-')}"
            f"{duration_text} · {summary.get('step_count', len(steps))} 步"
        )
        query = str(config.get("query_preview") or "").strip()
        if query:
            print(f"问题: {query}")
        if config:
            doc_id = config.get("doc_id") or "-"
            session_id = config.get("session_id") or "-"
            model = config.get("model") or "-"
            mode = "本地" if config.get("is_local") else "云端"
            forced = config.get("forced_task") or "-"
            print(
                f"配置: kb={doc_id} · session={session_id} · "
                f"mode={mode} · forced={forced} · model={model}"
            )
        if trace.get("trace_path"):
            print(f"文件: {trace['trace_path']}")
        for idx, step in enumerate(steps[:12]):
            print(f"  {self.trace_step_title(step, idx)}")
        if len(steps) > 12:
            print(f"  ... 还有 {len(steps) - 12} 步，输入 /steps 查看全部")
        print(
            "继续输入 /steps 查看节点，/rewrite 看改写，/evidence 看证据，/config 看配置，exit 退出。"
        )

    # 打印 trace 全部步骤。
    def print_trace_steps(self) -> None:
        trace = self.current_trace()
        if trace is None:
            return
        steps = list(trace.get("steps") or [])
        if not steps:
            print("（当前 trace 没有步骤。）")
            return
        print("\nTrace steps:")
        for idx, step in enumerate(steps):
            print(f"  {self.trace_step_title(step, idx)}")
            if step.get("node_name"):
                print(f"     原始节点: {step.get('node_name')}")
            details = []
            if step.get("task_type"):
                details.append(f"task={step.get('task_type')}")
            if step.get("model"):
                details.append(f"model={step.get('model')}")
            if step.get("retrieval_top_k") is not None:
                details.append(f"top_k={step.get('retrieval_top_k')}")
            if step.get("error_class"):
                details.append(f"error={step.get('error_class')}")
            if details:
                print(f"     {' | '.join(details)}")
            if step.get("router_reason"):
                print(f"     路由理由: {step.get('router_reason')}")
            rewritten = step.get("rewritten_queries") or []
            if rewritten:
                print("     改写查询:")
                for query in rewritten:
                    print(f"       - {query}")
            elif (step.get("counts") or {}).get("rewritten_query_count"):
                print("     改写查询: 此 trace 只记录了数量，未保存具体内容。")
            if step.get("critique"):
                print(f"     校验反馈: {step.get('critique')}")
            counts = step.get("counts") or {}
            if counts:
                print(f"     计数: {json.dumps(counts, ensure_ascii=False)}")
            step_traces = step.get("steps_trace") or []
            if step_traces:
                print("     调试摘要:")
                for item in step_traces[:3]:
                    name = item.get("step_name") or "-"
                    input_summary = item.get("input_summary") or ""
                    output_summary = item.get("output_summary") or ""
                    print(f"       - {name}")
                    if input_summary:
                        print(f"         input: {input_summary}")
                    if output_summary:
                        print(f"         output: {output_summary}")
            evidence = step.get("evidence") or []
            if evidence:
                print("     证据预览:")
                for item in evidence[:5]:
                    source = item.get("source") or "-"
                    page = item.get("page") or item.get("page_start") or "-"
                    chunk_id = item.get("chunk_id") or "-"
                    preview = item.get("text_preview") or ""
                    print(f"       - {source}:P{page} · {chunk_id}")
                    if preview:
                        print(f"         {preview}")

    # 解析 JSON 字段，失败时返回原文本。
    def _parse_json_summary(self, value):
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    # 打印问题改写。
    def print_trace_rewrite(self) -> None:
        trace = self.current_trace()
        if trace is None:
            return
        rewrites: list[str] = []
        candidates: list[str] = []
        kept: list[str] = []
        dropped: list[dict] = []
        threshold = None
        only_count = False
        for step in trace.get("steps") or []:
            rewrites.extend(step.get("rewritten_queries") or [])
            only_count = only_count or bool(
                (step.get("counts") or {}).get("rewritten_query_count")
            )
            for item in step.get("steps_trace") or []:
                input_summary = self._parse_json_summary(item.get("input_summary"))
                output_summary = self._parse_json_summary(item.get("output_summary"))
                if isinstance(input_summary, list):
                    candidates.extend(str(query) for query in input_summary)
                if isinstance(output_summary, dict):
                    threshold = output_summary.get("threshold", threshold)
                    kept.extend(str(query) for query in output_summary.get("kept", []))
                    dropped.extend(output_summary.get("dropped", []) or [])
        rewrites = list(dict.fromkeys(rewrites))
        candidates = list(dict.fromkeys(candidates))
        kept = list(dict.fromkeys(kept))
        if rewrites:
            print("\n改写查询:")
            for query in rewrites:
                print(f"  - {query}")
        if candidates:
            print("\n校验前候选:")
            for query in candidates:
                print(f"  - {query}")
        if kept:
            print("\n校验后保留:")
            for query in kept:
                print(f"  - {query}")
        if dropped:
            print("\n相似度过滤:")
            if threshold is not None:
                print(f"  阈值: {threshold}")
            for item in dropped:
                if isinstance(item, dict):
                    print(
                        f"  - {item.get('query', '-')} "
                        f"(similarity={item.get('similarity', '-')})"
                    )
                else:
                    print(f"  - {item}")
        if not any((rewrites, candidates, kept, dropped)) and only_count:
            print("当前 trace 只记录了改写数量，未保存具体改写查询。")
        elif not any((rewrites, candidates, kept, dropped)):
            print("（当前 trace 没有问题改写记录。）")

    # 打印证据摘要。
    def print_trace_evidence(self) -> None:
        trace = self.current_trace()
        if trace is None:
            return
        evidence = []
        for step in trace.get("steps") or []:
            for item in step.get("evidence") or []:
                evidence.append((step.get("node_name", ""), item))
        if not evidence:
            print("（当前 trace 没有证据摘要。）")
            return
        print("\n证据摘要:")
        for idx, (node, item) in enumerate(evidence[:20], start=1):
            source = item.get("source") or "-"
            page = item.get("page") or item.get("page_start") or "-"
            chunk_id = item.get("chunk_id") or "-"
            preview = item.get("text_preview") or ""
            rewrite_query = item.get("rewrite_query")
            suffix = f" · rewrite={rewrite_query}" if rewrite_query else ""
            print(
                f"  {idx}. {source}:P{page} · {chunk_id} · "
                f"{self.trace_node_key(node)}{suffix}"
            )
            if preview:
                print(f"     {preview}")

    # 打印请求配置。
    def print_trace_config(self) -> None:
        trace = self.current_trace()
        if trace is None:
            return
        config = trace.get("config") or {}
        if not config:
            print("（当前 trace 没有请求配置。）")
            return
        print("\n请求配置:")
        print(json.dumps(config, ensure_ascii=False, indent=2))

    # 打印 debug 帮助。
    def print_help(self) -> None:
        print(DEBUG_HELP_TEXT)

    # 执行带 debug 输出的问答。
    def ask(
        self,
        doc_id: str,
        query: str,
        is_local: bool = False,
        chat_history: list | None = None,
        forced_task: str | None = None,
        session_id: str | None = None,
        render_event=None,
        show_trace_summary: bool = True,
    ) -> ChatResult | None:
        final_result = None
        for event in run_chat(
            doc_id=doc_id,
            query=query,
            is_local=is_local,
            chat_history=chat_history,
            forced_task=forced_task,
            session_id=session_id,
        ):
            if render_event is not None:
                rendered = render_event(event)
                if rendered is not None:
                    final_result = rendered
                elif event.type == "final":
                    final_result = event.payload["result"]
            elif event.type == "error":
                print(f"\n⚠️ 执行中断: {event.payload.get('message', '')}")
            elif event.type == "final":
                final_result = event.payload["result"]

        if final_result is not None:
            self.remember_result(
                final_result,
                query=query,
                doc_id=doc_id,
                is_local=is_local,
                forced_task=forced_task,
                session_id=session_id,
            )
            if show_trace_summary:
                self.print_trace_summary()
        return final_result

    # 分发 debug 命令。
    def dispatch(self, text: str, ask_callback, retrieve_callback=None) -> str:
        low = text.lower()
        if low == "/help":
            self.print_help()
            return "handled"

        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        if cmd == "/trace":
            trace = self.load_trace_payload(arg) if arg else self.current_trace()
            if trace is not None:
                if arg:
                    self.last_trace = trace
                self.print_trace_summary(trace)
            return "handled"
        if cmd == "/steps":
            self.print_trace_steps()
            return "handled"
        if cmd == "/rewrite":
            self.print_trace_rewrite()
            return "handled"
        if cmd == "/evidence":
            self.print_trace_evidence()
            return "handled"
        if cmd == "/config":
            self.print_trace_config()
            return "handled"
        if cmd == "/retrieve":
            if not arg:
                print("⚠️ 请输入 /retrieve 后面的检索问题。")
                return "handled"
            if retrieve_callback is None:
                print("❓ 当前环境没有注册检索调试回调。")
                return "handled"
            retrieve_callback(arg)
            return "handled"

        forced_task, cleaned_query = parse_forced_mode(text)
        if forced_task:
            if not cleaned_query:
                print(f"⚠️ 请输入 /{forced_task} 后面的具体问题或文档指令。")
                return "handled"
            ask_callback(cleaned_query, forced_task)
            return "handled"

        if text.startswith("/"):
            print("❓ debug 控制台不处理这个命令。输入 /help 查看 debug 命令。")
            return "handled"

        ask_callback(text, None)
        return "handled"


# 写入索引 is current。
def _index_is_current(doc_id: str, doc_dir: str, engine):
    # 任一路索引缺失都视为不可复用。
    if not (engine.vector_retriever.exists() and engine.bm25_retriever.exists()):
        return False, None
    if not os.path.exists(doc_dir):
        return False, None

    abs_dir = os.path.abspath(doc_dir)
    # 与入库同样 stamp 构建版本，否则仅改 parser/tokenizer/嵌入模型时会误判索引仍有效。
    current_manifest = stamp_index_build_version(
        stamp_chunk_identity_contract(
            get_rust_core().scan_pdf_manifest_native(doc_id, abs_dir)
        )
    )
    return manifests_match(
        current_manifest, load_index_manifest(doc_id)
    ), current_manifest


# 完成 预热流程预热流程运行时 处理。
def warm_up_runtime(engine) -> None:
    print("🧠 正在预热检索与重排模型，请稍候...")
    settings = get_settings()
    Embedder.get_model()
    engine.bm25_retriever.warm_up()
    rerank_device = BGEReranker.default_device()
    if rerank_device != "cpu" or settings.qa_rerank_on_cpu:
        BGEReranker.warm_up()
    else:
        print("ℹ️ 未预热 CPU reranker；如需强制 CPU 精排，请设置 QA_RERANK_ON_CPU=true。")
    print("✅ 模型与分词资源预热完成。")


# 构建索引。
def build_index(doc_id: str, doc_dir: str):
    # 索引缺失或过期时全量重建。
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)
        RetrieverFactory.get_engine(doc_id).clear()
        print(f"📁 已自动创建待扫描的源目录: 【{doc_dir}】")
        print("📌 请将 PDF 放入该目录后重新启动。")
        return

    print("📚 检测到静态索引缺失或过期，开始构建知识库多轨索引...")
    result = build_kb_index_transactional(doc_id, doc_dir)
    if result.document_count == 0:
        print(f"⚠️ 提示: 目录 【{doc_dir}】 当前为空，未检测到任何 PDF 文件。")
        return

    for doc in result.documents:
        print(f"  -> {doc.name}: 抽取 {doc.chunk_count} 个语义 Chunk")
    print(f"\n📊 全量清洗完成，共生成 {result.chunk_count} 个标准知识片段。")
    print("✅ 物理多轨索引构建成功并落盘，数据管线安全关闭。\n")


# 输出final问答output。
def print_final_qa_output(subgraph_output: dict) -> None:
    final_docs = subgraph_output.get("reranked_docs", [])

    print("\n🎯 [RAG 召回与精排切块结果预览]:")
    if not final_docs:
        print("  （未检索到任何相关的参考本地知识库内容。）")
    else:
        for idx, doc in enumerate(final_docs):
            meta = doc.get("meta", {})
            retrieval_info = doc.get("retrieval", {})
            print(
                f"  📍 [{idx + 1}] 来源: {meta.get('source')} | 页码: P{meta.get('page')}"
            )
            print(f"     📊 精排得分: {retrieval_info.get('rerank_score', 'N/A')}")
            preview_text = doc["text"].strip().replace("\n", " ")
            print(f"     📄 核心内容: {preview_text[:120]}...")
            print("     " + "-" * 40)

    final_answer = subgraph_output.get("answer", "")
    has_critique_state = "critique" in subgraph_output
    final_critique = subgraph_output.get("critique")
    if not has_critique_state:
        print("\n⚠️ [AI]: QA 子图未返回引证校验状态，已拒绝打印未确认答案。")
    elif final_critique:
        print("\n❌ [AI]: 引证校验未通过，已达到最大自愈次数，本轮答案已拦截。")
        print(f"   ↳ 最终批注 >>>\n{final_critique}")
    elif final_answer:
        print(f"\n🤖 [AI]: {final_answer}")
    else:
        print("\n⚠️ [AI]: 模型返回了空内容，但引证校验已通过。")
    print("=" * 50)


# 格式化页码范围。
def _page_range_text(meta: dict) -> str:
    page = meta.get("page")
    page_start = meta.get("page_start", page)
    page_end = meta.get("page_end", page_start)
    if page_start is None and page_end is None:
        return "-"
    if page_end is None or page_start == page_end:
        return f"P{page_start}"
    if page_start is None:
        return f"P{page_end}"
    return f"P{page_start}-{page_end}"


# 输出检索调试结果。
def print_retrieve_debug_output(
    query: str, docs: list, reranked: bool, device: str
) -> None:
    print("\n🔎 [Retrieve Debug]")
    print(f"   query: {query}")
    print(f"   rerank: {'on' if reranked else 'off'} · device: {device}")
    if not docs:
        print("   （没有召回到任何 chunk。）")
        print("=" * 50)
        return
    for idx, doc in enumerate(docs, start=1):
        meta = doc.get("meta", {}) if isinstance(doc, dict) else {}
        retrieval = doc.get("retrieval", {}) if isinstance(doc, dict) else {}
        source = meta.get("source", "-")
        chunk_id = meta.get("chunk_id", "-")
        text = str(doc.get("text", "") if isinstance(doc, dict) else "")
        preview = " ".join(text.split())[:180]
        score_parts = []
        for key in ("rerank_score", "bm25_score", "distance"):
            if retrieval.get(key) is not None:
                score_parts.append(f"{key}={retrieval.get(key)}")
        scores = " · ".join(score_parts) if score_parts else "score=-"
        print(f"  [{idx}] {source} · {_page_range_text(meta)} · {chunk_id}")
        print(f"      {scores}")
        print(f"      {preview}...")
    print("=" * 50)


# 执行检索调试。
def run_retrieve_debug(doc_id: str, query: str) -> None:
    settings = get_settings()
    engine = RetrieverFactory.get_engine(doc_id)
    top_k = settings.qa_retrieval_top_k
    docs = engine.search(query=query, top_k=top_k)
    if not docs:
        print_retrieve_debug_output(query, [], False, "-")
        return
    target_device = BGEReranker.default_device()
    max_candidates = max(settings.qa_rerank_max_candidates, settings.qa_rerank_top_n)
    candidate_docs = docs[:max_candidates] if max_candidates > 0 else docs
    if target_device == "cpu" and not settings.qa_rerank_on_cpu:
        selected = skipped_cpu_rerank_docs(candidate_docs, settings.qa_rerank_top_n)
        print_retrieve_debug_output(query, selected, False, target_device)
        print("ℹ️ CPU 重排默认关闭；如需强制 CPU 精排，请设置 QA_RERANK_ON_CPU=true。")
        return
    reranked_docs = BGEReranker.rerank(
        query=query,
        docs=candidate_docs,
        top_n=settings.qa_rerank_top_n,
        device=target_device,
    )
    print_retrieve_debug_output(query, reranked_docs, True, target_device)


# 输出final摘要output。
def print_final_summary_output(subgraph_output: dict) -> None:
    summary_source = subgraph_output.get("summary_source", "")
    final_answer = subgraph_output.get("answer", "")
    evidence = subgraph_output.get("evidence", [])

    if summary_source:
        print(f"\n📝 [SummaryAgent 文档摘要]: {summary_source}")
    else:
        print("\n📝 [SummaryAgent 文档摘要]:")

    if final_answer:
        print(f"\n{final_answer}")
    else:
        print("\n⚠️ [SummaryAgent]: 摘要子图未返回可打印内容。")

    if evidence:
        print(f"\n📚 [摘要 Evidence]: 共 {len(evidence)} 个 chunk 参与摘要。")
    print("=" * 50)


# 输出final对比output。
def print_final_compare_output(subgraph_output: dict) -> None:
    content = extract_final_answer("compare", subgraph_output)

    print("\n📊 [CompareAgent 文档对比]:")
    if content:
        print(f"\n{content}")
    else:
        print("\n⚠️ [CompareAgent]: 对比子图未返回可打印内容。")
    print("=" * 50)


# 输出final未知意图output。
def print_final_unknown_output(subgraph_output: dict) -> None:
    content = extract_final_answer("unknown", subgraph_output) or UNKNOWN_RESPONSE
    print(f"\n🤖 [AI]: {content}")
    print("=" * 50)


# 渲染chat事件。
def render_chat_event(event: ChatEvent) -> ChatResult | None:
    if event.type == "router_decided":
        task = event.payload.get("task_type", "qa")
        reason = event.payload.get("reason", "无")
        print("🧠 [RouterAgent 智能路由判别报告]:")
        print(f"   ↳ 判定任务类型 -> 【{task.upper()}】")
        print(f"   ↳ 判定分类逻辑 -> {reason}\n")
        print("-" * 50)
    elif event.type == "rewrite_queries":
        queries = event.payload.get("queries", [])
        print("🔮 [QueryRewriteAgent 多路改写报告]:")
        for q_idx, q in enumerate(queries):
            print(f"   ├── ➔ 检索分支 #{q_idx + 1}: '{q}'")
        print("   └── 🚀 正在拉起多路并行召回与全局大去重机制...")
        print("-" * 50)
    elif event.type == "citation_rejected":
        critique = event.payload.get("critique", "")
        iter_num = event.payload.get("iteration_count", 1)
        if "未标出任何知识来源" in critique:
            reject_title = f"模型第 {iter_num} 轮回答未添加任何引用标签"
        else:
            reject_title = f"模型第 {iter_num} 轮回答包含捏造引证（错误页码/文件名）"
        print(f"\n🚨 [CitationAgent 拒绝]: {reject_title}")

        round_answer = event.payload.get("round_answer", "")
        if round_answer:
            preview = round_answer.replace("\n", " ")[:200]
            print(
                f"   ↳ 本轮模型回答预览 >>> {preview}{'...' if len(round_answer) > 200 else ''}"
            )

        print(f"   ↳ 拒绝原因 >>> {critique}")
        if event.payload.get("will_retry", False):
            print("   ↳ 🔄 正在强行打回控制流，驱使大模型执行自愈修正...")
        else:
            print("   ↳ ⛔ 已达到最大自愈次数，系统将拦截本轮未通过校验的答案。")
        print("-" * 50)
    elif event.type == "citation_passed":
        iter_num = event.payload.get("iteration_count", 1)
        print(
            f"\n🛡️ [CitationAgent 审计通过]: 第 {iter_num} 轮回答物理引用契约完全匹配，无名义页码幻觉。"
        )
        print("-" * 50)
    elif event.type == "compare_citation_rejected":
        critique = event.payload.get("critique", "")
        if "未包含任何引用标签" in critique:
            reject_reason = "对比结论未携带任何引用标签"
        else:
            reject_reason = "对比结论或单元格存在捏造引证（错误页码/文件名）"
        print(f"\n🚨 [CompareAgent 引用校验未通过]: {reject_reason}")
        print("   ↳ 已降级为纯对比表，并在答案末尾追加引用校验警告。")
        print(f"   ↳ 拒绝原因 >>> {critique}")
        print("-" * 50)
    elif event.type == "compare_citation_passed":
        print(
            "\n🛡️ [CompareAgent 审计通过]: 对比表与简短结论的引用契约完全匹配，无名义页码幻觉。"
        )
        print("-" * 50)
    elif event.type == "error":
        if event.payload.get("stage") == "stream":
            print(
                f"\n⚠️ [大模型流式通信管道在运行时遭遇意外中断]: {event.payload.get('message', '')}"
            )
        else:
            print(
                f"\n❌ [Pipeline 核心图调度执行失败]: {event.payload.get('message', '')}"
            )
    elif event.type == "final":
        result = event.payload["result"]
        output = event.payload.get("output", result.raw_output)
        printers = {
            "qa": print_final_qa_output,
            "summary": print_final_summary_output,
            "compare": print_final_compare_output,
            "unknown": print_final_unknown_output,
        }
        printers.get(result.task_type, print_final_compare_output)(output)
        print("\n" + "-" * 50)
        return result
    return None


# 提交问题结果。
def ask(
    doc_id: str,
    query: str,
    is_local: bool = False,
    chat_history: list = None,
    forced_task: str | None = None,
    session_id: str | None = None,
    debug_session: DebugSession | None = None,
):
    # 输出检索全过程与最终答案。
    print(f"\n[运行模式]: {'本地 Ollama' if is_local else '云端 API'}")
    debug_session = debug_session or DebugSession()
    final_result = debug_session.ask(
        doc_id=doc_id,
        query=query,
        is_local=is_local,
        chat_history=chat_history,
        forced_task=forced_task,
        session_id=session_id,
        render_event=render_chat_event,
        show_trace_summary=True,
    )
    if final_result is not None:
        return final_result.chat_messages
    return []


# 解析来源目录。
def _resolve_source_dir(kb_id: str) -> str:
    # 已注册 KB 用其隔离源目录；否则回退到 COGDOC_DOC_DIR。
    registry = KnowledgeBaseRegistry()
    if registry.exists(kb_id):
        return registry.source_dir(kb_id)
    return get_settings().cogdoc_doc_dir


# 列出知识库PDF 列表。
def _list_kb_pdfs(kb_id: str, source_dir: str) -> list[str]:
    # 优先读已提交 state/manifest，索引未就绪时回退扫描源目录。
    active = KBState(kb_id).active()
    docs = (
        active.get("documents", [])
        if active is not None
        else load_index_manifest(kb_id).get("documents", [])
    )
    names = [str(doc.get("name", "")) for doc in docs if doc.get("name")]
    if names:
        return sorted(names)
    if not os.path.isdir(source_dir):
        return []
    return sorted(
        name
        for name in os.listdir(source_dir)
        if name.lower().endswith(".pdf")
        and os.path.isfile(os.path.join(source_dir, name))
    )


# 输出知识库PDF 列表。
def print_kb_pdfs(kb_id: str, source_dir: str) -> None:
    # 展示当前 debug 知识库内的 PDF 文件名。
    pdfs = _list_kb_pdfs(kb_id, source_dir)
    if not pdfs:
        print("（当前知识库还没有 PDF。）")
        return
    print(f"📄 知识库 【{kb_id}】 PDF:")
    for name in pdfs:
        print(f"   • {name}")


# 配置独立 debug 控制台补全与方向键编辑。
def _setup_debug_completion(
    kb_id: str, source_dir: str, debug_session: DebugSession
) -> None:
    if readline is None:
        print("⚠️ 当前 Python 环境没有 readline，方向键编辑和 Tab 补全不可用。")
        return

    # 处理candidates。
    def _candidates(tokens: list[str]) -> list[str]:
        if not tokens:
            return DEBUG_COMMANDS
        cmd = tokens[0].lower()
        if cmd in ("/summary", "/compare"):
            return _list_kb_pdfs(kb_id, source_dir)
        if cmd == "/trace":
            trace = debug_session.last_trace or {}
            trace_id = trace.get("trace_id")
            return [str(trace_id)] if trace_id else []
        return []

    matches: list[str] = []

    # 补全_complete。
    def _complete(text: str, state: int) -> str | None:
        nonlocal matches
        try:
            if state == 0:
                buffer = readline.get_line_buffer()
                tokens = buffer[: readline.get_begidx()].split()
                matches = [
                    item
                    for item in _candidates(tokens)
                    if item and item.startswith(text)
                ]
            return matches[state] if state < len(matches) else None
        except Exception:
            return None

    readline.set_completer_delims(" \t\n")
    readline.set_completer(_complete)
    doc = getattr(readline, "__doc__", "") or ""
    if "libedit" in doc:
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")


# 启动入口。
def main():
    parser = argparse.ArgumentParser(description="CogDoc 检索可视化 / 可观测控制台")
    parser.add_argument(
        "--kb",
        default=get_settings().cogdoc_default_doc_id,
        help="目标知识库 ID（默认取 COGDOC_DEFAULT_DOC_ID）",
    )
    args = parser.parse_args()

    configure_logging()
    TARGET_DOC_ID = args.kb
    TARGET_DOC_DIR = _resolve_source_dir(TARGET_DOC_ID)

    # 与 CLI/API 共用同一把进程锁：debug 会构建索引（写），不得与运行中的实例并发写。
    lock_fh = acquire_single_instance_lock()
    if lock_fh is None and strict_single_process():
        reason = (
            "当前平台不支持进程锁，无法保证单实例"
            if not locking_supported()
            else "已有 CogDoc 实例（API 或 CLI）在运行"
        )
        print(f"❌ {reason}；如确需放行请设 COGDOC_ALLOW_MULTI=1。")
        sys.exit(1)
    atexit.register(_release_runtime_lock, lock_fh)

    shared_mutation_journal().recover_all()
    drain_purge_queue()

    try:
        get_rust_core()
    except RuntimeError as exc:
        print(f"❌ {exc}")
        _release_runtime_lock(lock_fh)
        sys.exit(1)

    engine = RetrieverFactory.get_engine(TARGET_DOC_ID)

    index_is_current, _ = _index_is_current(TARGET_DOC_ID, TARGET_DOC_DIR, engine)
    if not index_is_current:
        print(f"⚠️ 预检提示: 知识库 【{TARGET_DOC_ID}】 的本地索引缺失或已过期。")
        try:
            build_index(TARGET_DOC_ID, TARGET_DOC_DIR)
            # build 后重取引擎：transactional build 会 invalidate 旧缓存。
            engine = RetrieverFactory.get_engine(TARGET_DOC_ID)
            if not (
                engine.vector_retriever.exists() and engine.bm25_retriever.exists()
            ):
                print("❌ 错误: 源目录缺乏 PDF，无法建立有效索引。")
                _release_runtime_lock(lock_fh)
                sys.exit(1)
        except Exception as e:
            print(f"❌ 自动化索引流构建失败: {e}")
            _release_runtime_lock(lock_fh)
            sys.exit(1)
    else:
        print(f"✅ 知识库 【{TARGET_DOC_ID}】 索引与源PDF哈希指纹完全一致，无需重建。")

    try:
        warm_up_runtime(engine)
    except Exception as e:
        print(f"⚠️ 预热阶段失败，稍后提问时仍会尝试按需加载: {e}")

    is_local = True
    chat_history = []
    debug_session_id = uuid4().hex
    debug_session = DebugSession()

    print("=" * 60)
    print(f"🔬 CogDoc Debug 控制台 | 隔离域: {TARGET_DOC_ID}")
    print(f"   - 知识库 ID: {TARGET_DOC_ID}")
    print(f"   - Debug 会话 ID: {debug_session_id}")
    print(f"   - 知识库源目录: {os.path.abspath(TARGET_DOC_DIR)}")
    print(f"   - Trace 目录: {trace_dir()}")
    print("   - 直接提问会打印回答、路由/改写/检索/引用审计过程，并保存本次 trace")
    print("       · /qa <问题>            强制问答")
    print("       · /retrieve <问题>      只执行召回与重排")
    print("       · /summary <文件名>     强制总结指定文档")
    print("       · /compare <文件A> <文件B> ...  强制对比多篇文档（≥2，本地模式限 2）")
    print("       · /trace /steps /rewrite /evidence /config  查看最近一次请求 trace")
    print("       · /ls                  查看当前知识库 PDF")
    print("   - 输入 '/local' / '/cloud' 切换运行模式")
    print("   - 输入 'exit' 或 'quit' 退出")
    print("=" * 60)

    _setup_debug_completion(TARGET_DOC_ID, TARGET_DOC_DIR, debug_session)

    # 执行问答调试。
    def _ask_debug(query: str, forced_task: str | None) -> None:
        nonlocal chat_history
        new_messages = ask(
            doc_id=TARGET_DOC_ID,
            query=query,
            is_local=is_local,
            chat_history=chat_history,
            forced_task=forced_task,
            session_id=debug_session_id,
            debug_session=debug_session,
        )
        chat_history.extend(new_messages)
        chat_history = chat_history[-CHAT_HISTORY_MESSAGE_LIMIT:]

    # 执行检索调试。
    def _retrieve_debug(query: str) -> None:
        run_retrieve_debug(TARGET_DOC_ID, query)

    while True:
        try:
            mode_str = "本地Ollama" if is_local else "云端API"
            user_input = input(
                f"[debug|{TARGET_DOC_ID}|{debug_session_id[:8]}|{mode_str}] >>> "
            ).strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("👋 接收到安全退出指令，控制台正在释放资源，再见。")
                break

            if user_input.lower() == "/local":
                is_local = True
                print("🔄 控制面配置已成功切换到：本地 Ollama 模式。")
                continue

            elif user_input.lower() == "/cloud":
                is_local = False
                print("🔄 控制面配置已成功切换到：云端 API 模式。")
                continue

            elif user_input.lower() == "/ls":
                print_kb_pdfs(TARGET_DOC_ID, TARGET_DOC_DIR)
                continue

            debug_session.dispatch(user_input, _ask_debug, _retrieve_debug)

        except KeyboardInterrupt:
            safe_print_on_interrupt("\n👋 检测到系统中断信号（Ctrl+C），安全关闭。")
            break
        except EOFError:
            print("\n👋 输入流结束，控制台正在释放资源，再见。")
            break
        except Exception as e:
            print(f"⚠️ [控制台内部异常捕获]: {e}")

    _release_runtime_lock(lock_fh)


if __name__ == "__main__":
    main()
