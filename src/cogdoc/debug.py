import argparse
import atexit
import os
import signal
import sys

try:
    import readline
except ImportError:
    readline = None

from cogdoc.agents.conversation_memory import (
    CHAT_HISTORY_MESSAGE_LIMIT,
    extract_final_answer,
)
from cogdoc.api.ingest import KnowledgeBaseRegistry
from cogdoc.config.settings import get_settings
from cogdoc.graph.subgraphs.qa import RetrieverFactory
from cogdoc.graph.workflow import UNKNOWN_RESPONSE
from cogdoc.observability.logger import configure_logging
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
from cogdoc.tools.reranker import BGEReranker
from cogdoc.tools.rust_core_loader import ensure_rust_core

rust_core = None


# 释放 release runtime lock 相关逻辑。
def _release_runtime_lock(lock_fh) -> None:
    # 仅在后台 Timer 确已排空时显式释放锁；否则留给进程退出由 OS 释放。
    if cancel_all_timers():
        release_single_instance_lock(lock_fh)


# 获取 get rust core 相关逻辑。
def get_rust_core():
    global rust_core
    if rust_core is None:
        rust_core = ensure_rust_core("scan_pdf_manifest_native", "rrf_fusion_native")
    return rust_core


# 处理 safe print on interrupt 相关逻辑。
def safe_print_on_interrupt(message: str) -> None:
    # 打印退出提示时临时忽略 SIGINT，避免 Ctrl+C 连按打断清理路径。
    previous_handler = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print(message)
    finally:
        signal.signal(signal.SIGINT, previous_handler)


# 解析 parse forced mode 相关逻辑。
def parse_forced_mode(user_input: str) -> tuple[str | None, str]:
    from cogdoc.cli import FORCED_MODE_PATTERN

    match = FORCED_MODE_PATTERN.match(user_input.strip())
    if not match:
        return None, user_input
    return match.group(1).lower(), (match.group(2) or "").strip()


# 写入索引 index is current 相关逻辑。
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


# 预热 warm up runtime 相关逻辑。
def warm_up_runtime(engine) -> None:
    print("🧠 正在预热检索与重排模型，请稍候...")
    Embedder.get_model()
    engine.bm25_retriever.warm_up()
    BGEReranker.warm_up()
    print("✅ 模型与分词资源预热完成。")


# 构建 build index 相关逻辑。
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


# 输出 print final qa output 相关逻辑。
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
            print(f"     📊 融合得分(RRF): {retrieval_info.get('rrf_score', 'N/A')}")
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


# 输出 print final summary output 相关逻辑。
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


# 输出 print final compare output 相关逻辑。
def print_final_compare_output(subgraph_output: dict) -> None:
    content = extract_final_answer("compare", subgraph_output)

    print("\n📊 [CompareAgent 文档对比]:")
    if content:
        print(f"\n{content}")
    else:
        print("\n⚠️ [CompareAgent]: 对比子图未返回可打印内容。")
    print("=" * 50)


# 输出 print final unknown output 相关逻辑。
def print_final_unknown_output(subgraph_output: dict) -> None:
    content = extract_final_answer("unknown", subgraph_output) or UNKNOWN_RESPONSE
    print(f"\n🤖 [AI]: {content}")
    print("=" * 50)


# 渲染 render chat event 相关逻辑。
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


# 处理 ask 相关逻辑。
def ask(
    doc_id: str,
    query: str,
    is_local: bool = False,
    chat_history: list = None,
    forced_task: str | None = None,
):
    # 输出检索全过程与最终答案。
    print(f"\n[运行模式]: {'本地 Ollama' if is_local else '云端 API'}")
    final_result = None
    for event in run_chat(
        doc_id=doc_id,
        query=query,
        is_local=is_local,
        chat_history=chat_history,
        forced_task=forced_task,
    ):
        rendered = render_chat_event(event)
        if rendered is not None:
            final_result = rendered
    if final_result is not None:
        return final_result.chat_messages
    return []


# 解析 resolve source dir 相关逻辑。
def _resolve_source_dir(kb_id: str) -> str:
    # 已注册 KB 用其隔离源目录；否则回退到 COGDOC_DOC_DIR。
    registry = KnowledgeBaseRegistry()
    if registry.exists(kb_id):
        return registry.source_dir(kb_id)
    return get_settings().cogdoc_doc_dir


# 列出知识库 pdf 相关逻辑。
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


# 打印知识库 pdf 相关逻辑。
def print_kb_pdfs(kb_id: str, source_dir: str) -> None:
    # 展示当前 debug 知识库内的 PDF 文件名。
    pdfs = _list_kb_pdfs(kb_id, source_dir)
    if not pdfs:
        print("（当前知识库还没有 PDF。）")
        return
    print(f"📄 知识库 【{kb_id}】 PDF:")
    for name in pdfs:
        print(f"   • {name}")


# 处理 main 相关逻辑。
def main():
    parser = argparse.ArgumentParser(
        description="CogDoc 检索可视化 / 可观测控制台"
    )
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

    print("=" * 60)
    print(f"🔬 CogDoc 检索可视化 Debug 控制台 | 隔离域: {TARGET_DOC_ID}")
    print("   - 打印路由判别 / 多路改写 / 召回精排切块(含 RRF 分) / 引证审计全过程")
    print("       · /qa <问题>            强制问答")
    print("       · /summary <文件名>     强制总结指定文档")
    print("       · /compare <文件A> <文件B> ...  强制对比多篇文档（≥2，本地模式限 2）")
    print("       · /ls                  查看当前知识库 PDF")
    print("   - 输入 '/local' / '/cloud' 切换运行模式")
    print("   - 输入 'exit' 或 'quit' 退出")
    print("=" * 60)

    is_local = True
    chat_history = []

    while True:
        try:
            mode_str = "本地Ollama" if is_local else "云端API"
            user_input = input(f"[{mode_str}] 请输入您的问题 >>> ").strip()

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

            forced_task, cleaned_query = parse_forced_mode(user_input)
            if forced_task and not cleaned_query:
                print(f"⚠️ 请输入 /{forced_task} 后面的具体问题或文档指令。")
                continue

            new_messages = ask(
                doc_id=TARGET_DOC_ID,
                query=cleaned_query,
                is_local=is_local,
                chat_history=chat_history,
                forced_task=forced_task,
            )
            chat_history.extend(new_messages)
            chat_history = chat_history[-CHAT_HISTORY_MESSAGE_LIMIT:]

        except KeyboardInterrupt:
            safe_print_on_interrupt("\n👋 检测到系统中断信号（Ctrl+C），安全关闭。")
            break
        except Exception as e:
            print(f"⚠️ [控制台内部异常捕获]: {e}")

    _release_runtime_lock(lock_fh)


if __name__ == "__main__":
    main()
