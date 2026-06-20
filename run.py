import sys  
import os 
import signal
try:
    import readline  # 启用 input() 的方向键、历史记录和行内编辑。
except ImportError:
    readline = None
from tools.rust_core_loader import ensure_rust_core
from tools.manifest import load_index_manifest, manifests_match, save_index_manifest, stamp_chunk_identity_contract

try:
    rust_core = ensure_rust_core("scan_pdf_manifest_native", "rrf_fusion_native")
except RuntimeError as exc:
    print(f"❌ {exc}")
    raise SystemExit(1) from exc
from graph.workflow import app
from graph.subgraphs.qa import RetrieverFactory
from tools.parser import smart_parse 
from tools.chunker import chunk_paper
from tools.embedder import Embedder
from tools.reranker import BGEReranker 

DEFAULT_DOC_DIR = os.getenv("COGDOC_DOC_DIR", "测试论文")

def safe_print_on_interrupt(message: str) -> None:
    # 打印退出提示时临时忽略 SIGINT，避免 Ctrl+C 连按打断清理路径。
    previous_handler = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print(message)
    finally:
        signal.signal(signal.SIGINT, previous_handler)

def _index_is_current(doc_id: str, doc_dir: str, engine):
    # 任一路索引缺失都视为不可复用。
    if not (engine.vector_retriever.exists() and engine.bm25_retriever.exists()):
        return False, None
    if not os.path.exists(doc_dir):
        return False, None

    abs_dir = os.path.abspath(doc_dir)
    current_manifest = stamp_chunk_identity_contract(rust_core.scan_pdf_manifest_native(doc_id, abs_dir))
    return manifests_match(current_manifest, load_index_manifest(doc_id)), current_manifest

def warm_up_runtime(engine) -> None:
    print("🧠 正在预热检索与重排模型，请稍候...")
    Embedder.get_model()
    engine.bm25_retriever.warm_up()
    BGEReranker.warm_up()
    print("✅ 模型与分词资源预热完成。")

def build_index(doc_id: str, doc_dir: str = DEFAULT_DOC_DIR, current_manifest: dict = None):
    # 重建索引时同步刷新 manifest 和 chunk 身份契约。
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)
        RetrieverFactory.get_engine(doc_id).clear()
        print(f"📁 已自动为您创建待扫描的知识库目录: 【{doc_dir}】")
        print(f"📌 请将您的 PDF 论文或文档放入该目录下，然后重新拉起控制台。")
        return

    pdf_files = sorted(f for f in os.listdir(doc_dir) if f.lower().endswith(".pdf"))
    if not pdf_files:
        RetrieverFactory.get_engine(doc_id).clear()
        print(f"⚠️ 提示: 目录 【{doc_dir}】 当前为空，未检测到任何 PDF 文件。")
        return

    if current_manifest is None:
        abs_dir = os.path.abspath(doc_dir)
        current_manifest = rust_core.scan_pdf_manifest_native(doc_id, abs_dir)
    # chunk_id 依赖 manifest 里的文件 sha。
    current_manifest = stamp_chunk_identity_contract(current_manifest)
    source_hash_by_name = {
        doc["name"]: doc["sha256"]
        for doc in current_manifest.get("documents", [])
    }

    all_chunks = []
    # chunk_index 只用于展示，chunk_id 才是身份键。
    next_chunk_index = 0
    print("📚 检测到静态索引缺失或过期，开始构建知识库多轨索引...")

    for pdf in pdf_files:
        pdf_path = os.path.join(doc_dir, pdf)
        print(f" 正在深度解析文档: {pdf}")
        
        pages = smart_parse(pdf_path)
        chunks = chunk_paper(pages, source_sha256 = source_hash_by_name[pdf])
        for chunk in chunks:
            chunk["meta"]["chunk_index"] = next_chunk_index
            next_chunk_index += 1
        print(f"  -> 成功抽取 {len(chunks)} 个语义 Chunk")

        print("\n--- [✂️ 离线切块细节预览开始] ---")
        for i, chunk in enumerate(chunks):
            meta = chunk.get("meta", {})
            source = meta.get("source", pdf)
            page = meta.get("page", 1)
            chunk_idx = meta.get("chunk_index", i)
            
            print(f"📄 [Chunk #{chunk_idx}] 来源: {source} | 页码: P{page}")
            print(f"📝 文本内容:\n{chunk['text'].strip()}")
            print("-" * 30)
        print("--- [✂️ 离线切块细节预览结束] ---\n")

        all_chunks.extend(chunks)

    if not all_chunks:
        print("⚠️ 未从文档中提取到有效的文本片段，放弃索引构建。")
        return

    print(f"\n📊 全量清洗完成，共生成 {len(all_chunks)} 个标准知识片段。")
    print("正在写入双轨融合索引（Vector + BM25）...")
    
    engine = RetrieverFactory.get_engine(doc_id)
    engine.index(all_chunks)

    save_index_manifest(current_manifest)

    print("✅ 物理多轨索引构建成功并落盘，数据管线安全关闭。\n")

def ask(doc_id: str, query: str, is_local: bool = False):
    # 控制台运行时只输出通过引用校验的最终答案。
    initial_state = {
        "messages": [],
        "iteration_count": 0,
        "max_iteration_count": 2
    }
    
    runtime_config = {
        "configurable": {
            "doc_id": doc_id,
            "query": query,
            "is_local": is_local
        }
    }

    try:
        print(f"\n[运行模式]: {'本地 Ollama' if is_local else '云端 API'}")
        
        token_stream = app.stream(
            initial_state,
            config = runtime_config,
            stream_mode = ["messages", "updates"],
            subgraphs = True,
        )
        
        current_task = "qa"
        saw_parent_output = False
        fallback_outputs = {
            "qa": {},
            "summary": {},
            "compare": {},
        }

        def print_final_qa_output(subgraph_output: dict) -> None:
            final_docs = subgraph_output.get("reranked_docs", [])

            print("\n🎯 [RAG 召回与精排切块结果预览]:")
            if not final_docs:
                print("  （未检索到任何相关的参考本地知识库内容。）")
            else:
                for idx, doc in enumerate(final_docs):
                    meta = doc.get("meta", {})
                    retrieval_info = doc.get("retrieval", {})
                    print(f"  📍 [{idx+1}] 来源: {meta.get('source')} | 页码: P{meta.get('page')}")
                    print(f"     📊 融合得分(RRF): {retrieval_info.get('rrf_score', 'N/A')}")
                    preview_text = doc['text'].strip().replace('\n', ' ')
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

        def print_final_compare_output(subgraph_output: dict) -> None:
            messages = subgraph_output.get("messages", [])
            content = ""
            if messages:
                latest = messages[-1]
                content = latest.get("content", "") if isinstance(latest, dict) else getattr(latest, "content", "")
            content = content or subgraph_output.get("answer", "")

            print("\n📊 [CompareAgent 文档对比]:")
            if content:
                print(f"\n{content}")
            else:
                print("\n⚠️ [CompareAgent]: 对比子图未返回可打印内容。")
            print("=" * 50)

        parent_printers = {
            "qa_subgraph": ("qa", print_final_qa_output),
            "summary_subgraph": ("summary", print_final_summary_output),
            "compare_subgraph": ("compare", print_final_compare_output),
        }
        fallback_printers = {
            task_name: printer
            for task_name, printer in parent_printers.values()
        }

        try:
            for ns, mode, data in token_stream:
                in_subgraph = len(ns) > 0

                if mode == "updates" and not in_subgraph and "intent_router" in data:
                    router_output = data["intent_router"]
                    task = router_output.get("task_type", "qa")
                    current_task = task
                    reason = router_output.get("router_reason", "无")
                    print(f"🧠 [RouterAgent 智能路由判别报告]:")
                    print(f"   ↳ 判定任务类型 -> 【{task.upper()}】")
                    print(f"   ↳ 判定分类逻辑 -> {reason}\n")
                    print("-" * 50)
                elif mode == "updates" and in_subgraph and "rewrite_node" in data:
                    rewrite_output = data["rewrite_node"]
                    queries = rewrite_output.get("rewritten_queries", [])
                    print(f"🔮 [QueryRewriteAgent 多路改写报告]:")
                    for q_idx, q in enumerate(queries):
                        print(f"   ├── ➔ 检索分支 #{q_idx+1}: '{q}'")
                    print("   └── 🚀 正在拉起多路并行召回与全局大去重机制...")
                    print("-" * 50)
                elif mode == "updates" and in_subgraph and "citation_node" in data:
                    citation_output = data["citation_node"]
                    fallback_outputs["qa"].update(citation_output)
                    critique = citation_output.get("critique", "")
                    iter_num = citation_output.get("iteration_count", 1)
                    max_iter = citation_output.get("max_iteration_count", initial_state.get("max_iteration_count", 2))

                    if critique:
                        if "未标出任何知识来源" in critique:
                            reject_title = f"模型第 {iter_num} 轮回答未添加任何引用标签"
                        else:
                            reject_title = f"模型第 {iter_num} 轮回答包含捏造引证（错误页码/文件名）"
                        print(f"\n🚨 [CitationAgent 拒绝]: {reject_title}")

                        round_answer = fallback_outputs["qa"].get("answer", "")
                        if round_answer:
                            preview = round_answer.replace("\n", " ")[:200]
                            print(f"   ↳ 本轮模型回答预览 >>> {preview}{'...' if len(round_answer) > 200 else ''}")

                        print(f"   ↳ 拒绝原因 >>> {critique}")
                        if iter_num < max_iter:
                            print(f"   ↳ 🔄 正在强行打回控制流，驱使大模型执行自愈修正...")
                        else:
                            print(f"   ↳ ⛔ 已达到最大自愈次数，系统将拦截本轮未通过校验的答案。")
                        print("-" * 50)
                    else:
                        print(f"\n🛡️ [CitationAgent 审计通过]: 第 {iter_num} 轮回答物理引用契约完全匹配，无名义页码幻觉。")
                        print("-" * 50)
                elif mode == "updates" and in_subgraph:
                    task_output = fallback_outputs.setdefault(current_task, {})
                    for value in data.values():
                        if isinstance(value, dict):
                            task_output.update(value)
                elif mode == "updates" and not in_subgraph:
                    for parent_key, (task_name, printer) in parent_printers.items():
                        if parent_key in data:
                            saw_parent_output = True
                            printer(data[parent_key])
                            break

            if not saw_parent_output:
                task_output = fallback_outputs.get(current_task, {})
                if task_output:
                    fallback_printers.get(current_task, print_final_compare_output)(task_output)
                        
        except Exception as stream_err:
            print(f"\n⚠️ [大模型流式通信管道在运行时遭遇意外中断]: {stream_err}")
            
        print("\n" + "-" * 50)
        
    except Exception as e:
        print(f"\n❌ [Pipeline 核心图调度执行失败]: {e}")

def main():
    # CLI 默认绑定一个本地知识库隔离域。
    TARGET_DOC_ID = "arch_blueprint_2026"
    TARGET_DOC_DIR = DEFAULT_DOC_DIR
    
    engine = RetrieverFactory.get_engine(TARGET_DOC_ID)

    index_is_current, current_manifest = _index_is_current(TARGET_DOC_ID, TARGET_DOC_DIR, engine)
    if not index_is_current:
        print(f"⚠️ 预检提示: 知识库 【{TARGET_DOC_ID}】 的本地索引缺失或已过期。")
        try:
            build_index(TARGET_DOC_ID, TARGET_DOC_DIR, current_manifest)
            if not (engine.vector_retriever.exists() and engine.bm25_retriever.exists()):
                print("❌ 错误: 知识库目录中由于缺乏源文件，无法建立有效索引。")
                input("⚙️ 请在上述目标文件夹内放入 PDF 文档后，按回车键退出并重新拉起系统...")
                sys.exit(1)
        except Exception as e:
            print(f"❌ 自动化索引流构建失败: {e}")
            sys.exit(1)
    else:
        print(f"✅ 知识库 【{TARGET_DOC_ID}】 索引与源PDF哈希指纹完全一致，无需重建。")

    try:
        warm_up_runtime(engine)
    except Exception as e:
        print(f"⚠️ 预热阶段失败，稍后提问时仍会尝试按需加载: {e}")

    print("=" * 60)
    print(f"🚀 RAG 问答控制台 | LangGraph 原生大一统流范式 | 隔离域: {TARGET_DOC_ID}")
    print("   - 输入 'exit' 或 'quit' 优雅退出问答系统")
    print("   - 输入 '/local' 快捷切换为 本地 Ollama 调试模式")
    print("   - 输入 '/cloud' 快捷切换为 云端 API 高性能生产模式")
    print("=" * 60)

    is_local = True
    
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

            ask(doc_id = TARGET_DOC_ID, query = user_input, is_local = is_local)
            
        except KeyboardInterrupt:
            safe_print_on_interrupt("\n👋 检测到系统中断信号（Ctrl+C），安全关闭。")
            break
        except Exception as e:
            print(f"⚠️ [控制台内部异常捕获]: {e}")

if __name__ == "__main__":
    main()
