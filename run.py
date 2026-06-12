import sys
import os
from graph.workflow import app, RetrieverFactory
from tools.parser import smart_parse
from tools.chunker import chunk_paper

def build_index(doc_id: str, doc_dir: str = "tests"):
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)
        print(f"📁 已自动为您创建待扫描的知识库目录: 【{doc_dir}】")
        print(f"📌 请将您的 PDF 论文或文档放入该目录下，然后重新拉起控制台。")
        return

    pdf_files = [f for f in os.listdir(doc_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"⚠️ 提示: 目录 【{doc_dir}】 当前为空，未检测到任何 PDF 文件。")
        return

    all_chunks = []
    print("📚 检测到静态索引缺失，开始构建知识库多轨索引...")

    for pdf in pdf_files:
        pdf_path = os.path.join(doc_dir, pdf)
        print(f" 正在深度解析文档: {pdf}")
        
        # 依次调用你的底层解析器与分块器
        pages = smart_parse(pdf_path)
        chunks = chunk_paper(pages)
        print(f"  -> 成功抽取 {len(chunks)} 个语义 Chunk")

        print("\n--- [切块细节预览开始] ---")
        for i, chunk in enumerate(chunks):
            # 从 metadata 提取源文件和页码
            meta = chunk.get("meta", {})
            source = meta.get("source", pdf)
            page = meta.get("page", 1)
            chunk_idx = meta.get("chunk_index", i)
            
            print(f"📄 [Chunk #{chunk_idx}] 来源: {source} | 页码: P{page}")
            print(f"📝 文本内容:\n{chunk['text'].strip()}")
            print("-" * 30)
        print("--- [切块细节预览结束] ---\n")

        all_chunks.extend(chunks)

    if not all_chunks:
        print("⚠️ 未从文档中提取到有效的文本片段，放弃索引构建。")
        return

    print(f"\n📊 全量清洗完成，共生成 {len(all_chunks)} 个标准知识片段。")
    print("正在写入双轨融合索引（Vector + BM25）...")
    
    # 获取静态单例工厂并执行落盘
    engine = RetrieverFactory.get_engine(doc_id)
    engine.index(all_chunks)
    print("✅ 物理多轨索引构建成功并落盘，数据管线安全关闭。\n")


def ask(doc_id: str, query: str, is_local: bool = False):
    """
    100% 正统的 LangGraph 原生消息流消费客户端。
    通过隐藏的控制槽 config 隔离控制面，通过核心状态轨保持数据纯净。
    """
    initial_state = {"messages": []}
    
    runtime_config = {
        "configurable": {
            "doc_id": doc_id,
            "query": query,
            "is_local": is_local
        }
    }

    try:
        print(f"\n[运行模式]: {'本地 Ollama' if is_local else '云端 API'}")
        print("🤖 [AI]: ", end="", flush=True)
        
        # 利用 app.stream 拉起大一统的消息增量流
        token_stream = app.stream(initial_state, config=runtime_config, stream_mode="messages")
        
        try:
            for chunk, _ in token_stream:
                # 统一抽象消费：不看节点，只看是否有 content 属性
                if hasattr(chunk, "content") and chunk.content:
                    print(chunk.content, end="", flush=True)
                        
        except Exception as stream_err:
            print(f"\n⚠️ [大模型流式通信管道在运行时遭遇意外中断]: {stream_err}")
            
        print("\n" + "-" * 50)
        
    except Exception as e:
        print(f"\n❌ [Pipeline 核心图调度执行失败]: {e}")


def main():
    TARGET_DOC_ID = "arch_blueprint_2026"
    
    # 静态预检
    engine = RetrieverFactory.get_engine(TARGET_DOC_ID)

    # 启动前严密检查双轨索引的物理存续状态
    if not (engine.vector_retriever.exists() and engine.bm25_retriever.exists()):
        print(f"⚠️ 预检提示: 未发现知识库 【{TARGET_DOC_ID}】 的本地完整索引。")
        try:
            build_index(TARGET_DOC_ID)
            # 重新刷一下物理状态，如果依然不存在（比如空文件夹情况），则安全拦截
            if not (engine.vector_retriever.exists() and engine.bm25_retriever.exists()):
                print("❌ 知识库目录中由于缺乏源文件，无法建立索引。请放入 PDF 后重试。")
                sys.exit(1)
        except Exception as e:
            print(f"❌ 自动化索引流构建失败: {e}")
            sys.exit(1)

    print("=" * 60)
    print(f"🚀 RAG 问答控制台 | LangGraph 原生大一统流范式 | 隔离域: {TARGET_DOC_ID}")
    print("   - 输入 'exit' 或 'quit' 优雅退出问答系统")
    print("   - 输入 '/local' 快捷切换为 本地 Ollama 调试模式")
    print("   - 输入 '/cloud' 快捷切换为 云端 API 高性能生产模式")
    print("=" * 60)

    is_local = True
    
    # CLI 命令行终极稳固交互大循环
    while True:
        try:
            mode_str = "本地Ollama" if is_local else "云端API"
            user_input = input(f"[{mode_str}] 请输入您的问题 >>> ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                print("👋 接收到安全退出指令，控制台正在释怀资源，再见。")
                break
                
            if user_input.lower() == "/local":
                is_local = True
                print("🔄 控制面配置已成功切换到：本地 Ollama 模式。")
                continue

            elif user_input.lower() == "/cloud":
                is_local = False
                print("🔄 控制面配置已成功切换到：云端 API 模式。")
                continue

            # 安全平稳触发 RAG 工作流
            ask(doc_id=TARGET_DOC_ID, query = user_input, is_local = is_local)
            
        except KeyboardInterrupt:
            print("\n👋 检测到系统中断信号（Ctrl+C），安全关闭。")
            break
        except Exception as e:
            print(f"⚠️ [控制台内部异常捕获]: {e}")


if __name__ == "__main__":
    main()