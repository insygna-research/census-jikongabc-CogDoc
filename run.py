import sys  
import os 
import json 
import hashlib
from graph.workflow import app, RetrieverFactory
from tools.parser import smart_parse 
from tools.chunker import chunk_paper
from tools.embedder import Embedder
from tools.reranker import BGEReranker 

DEFAULT_DOC_DIR = os.getenv("COGDOC_DOC_DIR", "测试论文")  # 默认扫描的PDF目录，可用环境变量覆盖
MANIFEST_DIR = os.path.join("data", "manifests")  # 索引指纹清单保存目录


def _manifest_path(doc_id: str) -> str:  # 生成指定知识库的指纹清单路径
    return os.path.join(MANIFEST_DIR, f"{doc_id}.json")  # 返回manifest文件路径


def _file_sha256(path: str) -> str:  # 计算文件内容的sha256哈希
    digest = hashlib.sha256()  # 创建sha256计算器
    with open(path, "rb") as f:  # 以二进制方式读取文件
        for block in iter(lambda: f.read(1024 * 1024), b""):  # 分块读取避免大文件占满内存
            digest.update(block)  # 累加当前文件块到哈希计算器
    return digest.hexdigest()  # 返回十六进制哈希值


def _scan_pdf_manifest(doc_id: str, doc_dir: str) -> dict:  # 扫描PDF目录并生成当前指纹清单
    pdf_files = sorted(f for f in os.listdir(doc_dir) if f.lower().endswith(".pdf"))  # 获取稳定排序后的PDF文件列表
    documents = []  # 保存每个PDF的指纹信息
    for pdf in pdf_files:  # 遍历每个PDF文件
        pdf_path = os.path.join(doc_dir, pdf)  # 拼接PDF完整路径
        stat = os.stat(pdf_path)  # 读取文件元信息
        documents.append({
            "name": pdf,
            "size": stat.st_size,
            "sha256": _file_sha256(pdf_path)
        })  # 记录文件名、大小和内容哈希
    return {
        "doc_id": doc_id,
        "doc_dir": os.path.abspath(doc_dir),
        "documents": documents
    }  # 返回当前知识库源文件快照


def _load_index_manifest(doc_id: str) -> dict:  # 读取已保存的索引指纹清单
    path = _manifest_path(doc_id)  # 获取manifest路径
    if not os.path.exists(path):  # manifest不存在说明没有可验证的旧索引
        return {}  # 返回空清单触发重建
    try:
        with open(path, "r", encoding = "utf-8") as f:  # 读取manifest文件
            return json.load(f)  # 解析JSON清单
    except Exception:
        return {}  # manifest损坏时按过期处理


def _save_index_manifest(manifest: dict) -> None:  # 保存当前索引对应的源文件指纹
    os.makedirs(MANIFEST_DIR, exist_ok = True)  # 确保manifest目录存在
    with open(_manifest_path(manifest["doc_id"]), "w", encoding = "utf-8") as f:  # 打开manifest输出文件
        json.dump(manifest, f, ensure_ascii = False, indent = 2)  # 写入可读JSON


def _index_is_current(doc_id: str, doc_dir: str, engine) -> bool:  # 判断本地索引是否与PDF目录一致
    if not (engine.vector_retriever.exists() and engine.bm25_retriever.exists()):  # 任一路索引缺失都视为不可用
        return False  # 触发重建
    if not os.path.exists(doc_dir):  # 源目录不存在时旧索引不可信
        return False  # 触发重建或清理
    current_manifest = _scan_pdf_manifest(doc_id, doc_dir)  # 计算当前PDF目录指纹
    return current_manifest == _load_index_manifest(doc_id)  # 对比当前指纹和上次建库指纹


def warm_up_runtime(engine) -> None:  # 提前加载运行期重资源
    print("🧠 正在预热检索与重排模型，请稍候...")
    Embedder.get_model()  # 预加载向量模型
    engine.bm25_retriever.warm_up()  # 预热BM25分词器
    BGEReranker.warm_up()  # 预加载精排模型
    print("✅ 模型与分词资源预热完成。")


def build_index(doc_id: str, doc_dir: str = DEFAULT_DOC_DIR):  # 根据PDF目录重建双轨索引
    if not os.path.exists(doc_dir):  # 源目录不存在时创建目录并停止建库
        os.makedirs(doc_dir)  # 创建待扫描目录
        RetrieverFactory.get_engine(doc_id).clear()  # 清除旧索引避免误用历史数据
        print(f"📁 已自动为您创建待扫描的知识库目录: 【{doc_dir}】")
        print(f"📌 请将您的 PDF 论文或文档放入该目录下，然后重新拉起控制台。")
        return  # 无源文件时结束建库

    pdf_files = sorted(f for f in os.listdir(doc_dir) if f.lower().endswith(".pdf"))  # 获取稳定排序后的PDF列表
    if not pdf_files:  # 没有PDF时不能建立有效索引
        RetrieverFactory.get_engine(doc_id).clear()  # 清除旧索引避免误答
        print(f"⚠️ 提示: 目录 【{doc_dir}】 当前为空，未检测到任何 PDF 文件。")
        return  # 无源文件时结束建库

    all_chunks = []  # 汇总所有PDF的切块结果
    next_chunk_index = 0  # 为多PDF分配全局唯一chunk编号
    print("📚 检测到静态索引缺失或过期，开始构建知识库多轨索引...")

    for pdf in pdf_files:  # 逐个解析PDF
        pdf_path = os.path.join(doc_dir, pdf)  # 拼接当前PDF路径
        print(f" 正在深度解析文档: {pdf}")
        
        pages = smart_parse(pdf_path)  # 解析PDF页面文本
        chunks = chunk_paper(pages)  # 将页面文本切成语义块
        for chunk in chunks:  # 统一修正为跨文件全局chunk编号
            chunk["meta"]["chunk_index"] = next_chunk_index  # 写入全局chunk编号
            next_chunk_index += 1  # 递增下一个chunk编号
        print(f"  -> 成功抽取 {len(chunks)} 个语义 Chunk")

        print("\n--- [✂️ 离线切块细节预览开始] ---")
        for i, chunk in enumerate(chunks):  # 打印当前PDF切块预览
            meta = chunk.get("meta", {})  # 读取chunk元数据
            source = meta.get("source", pdf)  # 读取来源文件名
            page = meta.get("page", 1)  # 读取页码
            chunk_idx = meta.get("chunk_index", i)  # 读取chunk编号
            
            print(f"📄 [Chunk #{chunk_idx}] 来源: {source} | 页码: P{page}")
            print(f"📝 文本内容:\n{chunk['text'].strip()}")
            print("-" * 30)
        print("--- [✂️ 离线切块细节预览结束] ---\n")

        all_chunks.extend(chunks)  # 汇总当前PDF的全部chunk

    if not all_chunks:  # 所有PDF都没有提取出有效文本
        print("⚠️ 未从文档中提取到有效的文本片段，放弃索引构建。")
        return  # 无有效chunk时结束建库

    print(f"\n📊 全量清洗完成，共生成 {len(all_chunks)} 个标准知识片段。")
    print("正在写入双轨融合索引（Vector + BM25）...")
    
    engine = RetrieverFactory.get_engine(doc_id)  # 获取当前知识库的混合检索器
    engine.index(all_chunks)  # 写入向量索引和BM25索引
    _save_index_manifest(_scan_pdf_manifest(doc_id, doc_dir))  # 保存源文件指纹用于下次启动校验
    print("✅ 物理多轨索引构建成功并落盘，数据管线安全关闭。\n")


def ask(doc_id: str, query: str, is_local: bool = False):
    """
    100% 正统的 LangGraph 原生消息流消费客户端。
    支持在提问时在线将精排召回的切块详情喷射在终端上。
    """
    initial_state = {"messages": []}  # 初始化LangGraph消息状态
    
    runtime_config = {
        "configurable": {
            "doc_id": doc_id,
            "query": query,
            "is_local": is_local
        }  # 通过configurable传递运行时控制参数
    }  # 构造LangGraph运行配置

    try:
        print(f"\n[运行模式]: {'本地 Ollama' if is_local else '云端 API'}")
        
        token_stream = app.stream(
            initial_state, 
            config=runtime_config, 
            stream_mode=["messages", "updates"]
        )  # 同时订阅模型消息流和节点状态更新
        
        has_printed_ai_prompt = False  # 控制AI前缀只打印一次

        try:
            for mode, data in token_stream:  # 消费LangGraph多模式流
                if mode == "updates" and "rerank_node" in data:  # 捕获精排节点输出
                    final_docs = data["rerank_node"].get("docs", [])  # 读取精排后的最终文档
                    print("\n🎯 [RAG 召回与精排切块结果预览]:")
                    if not final_docs:  # 无召回结果时打印空提示
                        print("  （未检索到任何相关的参考本地知识库内容。）")
                    else:
                        for idx, doc in enumerate(final_docs):  # 逐条打印精排结果摘要
                            meta = doc.get("meta", {})  # 获取文档元数据
                            retrieval_info = doc.get("retrieval", {})  # 获取检索得分信息
                            print(f"  📍 [{idx+1}] 来源: {meta.get('source')} | 页码: P{meta.get('page')}")
                            print(f"     📊 融合得分(RRF): {retrieval_info.get('rrf_score', 'N/A')}")
                            preview_text = doc['text'].strip().replace('\n', ' ')  # 压缩预览文本为单行
                            print(f"     📄 核心内容: {preview_text[:120]}...")
                            print("     " + "-" * 40)
                    print("=" * 50)

                elif mode == "messages":  # 捕获模型消息流
                    chunk, _ = data  # 拆出消息对象和元信息
                    if hasattr(chunk, "content") and chunk.content:  # 只打印有正文的消息片段
                        if not has_printed_ai_prompt:  # 首次输出前打印AI标识
                            print("🤖 [AI]: ", end = "", flush = True)
                            has_printed_ai_prompt = True  # 标记AI前缀已打印
                        print(chunk.content, end = "", flush = True)  # 流式输出模型内容
                        
        except Exception as stream_err:
            print(f"\n⚠️ [大模型流式通信管道在运行时遭遇意外中断]: {stream_err}")
            
        print("\n" + "-" * 50)
        
    except Exception as e:
        print(f"\n❌ [Pipeline 核心图调度执行失败]: {e}")


def main():  # CLI主入口
    TARGET_DOC_ID = "arch_blueprint_2026"  # 当前知识库ID
    TARGET_DOC_DIR = DEFAULT_DOC_DIR  # 当前知识库源文档目录
    
    engine = RetrieverFactory.get_engine(TARGET_DOC_ID)  # 获取混合检索引擎

    if not _index_is_current(TARGET_DOC_ID, TARGET_DOC_DIR, engine):  # 索引缺失或源PDF变化时重建
        print(f"⚠️ 预检提示: 知识库 【{TARGET_DOC_ID}】 的本地索引缺失或已过期。")
        try:
            build_index(TARGET_DOC_ID, TARGET_DOC_DIR)  # 构建或重建本地索引
            if not (engine.vector_retriever.exists() and engine.bm25_retriever.exists()):  # 建库后仍不可用则退出
                print("❌ 错误: 知识库目录中由于缺乏源文件，无法建立有效索引。")
                input("⚙️ 请在上述目标文件夹内放入 PDF 文档后，按回车键退出并重新拉起系统...")
                sys.exit(1)  # 退出控制台
        except Exception as e:
            print(f"❌ 自动化索引流构建失败: {e}")
            sys.exit(1)  # 建库异常时退出控制台
    else:
        print(f"✅ 知识库 【{TARGET_DOC_ID}】 索引与源PDF哈希指纹完全一致，无需重建。")

    try:
        warm_up_runtime(engine)  # 进入问答前预热模型资源
    except Exception as e:
        print(f"⚠️ 预热阶段失败，稍后提问时仍会尝试按需加载: {e}")

    print("=" * 60)
    print(f"🚀 RAG 问答控制台 | LangGraph 原生大一统流范式 | 隔离域: {TARGET_DOC_ID}")
    print("   - 输入 'exit' 或 'quit' 优雅退出问答系统")
    print("   - 输入 '/local' 快捷切换为 本地 Ollama 调试模式")
    print("   - 输入 '/cloud' 快捷切换为 云端 API 高性能生产模式")
    print("=" * 60)

    is_local = True  # 默认使用本地Ollama模式
    
    while True:  # 启动命令行交互循环
        try:
            mode_str = "本地Ollama" if is_local else "云端API"  # 根据当前模式生成提示词
            user_input = input(f"[{mode_str}] 请输入您的问题 >>> ").strip()  # 读取用户输入
            
            if not user_input:  # 空输入直接继续等待
                continue
                
            if user_input.lower() in ["exit", "quit"]:  # 退出命令
                print("👋 接收到安全退出指令，控制台正在释放资源，再见。")
                break
                
            if user_input.lower() == "/local":  # 切换到本地模型
                is_local = True  # 设置为本地Ollama模式
                print("🔄 控制面配置已成功切换到：本地 Ollama 模式。")
                continue

            elif user_input.lower() == "/cloud":  # 切换到云端模型
                is_local = False  # 设置为云端API模式
                print("🔄 控制面配置已成功切换到：云端 API 模式。")
                continue

            ask(doc_id=TARGET_DOC_ID, query=user_input, is_local=is_local)  # 执行RAG问答流程
            
        except KeyboardInterrupt:
            print("\n👋 检测到系统中断信号（Ctrl+C），安全关闭。")
            break  # 用户中断时退出循环
        except Exception as e:
            print(f"⚠️ [控制台内部异常捕获]: {e}")


if __name__ == "__main__":
    main()
