import atexit
import os
import re
import shutil
import signal
import sys
from uuid import uuid4

try:
    import readline
except ImportError:
    readline = None

from cogdoc.agents.conversation_memory import extract_final_answer
from cogdoc.agents.router import FORCED_TASK_TYPES
from cogdoc.api.ingest import KBExistsError, KnowledgeBaseRegistry
from cogdoc.api.persistence import SqliteSessionStore
from cogdoc.config.settings import get_settings
from cogdoc.graph.subgraphs.qa import RetrieverFactory
from cogdoc.graph.workflow import UNKNOWN_RESPONSE
from cogdoc.observability.logger import configure_logging
from cogdoc.service.chat_service import run_chat
from cogdoc.service.ingest_service import (
    KBCleanupError,
    build_kb_index_transactional,
    cancel_all_timers,
    delete_kb_index_transactional,
    drain_purge_queue,
    mark_kb_deleted,
)
from cogdoc.service.kb_locks import kb_write_lock
from cogdoc.service.kb_state import KBState
from cogdoc.service.mutation_journal import shared_mutation_journal
from cogdoc.service.process_lock import (
    acquire_single_instance_lock,
    locking_supported,
    release_single_instance_lock,
    strict_single_process,
)
from cogdoc.tools.embedder import Embedder
from cogdoc.tools.manifest import load_index_manifest
from cogdoc.tools.reranker import BGEReranker
from cogdoc.tools.rust_core_loader import ensure_rust_core

FORCED_MODE_PATTERN = re.compile(
    rf"^/({'|'.join(re.escape(task) for task in FORCED_TASK_TYPES)})(?:\s+(.*))?$",
    re.I,
)

# Tab 补全的命令与 /kb 子命令候选。
COMPLETION_COMMANDS = [
    "/kb",
    "/inbox",
    "/add",
    "/docs",
    "/rm",
    "/new",
    "/chats",
    "/open",
    "/rmchat",
    "/local",
    "/cloud",
    "/config",
    "/qa",
    "/summary",
    "/compare",
    "/help",
    "exit",
    "quit",
]
KB_SUBCOMMANDS = ["new", "use", "rm", "list"]
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


# 解析 parse forced mode 相关逻辑。
def parse_forced_mode(user_input: str) -> tuple[str | None, str]:
    match = FORCED_MODE_PATTERN.match(user_input.strip())
    if not match:
        return None, user_input
    return match.group(1).lower(), (match.group(2) or "").strip()


# 处理 safe print on interrupt 相关逻辑。
def safe_print_on_interrupt(message: str) -> None:
    # 打印退出提示时临时忽略 SIGINT，避免 Ctrl+C 连按打断清理路径。
    previous_handler = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print(message)
    finally:
        signal.signal(signal.SIGINT, previous_handler)


# 脱敏 mask key 相关逻辑。
def _mask_key(key: str) -> str:
    # 提示当前 key 时只露头尾，避免整串明文回显到终端。
    return f"{key[:3]}***{key[-4:]}" if len(key) > 7 else "***"


# 预热 warm kb 相关逻辑。
def _warm_kb(kb_id: str) -> None:
    # 切库时预热该库 bm25 分词资源；索引尚未落盘时静默跳过，留待提问时按需加载。
    try:
        engine = RetrieverFactory.get_engine(kb_id)
        if engine.bm25_retriever.exists():
            engine.bm25_retriever.warm_up()
    except Exception:
        pass


# 列出 kb documents 相关逻辑。
def _kb_documents(kb_id: str) -> list[dict]:
    # generation state 是事务提交指针且内含 documents；manifest 是提交后派生缓存，写失败时可能滞后。
    active = KBState(kb_id).active()
    if active is not None:
        return active.get("documents", [])
    return load_index_manifest(kb_id).get("documents", [])


HELP_TEXT = """\
可用命令（全部以 / 开头）：
  知识库
    /kb                    列出全部知识库
    /kb new <名称>         新建知识库并切入
    /kb use <名称>         切换当前知识库
    /kb rm  <名称>         删除知识库（需确认）
  文档（针对当前知识库）
    /inbox                 列出 your_documents 收件箱里的 PDF
    /add <文件名.pdf>      把收件箱里的 PDF 加入当前库并重建索引
    /add                   把收件箱里所有尚未入库的 PDF 一次性加入
    /docs                  列出当前库内文档
    /rm  <文件名.pdf>      从当前库移除文档并重建索引
  对话（针对当前知识库，历史持久化）
    /new                   开启一个新对话
    /chats                 列出当前库的历史对话
    /open <对话ID>         打开/恢复历史对话（支持 ID 前缀）
    /rmchat <对话ID>       删除一个对话（需确认）
  模式与强制意图
    /local  /cloud         切换本地 Ollama / 云端 API（云端缺 key 会提示配置）
    /config                配置云端 Base URL / 模型 / API Key（写入 .env）
    /qa <问题>             强制问答
    /summary <文件名>      强制总结指定文档
    /compare <A> <B> ...   强制对比多篇文档（≥2，本地模式限 2）
  其他
    /help                  显示本帮助
    exit / quit            退出
直接输入文本 = 在当前对话里向当前知识库提问。\
"""


# 封装 Console 的状态与行为。
class Console:
    # 对话历史落 SqliteSessionStore，重启不丢。
    def __init__(self):
        settings = get_settings()
        self.registry = KnowledgeBaseRegistry()
        self.sessions = SqliteSessionStore(settings.state_db_path)
        # 用绝对路径，提示与列表里一眼看清 PDF 该放哪。
        self.inbox_dir = os.path.abspath(settings.cogdoc_doc_dir)
        self.active_kb: str | None = None
        self.active_session_id: str | None = None
        self.is_local = True
        self._completion_matches: list[str] = []
        os.makedirs(self.inbox_dir, exist_ok=True)

    # ---- 工具 ----

    # 处理 confirm 相关逻辑。
    def _confirm(self, prompt: str) -> bool:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")

    # 处理 require kb 相关逻辑。
    def _require_kb(self) -> bool:
        if self.active_kb is None:
            print("⚠️ 还没有选择知识库。用 /kb 查看、/kb new <名> 创建、/kb use <名> 切换。")
            return False
        return True

    # 列出 inbox pdfs 相关逻辑。
    def _inbox_pdfs(self) -> list[str]:
        os.makedirs(self.inbox_dir, exist_ok=True)
        return sorted(
            f
            for f in os.listdir(self.inbox_dir)
            if f.lower().endswith(".pdf")
            and os.path.isfile(os.path.join(self.inbox_dir, f))
        )

    # 解析 resolve session 相关逻辑。
    def _resolve_session(self, prefix: str) -> str | None:
        # 对话 ID 是 32 位 hex，太长，按前缀唯一匹配。
        if not prefix:
            print("用法: 需要提供对话 ID（可用前缀）。")
            return None
        matches = [
            s["session_id"]
            for s in self.sessions.list_sessions(self.active_kb)
            if s["session_id"].startswith(prefix)
        ]
        if not matches:
            print(f"⚠️ 找不到匹配的对话: {prefix}")
            return None
        if len(matches) > 1:
            print(f"⚠️ ID 前缀不唯一，匹配到 {len(matches)} 个，请补全后重试。")
            return None
        return matches[0]

    # 重建 rebuild 相关逻辑。
    def _rebuild(self) -> None:
        # 重建后索引已变，需重新预热新 bm25。
        kb = self.active_kb
        try:
            result = build_kb_index_transactional(kb, self.registry.source_dir(kb))
        except Exception as e:
            print(f"❌ 索引重建失败: {e}")
            return
        if result.document_count == 0:
            print("⚠️ 当前知识库已无 PDF，索引已清空。")
        else:
            for d in result.documents:
                print(f"  -> {d.name}: {d.chunk_count} 个 Chunk")
            print(f"✅ 重建完成，共 {result.chunk_count} 个知识片段。")
        _warm_kb(kb)

    # ---- 知识库命令 ----

    # 切换 use kb 相关逻辑。
    def _use_kb(self, name: str) -> None:
        self.active_kb = name
        self.active_session_id = None
        _warm_kb(name)
        print(f"📚 已切换到知识库: {name}（/new 开始新对话，/chats 查看历史）")

    # 删除 delete kb 相关逻辑。
    def _delete_kb(self, kb_id: str) -> None:
        # 写锁内先事务清理索引并落 tombstone，再撤 registry，避免半删除态。
        with kb_write_lock(kb_id):
            delete_kb_index_transactional(kb_id)
            mark_kb_deleted(kb_id)
            self.registry.delete(kb_id)
            # 连带清掉该库的会话历史，否则同名新库复用 doc_id 会捡到旧对话。
            self.sessions.clear_kb(kb_id)

    # 处理 cmd kb 相关逻辑。
    def cmd_kb(self, sub: str, name: str) -> None:
        if sub in ("", "list"):
            records = self.registry.list()
            if not records:
                print("（暂无知识库。用 /kb new <名称> 创建一个。）")
                return
            print("📚 知识库列表:")
            for r in records:
                kb = r["kb_id"]
                marker = "→" if kb == self.active_kb else " "
                print(f" {marker} {kb}  ({len(_kb_documents(kb))} 个文档)")
            return
        if sub == "new":
            if not name:
                print("用法: /kb new <名称>")
                return
            try:
                self.registry.create(name)
            except KBExistsError:
                print(f"⚠️ 知识库已存在: {name}")
                return
            print(f"✅ 已创建知识库: {name}")
            self._use_kb(name)
            return
        if sub == "use":
            if not name:
                print("用法: /kb use <名称>")
                return
            if not self.registry.exists(name):
                print(f"⚠️ 知识库不存在: {name}")
                return
            self._use_kb(name)
            return
        if sub == "rm":
            if not name:
                print("用法: /kb rm <名称>")
                return
            if not self.registry.exists(name):
                print(f"⚠️ 知识库不存在: {name}")
                return
            if not self._confirm(
                f"确认删除知识库 【{name}】 及其全部文档与索引？此操作不可恢复"
            ):
                print("已取消。")
                return
            try:
                self._delete_kb(name)
            except KBCleanupError:
                print(f"❌ 知识库清理未完成，请重试: {name}")
                return
            print(f"🗑️ 已删除知识库: {name}")
            if self.active_kb == name:
                self.active_kb = None
                self.active_session_id = None
            return
        print(f"❓ 未知 /kb 子命令: {sub}。可用: new / use / rm（或不带参数列出）。")

    # ---- 文档命令 ----

    # 处理 cmd inbox 相关逻辑。
    def cmd_inbox(self) -> None:
        pdfs = self._inbox_pdfs()
        if not pdfs:
            print(f"（收件箱 {self.inbox_dir} 里没有 PDF。把 PDF 放进去再 /add。）")
            return
        in_kb = (
            {d.get("name") for d in _kb_documents(self.active_kb)}
            if self.active_kb
            else set()
        )
        print(f"📥 收件箱 {self.inbox_dir}:")
        for f in pdfs:
            tag = " （已在当前库）" if f in in_kb else ""
            print(f"   • {f}{tag}")

    # 处理 cmd add 相关逻辑。
    def cmd_add(self, arg: str) -> None:
        if not self._require_kb():
            return
        pdfs = self._inbox_pdfs()
        if not pdfs:
            print(f"（收件箱 {self.inbox_dir} 里没有 PDF。）")
            return
        if arg:
            name = os.path.basename(arg)
            if name not in pdfs:
                print(f"⚠️ 收件箱里找不到该 PDF: {name}（用 /inbox 查看）")
                return
            targets = [name]
        else:
            existing = {d.get("name") for d in _kb_documents(self.active_kb)}
            targets = [f for f in pdfs if f not in existing]
            if not targets:
                print("收件箱里没有需要新增的 PDF。")
                return
        dst_dir = self.registry.source_dir(self.active_kb)
        os.makedirs(dst_dir, exist_ok=True)
        for f in targets:
            shutil.copy2(os.path.join(self.inbox_dir, f), os.path.join(dst_dir, f))
        print(f"📎 已复制 {len(targets)} 个 PDF 进知识库源目录，开始同步重建索引...")
        self._rebuild()

    # 处理 cmd docs 相关逻辑。
    def cmd_docs(self) -> None:
        if not self._require_kb():
            return
        docs = _kb_documents(self.active_kb)
        if not docs:
            print("（当前知识库还没有文档。用 /add 加入。）")
            return
        print(f"📄 知识库 【{self.active_kb}】 文档:")
        for d in docs:
            print(f"   • {d.get('name')}")

    # 处理 cmd rm 相关逻辑。
    def cmd_rm(self, arg: str) -> None:
        if not self._require_kb():
            return
        if not arg:
            print("用法: /rm <文件名.pdf>")
            return
        name = os.path.basename(arg)
        path = os.path.join(self.registry.source_dir(self.active_kb), name)
        if not os.path.exists(path):
            print(f"⚠️ 当前库里找不到该文档: {name}")
            return
        os.remove(path)
        print(f"🗑️ 已移除文档 {name}，开始同步重建索引...")
        self._rebuild()

    # ---- 对话命令 ----

    # 处理 cmd new 相关逻辑。
    def cmd_new(self) -> None:
        if not self._require_kb():
            return
        self.active_session_id = uuid4().hex
        print(f"🆕 已开启新对话（{self.active_session_id[:8]}）。")

    # 处理 cmd chats 相关逻辑。
    def cmd_chats(self) -> None:
        if not self._require_kb():
            return
        sessions = self.sessions.list_sessions(self.active_kb)
        if not sessions:
            print("（当前知识库还没有历史对话。用 /new 开始。）")
            return
        print(f"💬 知识库 【{self.active_kb}】 历史对话:")
        for s in sessions:
            marker = "→" if s["session_id"] == self.active_session_id else " "
            print(
                f" {marker} {s['session_id'][:8]}  {s['title']}  （{s['message_count']} 条）"
            )
        print("（用 /open <ID前缀> 打开，/rmchat <ID前缀> 删除）")

    # 处理 cmd open 相关逻辑。
    def cmd_open(self, arg: str) -> None:
        if not self._require_kb():
            return
        sid = self._resolve_session(arg)
        if sid is None:
            return
        self.active_session_id = sid
        messages = self.sessions.get_display(self.active_kb, sid)
        print(f"📖 已打开对话 {sid[:8]}（{len(messages)} 条消息）:")
        for m in messages:
            role = "你" if m.get("role") == "user" else "AI"
            print(f"  [{role}] {m.get('content', '')}")
        print("-" * 50)

    # 处理 cmd rmchat 相关逻辑。
    def cmd_rmchat(self, arg: str) -> None:
        if not self._require_kb():
            return
        sid = self._resolve_session(arg)
        if sid is None:
            return
        if not self._confirm(f"确认删除对话 {sid[:8]}？"):
            print("已取消。")
            return
        self.sessions.clear(self.active_kb, sid)
        if self.active_session_id == sid:
            self.active_session_id = None
        print(f"🗑️ 已删除对话 {sid[:8]}。")

    # ---- 云端配置 ----

    # 配置 configure cloud 相关逻辑。
    def _configure_cloud(self, first_time: bool) -> bool:
        # 写入 .env 并即时生效；返回云端是否可用（有 key）。
        from cogdoc.config.llm_config import apply_llm_config

        settings = get_settings()
        if first_time:
            print("⚠️ 还没有配置云端 API Key，无法使用云端模式。现在配置（回车保留当前值）：")
        else:
            print("✏️ 修改云端模型配置（回车保留当前值）：")
        base = input(f"  云端 Base URL [{settings.llm_base_url}]: ").strip()
        model = input(f"  云端模型名 [{settings.llm_model_name}]: ").strip()
        cur_key = settings.llm_api_key
        key_hint = _mask_key(cur_key) if cur_key else "未设置"
        key = input(f"  云端 API Key [{key_hint}]: ").strip()
        final_key = key or cur_key
        if not final_key:
            print("❌ 未提供 API Key，云端模式不可用。")
            return False
        apply_llm_config(
            api_key=final_key,
            base_url=base or settings.llm_base_url,
            model=model or settings.llm_model_name,
        )
        print("✅ 云端配置已写入 .env 并即时生效。")
        return True

    # ---- 问答 ----

    # 输出 print answer 相关逻辑。
    def _print_answer(self, task_type: str, output: dict) -> None:
        if task_type == "qa":
            if "critique" not in output:
                print("\n⚠️ 未返回引证校验状态，已拒绝输出未确认答案。")
            elif output.get("critique"):
                print("\n❌ 引证校验未通过，已达最大自愈次数，本轮答案已拦截。")
            else:
                ans = output.get("answer", "")
                print(f"\n🤖 {ans}" if ans else "\n⚠️ 模型返回了空内容。")
        elif task_type == "summary":
            ans = output.get("answer", "")
            print(f"\n🤖 {ans}" if ans else "\n⚠️ 摘要为空。")
        else:
            content = extract_final_answer(task_type, output) or UNKNOWN_RESPONSE
            print(f"\n🤖 {content}")
        print()

    # 处理 do chat 相关逻辑。
    def do_chat(self, query: str, forced_task: str | None) -> None:
        if not self._require_kb():
            return
        if self.active_session_id is None:
            self.active_session_id = uuid4().hex
            print(f"🆕 （已自动开启新对话 {self.active_session_id[:8]}）")
        kb, sid = self.active_kb, self.active_session_id
        chat_history = self.sessions.get_history(kb, sid)
        final_result = None
        try:
            for event in run_chat(
                doc_id=kb,
                query=query,
                is_local=self.is_local,
                chat_history=chat_history,
                forced_task=forced_task,
            ):
                if event.type == "error":
                    print(f"\n⚠️ 执行中断: {event.payload.get('message', '')}")
                elif event.type == "final":
                    result = event.payload["result"]
                    output = event.payload.get("output", result.raw_output)
                    self._print_answer(result.task_type, output)
                    final_result = result
        except Exception as e:
            print(f"⚠️ 问答执行异常: {e}")
            return
        if final_result is not None:
            self.sessions.record(
                kb,
                sid,
                final_result.chat_messages,
                [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": final_result.answer},
                ],
            )

    # ---- Tab 补全 ----

    # 列出 doc names 相关逻辑。
    def _doc_names(self) -> list[str]:
        if self.active_kb is None:
            return []
        return [d.get("name", "") for d in _kb_documents(self.active_kb)]

    # 列出 session prefixes 相关逻辑。
    def _session_prefixes(self) -> list[str]:
        if self.active_kb is None:
            return []
        return [s["session_id"][:8] for s in self.sessions.list_sessions(self.active_kb)]

    # 计算 completion candidates 相关逻辑。
    def _completion_candidates(self, tokens: list[str]) -> list[str]:
        # tokens 是光标前已完成的词；为空说明正在补第一个词，给出全部命令。
        if not tokens:
            return COMPLETION_COMMANDS
        cmd = tokens[0].lower()
        if cmd == "/kb":
            if len(tokens) == 1:
                return KB_SUBCOMMANDS
            if len(tokens) == 2 and tokens[1].lower() in ("use", "rm"):
                return [r["kb_id"] for r in self.registry.list()]
            return []
        if cmd == "/add":
            return self._inbox_pdfs() if len(tokens) == 1 else []
        if cmd in ("/rm", "/summary"):
            return self._doc_names() if len(tokens) == 1 else []
        if cmd == "/compare":
            return self._doc_names()
        if cmd in ("/open", "/rmchat"):
            return self._session_prefixes() if len(tokens) == 1 else []
        return []

    # 处理 complete 相关逻辑。
    def complete(self, text: str, state: int) -> str | None:
        # readline 对同一补全会按 state 递增回调，state==0 时重算候选并缓存。
        try:
            if state == 0:
                tokens = readline.get_line_buffer()[: readline.get_begidx()].split()
                self._completion_matches = [
                    c
                    for c in self._completion_candidates(tokens)
                    if c and c.startswith(text)
                ]
            return (
                self._completion_matches[state]
                if state < len(self._completion_matches)
                else None
            )
        except Exception:
            return None

    # ---- 分发 ----

    # 处理 dispatch 相关逻辑。
    def dispatch(self, raw: str) -> bool:
        # 返回 False 表示退出控制台。
        text = raw.strip()
        if not text:
            return True
        low = text.lower()
        if low in ("exit", "quit", "/exit", "/quit"):
            return False
        if low == "/help":
            print(HELP_TEXT)
            return True
        if low == "/local":
            self.is_local = True
            print("🔄 已切换到：本地 Ollama 模式。")
            return True
        if low == "/cloud":
            if not get_settings().llm_api_key and not self._configure_cloud(
                first_time=True
            ):
                return True
            self.is_local = False
            print("🔄 已切换到：云端 API 模式。")
            return True
        if low == "/config":
            self._configure_cloud(first_time=False)
            return True
        if low == "/inbox":
            self.cmd_inbox()
            return True
        if low == "/docs":
            self.cmd_docs()
            return True
        if low == "/new":
            self.cmd_new()
            return True
        if low == "/chats":
            self.cmd_chats()
            return True

        if low.startswith("/kb"):
            toks = text.split(maxsplit=2)
            sub = toks[1].lower() if len(toks) > 1 else ""
            name = toks[2].strip() if len(toks) > 2 else ""
            self.cmd_kb(sub, name)
            return True

        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        if cmd == "/add":
            self.cmd_add(arg)
            return True
        if cmd == "/rm":
            self.cmd_rm(arg)
            return True
        if cmd == "/open":
            self.cmd_open(arg)
            return True
        if cmd == "/rmchat":
            self.cmd_rmchat(arg)
            return True

        forced_task, cleaned_query = parse_forced_mode(text)
        if forced_task:
            if not cleaned_query:
                print(f"⚠️ 请输入 /{forced_task} 后面的具体问题或文档指令。")
                return True
            self.do_chat(cleaned_query, forced_task)
            return True

        if text.startswith("/"):
            print(f"❓ 未知命令: {cmd}。输入 /help 查看可用命令。")
            return True

        self.do_chat(text, None)
        return True


# 配置 setup completion 相关逻辑。
def _setup_completion(console: "Console") -> None:
    # 无 readline（如 Windows 原生）则静默跳过，不影响主流程。
    if readline is None:
        return
    # 仅以空白为分隔符，使补全词保留前导 / 与中文文件名（默认分隔符会切碎它们）。
    readline.set_completer_delims(" \t\n")
    readline.set_completer(console.complete)
    # macOS 自带的是 libedit，绑定语法与 GNU readline 不同。
    if "libedit" in (getattr(readline, "__doc__", "") or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")


# 处理 main 相关逻辑。
# 启动横幅：纯静态 ASCII（ansi_shadow 字体），不引入运行时依赖。
BANNER = r"""
 ██████╗ ██████╗  ██████╗ ██████╗  ██████╗  ██████╗
██╔════╝██╔═══██╗██╔════╝ ██╔══██╗██╔═══██╗██╔════╝
██║     ██║   ██║██║  ███╗██║  ██║██║   ██║██║
██║     ██║   ██║██║   ██║██║  ██║██║   ██║██║
╚██████╗╚██████╔╝╚██████╔╝██████╔╝╚██████╔╝╚██████╗
 ╚═════╝ ╚═════╝  ╚═════╝ ╚═════╝  ╚═════╝  ╚═════╝
"""


def main():
    configure_logging()

    # CLI 与 API 共用进程锁：CLI 会构建索引（写），不得与运行中的实例并发写同一数据目录。
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

    # 必须在任何构建前回放 journal，使源目录与 active 代一致。
    shared_mutation_journal().recover_all()
    drain_purge_queue()

    try:
        get_rust_core()
    except RuntimeError as exc:
        print(f"❌ {exc}")
        _release_runtime_lock(lock_fh)
        sys.exit(1)

    # 全局检索/重排模型是单例，启动时预热一次；per-KB bm25 在切库时按需预热。
    try:
        print("🧠 正在预热检索与重排模型，请稍候...")
        Embedder.get_model()
        BGEReranker.warm_up()
        print("✅ 模型资源预热完成。")
    except Exception as e:
        print(f"⚠️ 预热阶段失败，稍后提问时仍会尝试按需加载: {e}")

    console = Console()
    _setup_completion(console)

    print(BANNER)
    print("=" * 60)
    print("🚀 CogDoc 控制台 | 多知识库 + 多对话 | 输入 /help 查看命令")
    print(f"📥 收件箱目录: {console.inbox_dir}（把 PDF 放进来，再 /add 入库）")
    records = console.registry.list()
    if not records:
        print("ℹ️ 当前还没有知识库。用 /kb new <名称> 创建你的第一个知识库。")
    elif len(records) == 1:
        # 仅一个库时自动切入。
        console._use_kb(records[0]["kb_id"])
    else:
        print("ℹ️ 已有知识库，用 /kb 查看、/kb use <名称> 切入。")
    print("=" * 60)

    while True:
        try:
            scope = console.active_kb or "未选库"
            chat = (
                console.active_session_id[:8] if console.active_session_id else "无对话"
            )
            mode = "本地" if console.is_local else "云端"
            user_input = input(f"[{scope}|{chat}|{mode}] >>> ")
            if not console.dispatch(user_input):
                print("👋 控制台正在释放资源，再见。")
                break
        except KeyboardInterrupt:
            safe_print_on_interrupt("\n👋 检测到系统中断信号（Ctrl+C），安全关闭。")
            break
        except EOFError:
            print("\n👋 输入流结束，安全关闭。")
            break
        except Exception as e:
            print(f"⚠️ [控制台内部异常捕获]: {e}")

    _release_runtime_lock(lock_fh)


if __name__ == "__main__":
    main()
