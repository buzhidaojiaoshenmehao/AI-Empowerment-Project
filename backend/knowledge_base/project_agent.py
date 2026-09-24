"""Read-only project agent layered on top of the existing knowledge services.

The implementation uses the LangChain 1.x ``create_agent`` API. Agent failures
are handled by the API layer, which falls back to the existing RAG flow.
"""
import asyncio
import json
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

import httpx
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from backend.config import settings
from backend.knowledge_base.knowledge_graph import knowledge_graph
from backend.knowledge_base.project_context import project_context
from backend.knowledge_base.retriever import retriever
from backend.knowledge_base.vector_store import vector_store


AGENT_SYSTEM_PROMPT = """你是“项目知识 Agent”，负责基于企业项目资产回答问题。
你只能执行提供的只读工具，不得声称已经上传、删除、修改、推送或审批任何数据。
涉及项目事实、文档、知识关系、新人学习或离职交接时，必须先调用适合的工具核实。
回答应先给结论，再列依据；引用资料时使用工具返回的真实文件名。
如果工具没有找到足够证据，请明确说明知识缺口，不得编造。
不要输出私有思维链，只输出可核验的工具结论和最终回答。"""


class SearchKnowledgeInput(BaseModel):
    query: str = Field(description="需要在项目知识库中检索的问题或关键词")
    k: int = Field(default=4, ge=1, le=6, description="返回的知识片段数量")


class QueryInput(BaseModel):
    query: str = Field(description="需要查询或追溯的问题")


class RoleInput(BaseModel):
    role: str = Field(default="开发工程师", description="新人岗位，例如开发工程师、测试工程师或项目经理")


def _json_result(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts: List[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
            parts.append(str(block.get("text") or ""))
    return "".join(parts)


class ProjectAgentTraceCallback(AsyncCallbackHandler):
    """Translate real tool lifecycle callbacks into frontend-safe events."""

    TOOL_TITLES = {
        "search_project_knowledge": "检索项目知识",
        "trace_knowledge_graph": "追溯知识关系",
        "get_onboarding_guide": "核对岗位知识覆盖",
        "get_handover_status": "检查交接状态",
        "get_project_overview": "读取项目概览",
    }

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.tool_runs: Dict[str, str] = {}
        self.sources: Dict[str, Dict[str, str]] = {}
        self.query_evidence_checked = False
        self.query_evidence_found = False

    async def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        name = str((serialized or {}).get("name") or "project_tool")
        self.tool_runs[str(run_id)] = name
        title = self.TOOL_TITLES.get(name, "查询项目数据")
        await self.queue.put({
            "type": "trace",
            "payload": {
                "id": name,
                "title": title,
                "detail": f"正在核验项目数据：{title}。",
                "status": "active",
                "evidence": [],
            },
        })

    async def on_tool_end(self, output: Any, *, run_id: Any, **kwargs: Any) -> None:
        name = self.tool_runs.pop(str(run_id), "project_tool")
        payload = self._decode_output(output)
        if name == "search_project_knowledge":
            self.query_evidence_checked = True
            self.query_evidence_found = self.query_evidence_found or int(payload.get("result_count") or 0) > 0
        elif name == "trace_knowledge_graph":
            self.query_evidence_checked = True
            self.query_evidence_found = self.query_evidence_found or bool(
                int(payload.get("document_count") or 0) or int(payload.get("relation_count") or 0)
            )
        for source in payload.get("sources") or []:
            if not isinstance(source, dict):
                continue
            source_name = str(source.get("source") or "").strip()
            if source_name:
                self.sources[source_name] = {
                    "source": source_name,
                    "content": str(source.get("content") or "")[:240],
                    **{
                        key: source.get(key)
                        for key in (
                            "asset_id", "version_id", "document_id", "chunk_id", "page", "section",
                        )
                        if source.get(key) not in (None, "")
                    },
                }

        evidence = list(self.sources)[:6]
        await self.queue.put({
            "type": "trace",
            "payload": {
                "id": name,
                "title": self.TOOL_TITLES.get(name, "查询项目数据"),
                "detail": self._result_summary(name, payload),
                "status": "done",
                "evidence": evidence,
            },
        })
        if self.sources:
            await self.queue.put({"type": "sources", "sources": list(self.sources.values())})

    async def on_tool_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        name = self.tool_runs.pop(str(run_id), "project_tool")
        await self.queue.put({
            "type": "trace",
            "payload": {
                "id": name,
                "title": self.TOOL_TITLES.get(name, "查询项目数据"),
                "detail": "当前核验未完成，将由兼容检索流程继续处理。",
                "status": "error",
                "evidence": [],
            },
        })

    @staticmethod
    def _decode_output(output: Any) -> Dict[str, Any]:
        content = getattr(output, "content", output)
        if isinstance(content, dict):
            return content
        try:
            decoded = json.loads(str(content or "{}"))
            return decoded if isinstance(decoded, dict) else {}
        except (TypeError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _result_summary(name: str, payload: Dict[str, Any]) -> str:
        if name == "search_project_knowledge":
            return f"检索到 {payload.get('result_count', 0)} 个相关片段，来自 {len(payload.get('sources') or [])} 份资料。"
        if name == "trace_knowledge_graph":
            return (
                f"找到 {payload.get('document_count', 0)} 份来源文档和 "
                f"{payload.get('relation_count', 0)} 条知识关系。"
            )
        if name == "get_onboarding_guide":
            return (
                f"{payload.get('role', '当前岗位')}知识就绪度为 {payload.get('readiness', 0)}%，"
                f"发现 {payload.get('gap_count', 0)} 个待补充范围。"
            )
        if name == "get_handover_status":
            return (
                f"已检查 {payload.get('record_count', 0)} 个交接任务，"
                f"发现 {payload.get('risk_count', 0)} 项待处理风险。"
            )
        if name == "get_project_overview":
            return (
                f"项目包含 {payload.get('document_count', 0)} 份文档、"
                f"{payload.get('chunk_count', 0)} 个知识块和 {payload.get('graph_nodes', 0)} 个图谱节点。"
            )
        return "项目数据核验完成。"


class ProjectAgentService:
    """Create and run a read-only project agent with real tool traces."""

    def __init__(
        self,
        onboarding_provider: Callable[[str], Dict[str, Any]],
        handover_provider: Callable[[], List[Dict[str, Any]]],
    ):
        self.onboarding_provider = onboarding_provider
        self.handover_provider = handover_provider

    def build_tools(self) -> List[StructuredTool]:
        def search_project_knowledge(query: str, k: int = 4) -> str:
            """Search project documents and return traceable source excerpts."""
            docs = retriever.retrieve(query, k=k)
            results = [
                {
                    "source": str(doc.metadata.get("source_file") or "未知资料"),
                    "content": doc.page_content[:600],
                    "category": doc.metadata.get("category") or "",
                    "asset_id": str(doc.metadata.get("asset_id") or ""),
                    "version_id": str(doc.metadata.get("version_id") or ""),
                    "document_id": str(doc.metadata.get("document_id") or ""),
                    "chunk_id": str(doc.metadata.get("chunk_id") or ""),
                    "page": doc.metadata.get("page"),
                    "section": str(doc.metadata.get("section") or ""),
                }
                for doc in docs
            ]
            return _json_result({
                "query": query,
                "result_count": len(results),
                "results": results,
                "sources": [{**item, "content": item["content"][:240]} for item in results],
            })

        def trace_knowledge_graph(query: str) -> str:
            """Trace document evidence and relationships in the knowledge graph."""
            trace = knowledge_graph.trace_decision(query)
            documents = trace.get("source_documents") or []
            relations = trace.get("related_knowledge") or []
            return _json_result({
                "query": query,
                "document_count": len(documents),
                "relation_count": len(relations),
                "source_documents": documents[:5],
                "related_knowledge": relations[:8],
                "trace_path": trace.get("trace_path") or [],
                "sources": [
                    {"source": item.get("source") or "未知资料", "content": item.get("content") or ""}
                    for item in documents[:5]
                ],
            })

        def get_onboarding_guide(role: str = "开发工程师") -> str:
            """Get role-specific readiness, knowledge gaps and recommended documents."""
            guide = self.onboarding_provider(role)
            recommended = guide.get("recommended_documents") or []
            return _json_result({
                "role": guide.get("role") or role,
                "focus": guide.get("focus") or "",
                "readiness": guide.get("readiness") or 0,
                "gap_count": len(guide.get("gaps") or []),
                "gaps": guide.get("gaps") or [],
                "learning_path": guide.get("learning_path") or [],
                "recommended_documents": recommended,
                "sources": [
                    {"source": item.get("source_file") or "未知资料", "content": item.get("category_label") or "岗位推荐资料"}
                    for item in recommended
                ],
            })

        def get_handover_status(query: str) -> str:
            """Inspect current handover progress and unresolved risks without changing records."""
            records = self.handover_provider()
            keyword = query.strip().lower()
            matched = [
                record for record in records
                if not keyword or keyword in json.dumps(record, ensure_ascii=False).lower()
            ]
            if not matched:
                matched = records
            compact = [
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "role": item.get("role"),
                    "recipient": item.get("recipient"),
                    "status": item.get("status"),
                    "completion": item.get("completion", 0),
                    "file_count": item.get("file_count", 0),
                    "risks": item.get("risks") or [],
                }
                for item in matched[:8]
            ]
            return _json_result({
                "query": query,
                "record_count": len(compact),
                "risk_count": sum(len(item["risks"]) for item in compact),
                "records": compact,
                "sources": [],
            })

        def get_project_overview(query: str) -> str:
            """Read project knowledge inventory, categories and graph scale."""
            context = project_context.get_full_context()
            graph = knowledge_graph.get_graph_data()
            documents = context.get("documents") or []
            return _json_result({
                "query": query,
                "project_name": context.get("project_name") or settings.PROJECT_NAME,
                "document_count": context.get("document_count") or len(documents),
                "chunk_count": vector_store.count(),
                "category_counts": context.get("category_counts") or {},
                "graph_nodes": len(graph.get("nodes") or []),
                "graph_edges": len(graph.get("edges") or []),
                "documents": documents[:12],
                "sources": [
                    {"source": item.get("source_file") or item.get("name") or "未知资料", "content": "项目知识清单"}
                    for item in documents[:6]
                ],
            })

        return [
            StructuredTool.from_function(
                func=search_project_knowledge,
                name="search_project_knowledge",
                description="检索项目知识库中的文档片段，适用于项目事实、流程、需求、架构和经验问题。",
                args_schema=SearchKnowledgeInput,
            ),
            StructuredTool.from_function(
                func=trace_knowledge_graph,
                name="trace_knowledge_graph",
                description="追溯知识之间的关系、决策依据和来源路径，适用于‘有什么关系’或‘依据是什么’。",
                args_schema=QueryInput,
            ),
            StructuredTool.from_function(
                func=get_onboarding_guide,
                name="get_onboarding_guide",
                description="查询指定岗位的知识就绪度、知识缺口、学习路径和推荐资料。",
                args_schema=RoleInput,
            ),
            StructuredTool.from_function(
                func=get_handover_status,
                name="get_handover_status",
                description="只读查询离职交接任务的进度、资料数量、接替人和待处理风险。",
                args_schema=QueryInput,
            ),
            StructuredTool.from_function(
                func=get_project_overview,
                name="get_project_overview",
                description="读取项目知识资产总览，包括文档、分类、知识块和图谱规模。",
                args_schema=QueryInput,
            ),
        ]

    @staticmethod
    def _chat_history(history: Optional[List[Dict[str, Any]]]) -> List[Any]:
        messages: List[Any] = []
        for item in (history or [])[-8:]:
            role = str(item.get("role") or "")
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    @staticmethod
    def _build_model() -> Tuple[Any, Optional[httpx.Client], Optional[httpx.AsyncClient]]:
        from langchain_openai import ChatOpenAI

        if settings.OPENAI_API_KEY:
            api_key = settings.OPENAI_API_KEY
            base_url = settings.OPENAI_BASE_URL or "https://api.openai.com/v1"
        else:
            api_key = "ollama"
            base_url = "http://localhost:11434/v1"
        model_name = (settings.LLM_MODEL or "").strip()
        if not model_name:
            raise RuntimeError("模型名称为空")

        base_url = base_url.strip().rstrip("/")
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")].rstrip("/")

        sync_client: Optional[httpx.Client] = None
        async_client: Optional[httpx.AsyncClient] = None
        client_options: Dict[str, Any] = {}
        if settings.HTTP_PROXY_ENABLED and settings.HTTP_PROXY_URL:
            client_options["proxy"] = settings.HTTP_PROXY_URL
        if not settings.HTTP_VERIFY_SSL:
            client_options["verify"] = False
        if client_options:
            sync_client = httpx.Client(timeout=45.0, **client_options)
            async_client = httpx.AsyncClient(timeout=45.0, **client_options)

        model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.2,
            max_retries=1,
            timeout=45.0,
            streaming=False,
            http_client=sync_client,
            http_async_client=async_client,
        )
        return model, sync_client, async_client

    async def _invoke(self, query: str, history: List[Dict[str, Any]], callback: ProjectAgentTraceCallback) -> str:
        from langchain.agents import create_agent

        model, sync_client, async_client = self._build_model()
        tools = self.build_tools()
        chat_history = self._chat_history(history)
        try:
            agent = create_agent(
                model=model,
                tools=tools,
                system_prompt=AGENT_SYSTEM_PROMPT,
                name="project_knowledge_agent",
            )
            result = await agent.ainvoke(
                {"messages": [*chat_history, HumanMessage(content=query)]},
                config={"callbacks": [callback]},
            )
            result_messages = result.get("messages") or []
            return _message_text(result_messages[-1].content) if result_messages else ""
        finally:
            if async_client is not None:
                await async_client.aclose()
            if sync_client is not None:
                sync_client.close()

    async def stream(
        self,
        query: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        callback = ProjectAgentTraceCallback()
        task = asyncio.create_task(self._invoke(query, history or [], callback))
        while not task.done() or not callback.queue.empty():
            try:
                event = await asyncio.wait_for(callback.queue.get(), timeout=0.08)
                yield event
            except asyncio.TimeoutError:
                continue
        answer = await task
        if not answer.strip():
            raise RuntimeError("Agent 未生成有效回答")
        yield {
            "type": "answer",
            "content": answer,
            "sources": list(callback.sources.values()),
            "no_answer": bool(callback.query_evidence_checked and not callback.query_evidence_found),
        }
