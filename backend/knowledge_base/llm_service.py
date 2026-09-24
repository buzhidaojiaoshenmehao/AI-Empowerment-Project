"""大模型服务 —— 对接 OpenAI / 兼容 API（手工 SSE 流式解析）"""
import base64
import json
from typing import AsyncGenerator, Optional

from backend.config import settings
from backend.knowledge_base.retriever import retriever


class LLMService:
    """大模型对话服务（支持流式输出）"""

    def __init__(self):
        self._http_client = None

    def reinitialize(self):
        """重新初始化 HTTP 客户端（API Key 变更后调用）"""
        self._http_client = None
        return True

    @staticmethod
    def normalize_base_url(base_url: str = None) -> str:
        """规范化 OpenAI 兼容接口地址，允许用户粘贴完整 chat/completions 地址。"""
        base = (base_url or "").strip() or "https://api.openai.com/v1"
        if "://" not in base:
            base = f"https://{base}"
        base = base.rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")].rstrip("/")
        return base

    def _chat_completions_url(self, base_url: str = None) -> str:
        return f"{self.normalize_base_url(base_url)}/chat/completions"

    @property
    def http_client(self):
        if self._http_client is None:
            import httpx
            kwargs = dict(
                timeout=httpx.Timeout(60.0, connect=10.0),
                follow_redirects=True,
            )
            if settings.HTTP_PROXY_ENABLED and settings.HTTP_PROXY_URL:
                kwargs["proxy"] = settings.HTTP_PROXY_URL
            if not settings.HTTP_VERIFY_SSL:
                kwargs["verify"] = False
            self._http_client = httpx.AsyncClient(**kwargs)
        return self._http_client

    async def _sse_stream(
        self,
        messages: list,
        model: str = None,
    ) -> AsyncGenerator[str, None]:
        """手工解析 SSE 流式响应，逐 chunk 产出文本 token"""
        if settings.OPENAI_API_KEY:
            api_key = settings.OPENAI_API_KEY
            base_url = settings.OPENAI_BASE_URL or "https://api.openai.com/v1"
        else:
            api_key = "ollama"
            base_url = "http://localhost:11434/v1"

        model = model or settings.LLM_MODEL
        if not model:
            raise RuntimeError("模型名称为空，请在系统设置中填写模型名称")

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": 0.3,
            "max_tokens": 2048,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with self.http_client.stream(
            "POST",
            self._chat_completions_url(base_url),
            json=payload,
            headers=headers,
        ) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                detail = error_body.decode("utf-8", errors="replace")[:200]
                raise RuntimeError(
                    f"API 返回 {resp.status_code}: {detail}"
                )

            # 逐行读取 SSE 事件流
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                if line == "data: [DONE]":
                    return
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue

    async def _non_stream_call(
        self,
        messages: list,
        model: str = None,
        max_tokens: int = 2048,
    ) -> str:
        """非流式调用（兜底方案）"""
        if settings.OPENAI_API_KEY:
            api_key = settings.OPENAI_API_KEY
            base_url = settings.OPENAI_BASE_URL or "https://api.openai.com/v1"
        else:
            api_key = "ollama"
            base_url = "http://localhost:11434/v1"

        model = model or settings.LLM_MODEL
        if not model:
            raise RuntimeError("模型名称为空，请在系统设置中填写模型名称")

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        async with self.http_client.stream(
            "POST",
            self._chat_completions_url(base_url),
            json=payload,
            headers=headers,
        ) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                detail = error_body.decode("utf-8", errors="replace")[:200]
                raise RuntimeError(
                    f"API 返回 {resp.status_code}: {detail}"
                )
            body = await resp.aread()
            result = json.loads(body.decode("utf-8"))
            choices = result.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return ""

    async def test_connection(self) -> str:
        """发送最小请求验证当前模型配置是否真实可用。"""
        if not (settings.OPENAI_API_KEY or "").strip():
            raise RuntimeError("未配置 API Key，请先填写并保存")
        reply = await self._non_stream_call(
            [{"role": "user", "content": "请只回复 OK，用于连接测试。"}],
            max_tokens=16,
        )
        return reply.strip() or "OK"

    async def extract_image_knowledge(self, image_bytes: bytes, content_type: str) -> str:
        """使用当前 OpenAI 兼容多模态模型提取截图中的项目知识。"""
        if not image_bytes:
            raise ValueError("图片内容为空")
        model = str(settings.LLM_MODEL or "").lower()
        base_url = str(settings.OPENAI_BASE_URL or "").lower()
        if "deepseek" in model or "deepseek" in base_url:
            raise RuntimeError(f"当前模型 {settings.LLM_MODEL or 'DeepSeek'} 不支持图片输入，请使用飞书 OCR 或配置多模态模型")
        mime_type = str(content_type or "image/png").split(";", 1)[0].strip()
        data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        messages = [
            {
                "role": "system",
                "content": (
                    "你是项目知识提取助手。准确提取图片中的可见文字、流程、表格要点、界面状态和结论。"
                    "不得补写图片中不存在的信息；无法辨认的部分明确标记为无法辨认。使用简洁中文 Markdown 输出。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请先列出识别文字，再总结这张图片包含的项目知识。若只是装饰图片，请如实说明。",
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        return await self._non_stream_call(messages, max_tokens=1400)

    def _build_prompt(self, query: str, context: str, history: list = None) -> list:
        """组装 RAG 提示词"""
        system_prompt = (
            "你是一个专业的 AI 知识库助手。请基于以下提供的上下文信息回答用户的问题。\n"
            "规则：\n"
            "1. 如果上下文信息足够，请给出详细、准确的回答\n"
            "2. 如果上下文中没有足够信息，请如实告知，不要编造\n"
            "3. 引用来源时标注对应的编号，如 [1]、[2]\n"
            "4. 使用中文回答"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # 添加上文历史
        if history:
            for msg in history[-6:]:  # 最多保留 6 轮对话历史
                messages.append(msg)

        # 注入检索到的上下文
        if context:
            user_content = f"上下文信息：\n{context}\n\n---\n\n用户问题：{query}"
        else:
            user_content = query

        messages.append({"role": "user", "content": user_content})
        return messages

    async def chat(
        self,
        query: str,
        history: list = None,
        use_rag: bool = True,
    ) -> AsyncGenerator[str, None]:
        """流式对话，每次 RAG 增强检索"""
        context = ""
        if use_rag:
            context = retriever.retrieve_with_context(query)

        messages = self._build_prompt(query, context, history)

        # 优先尝试 SSE 流式，失败则回退到非流式
        try:
            async for token in self._sse_stream(messages):
                yield token
        except Exception as sse_err:
            # 流式失败，回退到非流式
            try:
                full_text = await self._non_stream_call(messages)
                if full_text:
                    yield full_text
            except Exception as ns_err:
                yield f"\n\n❌ 调用大模型失败 (流式: {sse_err}, 非流式: {ns_err})。请检查 API Key 是否有效。"

    async def chat_with_sources(
        self,
        query: str,
        history: list = None,
    ) -> AsyncGenerator[str, None]:
        """流式对话 + 附带来源信息"""
        context = retriever.retrieve_with_context(query)
        sources = retriever.retrieve(query)

        # 先发送 sources 元数据（JSON 行）
        source_list = [
            {
                "content": doc.page_content[:200] + "...",
                "source": doc.metadata.get("source_file", "未知"),
            }
            for doc in sources
        ]
        yield f"__SOURCES__:{json.dumps(source_list, ensure_ascii=False)}\n"

        # 再流式输出回答
        messages = self._build_prompt(query, context, history)

        try:
            async for token in self._sse_stream(messages):
                yield token
        except Exception as sse_err:
            try:
                full_text = await self._non_stream_call(messages)
                if full_text:
                    yield full_text
            except Exception as ns_err:
                yield f"\n\n❌ 调用大模型失败 (流式: {sse_err}, 非流式: {ns_err})。请检查 API Key 是否有效。"


# 全局单例
llm_service = LLMService()
