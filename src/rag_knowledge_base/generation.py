"""生成：接口、确定性 Mock 实现与 OpenAI 兼容真实客户端。

Mock 不调用真实 API、不产生费用；real 模式通过 OpenAI 兼容的
/chat/completions 接口调用真实服务（已验证目标：智谱 GLM），
Prompt 强制要求模型依据检索资料回答、不编造。
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import httpx

from .config import Settings
from .errors import InvalidResponseError
from .http_transport import post_json
from .vector_store import SearchHit


@runtime_checkable
class GenerationClient(Protocol):
    """生成客户端接口：问题与上下文进，回答文本出。"""

    def generate(self, question: str, contexts: Sequence[SearchHit]) -> str:
        """基于上下文生成回答。"""
        ...


class MockGenerationClient:
    """确定性 Mock 生成：把检索命中拼接为固定格式回答。"""

    def generate(self, question: str, contexts: Sequence[SearchHit]) -> str:
        """无命中时明确说明；有命中时引用每条命中的得分与片段。"""
        if not contexts:
            return f"[mock] 未检索到相关资料，无法回答：{question}"
        lines = [f"[mock] 基于检索到的 {len(contexts)} 段资料回答：{question}"]
        for hit in contexts:
            lines.append(f"- (score={hit.score:.2f}) {hit.chunk.text[:50]}")
        return "\n".join(lines)


REAL_SYSTEM_PROMPT = (
    "你是一个严谨的中文问答助手。回答规则："
    "1. 优先且仅依据下面提供的资料回答；"
    "2. 如果资料中没有足够信息，必须明确说明资料中没有足够信息，不要猜测；"
    "3. 不要编造资料中不存在的内容。"
)


def _format_contexts(contexts: Sequence[SearchHit]) -> str:
    """把检索命中格式化为带编号与得分的资料块。"""
    lines = [
        f"[{position}] (score={hit.score:.2f}) {hit.chunk.text}"
        for position, hit in enumerate(contexts, start=1)
    ]
    return "\n".join(lines) if lines else "（无）"


class OpenAICompatibleGenerationClient:
    """通过 OpenAI 兼容的 /chat/completions 接口生成真实回答（已验证目标：智谱 GLM）。

    官方接口（2026-11 文档确认）：POST {base_url}/chat/completions，
    请求体 {"model": ..., "messages": [...]}，响应取 choices[0].message.content。
    """

    provider = "openai-compatible"

    def __init__(
        self,
        settings: Settings,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """保存配置；transport 仅供测试注入（httpx.MockTransport），生产不传。"""
        self._settings = settings
        self._transport = transport

    def generate(self, question: str, contexts: Sequence[SearchHit]) -> str:
        """把「问题 + 检索资料」交给真实模型，并强制其依据资料回答。"""
        assert self._settings.api_key is not None  # load_settings 已保证非空
        url = (self._settings.base_url or "").rstrip("/") + "/chat/completions"
        user_content = f"资料：\n{_format_contexts(contexts)}\n\n问题：{question}"
        payload = {
            "model": self._settings.generation_model,
            "messages": [
                {"role": "system", "content": REAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        }
        data = post_json(
            url,
            api_key=self._settings.api_key,
            payload=payload,
            timeout_seconds=self._settings.timeout_seconds,
            transport=self._transport,
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InvalidResponseError("响应缺少 choices[0].message.content 字段") from exc
        if not isinstance(content, str) or not content.strip():
            raise InvalidResponseError("响应内容为空")
        return content
