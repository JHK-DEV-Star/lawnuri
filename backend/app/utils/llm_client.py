"""
LLM client supporting multiple providers.

Provider-specific SDK usage:
- OpenAI / Custom: OpenAI SDK (standard Bearer auth)
- Gemini (AI Studio): Google Gen AI SDK (API key)
- Vertex AI: Google Gen AI SDK (service account OAuth)
- Anthropic: Native Anthropic SDK
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openai import AsyncOpenAI, OpenAI

from app.utils.logger import logger

# Regex to strip <think>...</think> blocks (e.g. DeepSeek reasoning)
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Regex to strip markdown code block wrappers from JSON responses
_CODE_BLOCK_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    return _THINK_TAG_RE.sub("", text).strip()


def _strip_code_block(text: str) -> str:
    text = text.strip()
    match = _CODE_BLOCK_RE.match(text)
    if match:
        return match.group(1).strip()
    return text


class _FakeFunction:
    """Shim mimicking OpenAI's Function object for Google GenAI compatibility."""

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments

class _FakeToolCall:
    """Shim mimicking OpenAI's ToolCall object for Google GenAI compatibility."""

    def __init__(self, id: str, function: _FakeFunction):
        self.id = id
        self.function = function
        self.type = "function"

class _FakeMessage:
    """Shim mimicking OpenAI's Message object for Google GenAI compatibility."""

    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls

class _FakeChoice:
    """Shim mimicking OpenAI's Choice object for Google GenAI compatibility."""

    def __init__(self, content, tool_calls, finish_reason):
        self.message = _FakeMessage(content, tool_calls)
        self.finish_reason = finish_reason

class _FakeResponse:
    """Shim mimicking OpenAI's ChatCompletion response for Google GenAI compatibility."""

    def __init__(self, content, tool_calls=None, finish_reason="stop"):
        self.choices = [_FakeChoice(content, tool_calls, finish_reason)]


class LLMClient:
    """
    Unified LLM client supporting OpenAI, Gemini, Vertex AI, Anthropic, and custom providers.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        max_retries: int = 2,
        provider: str = "",
        vertex_project_id: str = "",
        vertex_location: str = "global",
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._provider = provider
        self._is_anthropic = False
        self._is_google = False  # Gemini or Vertex AI

        if provider == "anthropic":
            from anthropic import Anthropic, AsyncAnthropic
            self._is_anthropic = True
            self._anthropic_sync = Anthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)
            self._anthropic_async = AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)

        elif provider in ("gemini", "vertex_ai"):
            from google import genai
            self._is_google = True

            if provider == "gemini":
                self._genai_client = genai.Client(api_key=api_key)
            else:
                _vx_kwargs: dict = {"vertexai": True, "api_key": api_key}
                if vertex_project_id:
                    _vx_kwargs["project"] = vertex_project_id
                if vertex_location and vertex_location != "global":
                    _vx_kwargs["location"] = vertex_location
                self._genai_client = genai.Client(**_vx_kwargs)

        else:
            self._sync_client = OpenAI(
                api_key=api_key,
                base_url=base_url or None,
                timeout=timeout,
                max_retries=max_retries,
            )
            self._async_client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url or None,
                timeout=timeout,
                max_retries=max_retries,
            )

        # Token usage tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0

        logger.info(
            "LLMClient initialized: model=%s, provider=%s",
            model, provider or "openai",
        )

    # ------------------------------------------------------------------
    # Token usage tracking
    # ------------------------------------------------------------------

    def _track_usage(self, input_tokens: int, output_tokens: int) -> None:
        """Accumulate token usage from a single LLM call."""
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._call_count += 1

    @property
    def usage_summary(self) -> dict:
        """Return cumulative token usage stats."""
        return {
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "total_tokens": self._total_input_tokens + self._total_output_tokens,
            "call_count": self._call_count,
        }

    def log_usage_summary(self, label: str = "") -> None:
        """Log accumulated token usage."""
        s = self.usage_summary
        logger.info(
            "[LLM usage%s] %d calls, %d input + %d output = %d total tokens",
            f" ({label})" if label else "",
            s["call_count"], s["input_tokens"], s["output_tokens"], s["total_tokens"],
        )

    def reset_usage(self) -> None:
        """Reset token counters (e.g., at start of a new debate)."""
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._call_count = 0

    # ------------------------------------------------------------------
    # Multimodal message preparation (PDF pass-through)
    # ------------------------------------------------------------------

    def _prepare_messages_openai(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert internal multimodal format to OpenAI content arrays."""
        import base64
        result = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                parts = []
                for p in content:
                    if p.get("type") == "text":
                        parts.append({"type": "text", "text": p["text"]})
                    elif p.get("type") == "pdf":
                        b64 = base64.b64encode(p["data"]).decode()
                        parts.append({"type": "file", "file": {
                            "filename": p.get("filename", "document.pdf"),
                            "file_data": f"data:application/pdf;base64,{b64}",
                        }})
                result.append({**m, "content": parts})
            else:
                result.append(m)
        return result

    def _prepare_messages_anthropic(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], bool]:
        """Convert internal multimodal format to Anthropic content arrays.

        Returns (converted_messages, has_pdf).
        """
        import base64
        has_pdf = False
        result = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                parts = []
                for p in content:
                    if p.get("type") == "text":
                        parts.append({"type": "text", "text": p["text"]})
                    elif p.get("type") == "pdf":
                        has_pdf = True
                        b64 = base64.b64encode(p["data"]).decode()
                        parts.append({"type": "document", "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": b64,
                        }})
                result.append({**m, "content": parts})
            else:
                result.append(m)
        return result, has_pdf

    # ------------------------------------------------------------------
    # Google Gen AI helpers
    # ------------------------------------------------------------------

    def _convert_messages_to_genai(self, messages: List[Dict[str, Any]]):
        """Convert OpenAI-format messages to Google Gen AI format."""
        from google.genai import types

        system_instruction = None
        contents = []

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")

            if role == "system":
                system_instruction = content
            elif role == "tool":
                # Tool result -> function_response part
                # Gemini requires all function_response parts in a single Content
                # when responding to parallel function calls
                fn_resp_part = types.Part(function_response=types.FunctionResponse(
                    name=m.get("name", "unknown"),
                    response={"result": content},
                ))
                if (contents and contents[-1].role == "user" and contents[-1].parts
                        and any(hasattr(p, 'function_response') and p.function_response
                                for p in contents[-1].parts)):
                    # Append to existing tool response Content
                    contents[-1].parts.append(fn_resp_part)
                else:
                    contents.append(types.Content(role="user", parts=[fn_resp_part]))
            elif role == "assistant" and m.get("tool_calls"):
                # Assistant message with tool calls -> function_call parts
                parts = []
                if content:
                    parts.append(types.Part(text=content))
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    args = json.loads(fn.get("arguments", "{}")) if isinstance(fn.get("arguments"), str) else fn.get("arguments", {})
                    parts.append(types.Part(function_call=types.FunctionCall(
                        name=fn.get("name", ""),
                        args=args,
                    )))
                contents.append(types.Content(role="model", parts=parts))
            else:
                genai_role = "user" if role == "user" else "model"
                if isinstance(content, list):
                    # Multimodal content array (text + PDF parts)
                    parts = []
                    for part in content:
                        if part.get("type") == "text":
                            parts.append(types.Part(text=part["text"]))
                        elif part.get("type") == "pdf":
                            parts.append(types.Part(inline_data=types.Blob(
                                mime_type="application/pdf",
                                data=part["data"],
                            )))
                    contents.append(types.Content(role=genai_role, parts=parts))
                else:
                    contents.append(types.Content(
                        role=genai_role,
                        parts=[types.Part(text=content or "")],
                    ))

        return contents, system_instruction

    def _genai_chat_sync(
        self, messages: List[Dict[str, Any]], temperature: float, max_tokens: Optional[int]
    ) -> str:
        from google.genai import types

        contents, system_instruction = self._convert_messages_to_genai(messages)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens or 4096,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        response = self._genai_client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        text = response.text or ""
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            self._track_usage(
                getattr(response.usage_metadata, 'prompt_token_count', 0) or 0,
                getattr(response.usage_metadata, 'candidates_token_count', 0) or 0,
            )
        return _strip_think_tags(text)

    async def _genai_chat_async(
        self, messages: List[Dict[str, Any]], temperature: float, max_tokens: Optional[int]
    ) -> str:
        from google.genai import types

        contents, system_instruction = self._convert_messages_to_genai(messages)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens or 4096,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        response = await self._genai_client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        text = response.text or ""
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            self._track_usage(
                getattr(response.usage_metadata, 'prompt_token_count', 0) or 0,
                getattr(response.usage_metadata, 'candidates_token_count', 0) or 0,
            )
        return _strip_think_tags(text)

    def _genai_tools_sync(self, messages, tools, temperature, max_tokens):
        from google.genai import types

        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                fn = tool["function"]
                function_declarations.append(types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parameters=fn.get("parameters", {}),
                ))

        genai_tools = [types.Tool(function_declarations=function_declarations)]
        contents, system_instruction = self._convert_messages_to_genai(messages)
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens or 4096,
            tools=genai_tools,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        response = self._genai_client.models.generate_content(
            model=self.model, contents=contents, config=config,
        )
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            self._track_usage(
                getattr(response.usage_metadata, 'prompt_token_count', 0) or 0,
                getattr(response.usage_metadata, 'candidates_token_count', 0) or 0,
            )
        return self._wrap_genai_response(response)

    async def _genai_tools_async(self, messages, tools, temperature, max_tokens):
        from google.genai import types

        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                fn = tool["function"]
                function_declarations.append(types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parameters=fn.get("parameters", {}),
                ))

        genai_tools = [types.Tool(function_declarations=function_declarations)]
        contents, system_instruction = self._convert_messages_to_genai(messages)
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens or 4096,
            tools=genai_tools,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        response = await self._genai_client.aio.models.generate_content(
            model=self.model, contents=contents, config=config,
        )
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            self._track_usage(
                getattr(response.usage_metadata, 'prompt_token_count', 0) or 0,
                getattr(response.usage_metadata, 'candidates_token_count', 0) or 0,
            )
        return self._wrap_genai_response(response)

    def _wrap_genai_response(self, response):
        """Convert Google GenAI response to OpenAI-compatible wrapper."""
        text = ""
        tool_calls = []

        try:
            parts = response.candidates[0].content.parts or []
        except (IndexError, AttributeError):
            parts = []

        for part in parts:
            if hasattr(part, 'text') and part.text:
                text += part.text
            if hasattr(part, 'function_call') and part.function_call:
                fc = part.function_call
                tool_calls.append(_FakeToolCall(
                    id=f"call_{uuid4().hex[:8]}",
                    function=_FakeFunction(
                        name=fc.name,
                        arguments=json.dumps(dict(fc.args) if fc.args else {}),
                    ),
                ))

        return _FakeResponse(
            content=text or None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason="tool_calls" if tool_calls else "stop",
        )

    # ------------------------------------------------------------------
    # Anthropic helpers
    # ------------------------------------------------------------------

    def _anthropic_chat_sync(
        self, messages: List[Dict[str, Any]], temperature: float, max_tokens: Optional[int]
    ) -> str:
        system_msg, user_msgs = self._split_system(messages)
        prepared, has_pdf = self._prepare_messages_anthropic(user_msgs)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or 4096,
            "messages": prepared,
            "temperature": temperature,
        }
        if system_msg:
            kwargs["system"] = system_msg
        if has_pdf:
            kwargs["betas"] = ["pdfs-2024-09-25"]

        resp = self._anthropic_sync.beta.messages.create(**kwargs) if has_pdf else self._anthropic_sync.messages.create(**kwargs)
        content = resp.content[0].text if resp.content else ""
        if hasattr(resp, 'usage') and resp.usage:
            self._track_usage(
                getattr(resp.usage, 'input_tokens', 0),
                getattr(resp.usage, 'output_tokens', 0),
            )
        return _strip_think_tags(content)

    async def _anthropic_chat_async(
        self, messages: List[Dict[str, Any]], temperature: float, max_tokens: Optional[int]
    ) -> str:
        system_msg, user_msgs = self._split_system(messages)
        prepared, has_pdf = self._prepare_messages_anthropic(user_msgs)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or 4096,
            "messages": prepared,
            "temperature": temperature,
        }
        if system_msg:
            kwargs["system"] = system_msg
        if has_pdf:
            kwargs["betas"] = ["pdfs-2024-09-25"]

        resp = await self._anthropic_async.beta.messages.create(**kwargs) if has_pdf else await self._anthropic_async.messages.create(**kwargs)
        content = resp.content[0].text if resp.content else ""
        if hasattr(resp, 'usage') and resp.usage:
            self._track_usage(
                getattr(resp.usage, 'input_tokens', 0),
                getattr(resp.usage, 'output_tokens', 0),
            )
        return _strip_think_tags(content)

    # ------------------------------------------------------------------
    # Basic chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        logger.debug("[LLM.chat] Requesting %s with %d messages", self.model, len(messages))

        if self._is_google:
            result = self._genai_chat_sync(messages, temperature, max_tokens)
        elif self._is_anthropic:
            result = self._anthropic_chat_sync(messages, temperature, max_tokens)
        else:
            prepared = self._prepare_messages_openai(messages)
            params = self._build_params(prepared, temperature, max_tokens, **kwargs)
            response = self._sync_client.chat.completions.create(**params)
            content = response.choices[0].message.content or ""
            result = _strip_think_tags(content)
            if hasattr(response, 'usage') and response.usage:
                self._track_usage(
                    response.usage.prompt_tokens or 0,
                    response.usage.completion_tokens or 0,
                )

        logger.debug("[LLM.chat] Response length: %d chars", len(result))
        return result

    async def achat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        model_override: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        _model = model_override or self.model
        logger.debug("[LLM.achat] Requesting %s with %d messages", _model, len(messages))

        # Temporarily swap model if override is provided
        _orig_model = self.model
        if model_override:
            self.model = model_override

        last_error: Optional[Exception] = None
        try:
            for attempt in range(10):
                try:
                    if self._is_google:
                        result = await self._genai_chat_async(messages, temperature, max_tokens)
                    elif self._is_anthropic:
                        result = await self._anthropic_chat_async(messages, temperature, max_tokens)
                    else:
                        prepared = self._prepare_messages_openai(messages)
                        params = self._build_params(prepared, temperature, max_tokens, **kwargs)
                        response = await self._async_client.chat.completions.create(**params)
                        content = response.choices[0].message.content or ""
                        result = _strip_think_tags(content)
                        if hasattr(response, 'usage') and response.usage:
                            self._track_usage(
                                response.usage.prompt_tokens or 0,
                                response.usage.completion_tokens or 0,
                            )

                    logger.debug("[LLM.achat] Response length: %d chars", len(result))
                    return result
                except Exception as e:
                    err = str(e)
                    if any(code in err for code in ('503', '429', '500', 'UNAVAILABLE', 'rate_limit', 'overloaded')):
                        last_error = e
                        delay = min(10 * 2 ** attempt, 240)
                        logger.warning("[LLM.achat] Attempt %d/10 transient error: %s. Retrying in %ds...", attempt + 1, e, delay)
                        await asyncio.sleep(delay)
                        continue
                    raise
            raise last_error  # type: ignore[misc]
        finally:
            self.model = _orig_model

    # ------------------------------------------------------------------
    # JSON chat
    # ------------------------------------------------------------------

    def chat_json(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        raw = self.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
        cleaned = _strip_code_block(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("[LLM.chat_json] Failed to parse JSON. Raw:\n%s", raw)
            raise

    async def achat_json(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        raw = await self.achat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
        cleaned = _strip_code_block(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("[LLM.achat_json] Failed to parse JSON. Raw:\n%s", raw)
            raise

    # ------------------------------------------------------------------
    # Tool calling
    # ------------------------------------------------------------------

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str | Dict[str, Any] = "auto",
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        if self._is_anthropic:
            return self._anthropic_tools_sync(messages, tools, temperature, max_tokens)
        if self._is_google:
            return self._genai_tools_sync(messages, tools, temperature, max_tokens)

        params = self._build_params(messages, temperature, max_tokens, **kwargs)
        params["tools"] = tools
        params["tool_choice"] = tool_choice
        return self._sync_client.chat.completions.create(**params)

    async def achat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: str | Dict[str, Any] = "auto",
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        model_override: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Chat with tool calling support. Retries up to 4 times on tool-call loops."""
        _orig_model = self.model
        if model_override:
            self.model = model_override

        last_error: Optional[Exception] = None
        try:
            for attempt in range(10):
                try:
                    if self._is_anthropic:
                        return await self._anthropic_tools_async(messages, tools, temperature, max_tokens)
                    if self._is_google:
                        return await self._genai_tools_async(messages, tools, temperature, max_tokens)

                    params = self._build_params(messages, temperature, max_tokens, **kwargs)
                    params["tools"] = tools
                    params["tool_choice"] = tool_choice
                    _resp = await self._async_client.chat.completions.create(**params)
                    if hasattr(_resp, 'usage') and _resp.usage:
                        self._track_usage(
                            _resp.usage.prompt_tokens or 0,
                            _resp.usage.completion_tokens or 0,
                        )
                    return _resp
                except Exception as e:
                    err = str(e)
                    if any(code in err for code in ('503', '429', '500', 'UNAVAILABLE', 'rate_limit', 'overloaded')):
                        last_error = e
                        delay = min(10 * 2 ** attempt, 240)
                        logger.warning("[LLM.achat_with_tools] Attempt %d/10 transient: %s. Retry in %ds...", attempt + 1, e, delay)
                        await asyncio.sleep(delay)
                        continue
                    raise
            raise last_error  # type: ignore[misc]
        finally:
            self.model = _orig_model

    # ------------------------------------------------------------------
    # Anthropic tool calling
    # ------------------------------------------------------------------

    def _anthropic_tools_sync(self, messages, tools, temperature, max_tokens) -> Any:
        anthropic_tools = self._convert_tools_to_anthropic(tools)
        system_msg, user_msgs = self._split_system(messages)
        kwargs: Dict[str, Any] = {
            "model": self.model, "max_tokens": max_tokens or 4096,
            "messages": user_msgs, "temperature": temperature, "tools": anthropic_tools,
        }
        if system_msg:
            kwargs["system"] = system_msg
        return self._anthropic_sync.messages.create(**kwargs)

    async def _anthropic_tools_async(self, messages, tools, temperature, max_tokens) -> Any:
        anthropic_tools = self._convert_tools_to_anthropic(tools)
        system_msg, user_msgs = self._split_system(messages)
        kwargs: Dict[str, Any] = {
            "model": self.model, "max_tokens": max_tokens or 4096,
            "messages": user_msgs, "temperature": temperature, "tools": anthropic_tools,
        }
        if system_msg:
            kwargs["system"] = system_msg
        resp = await self._anthropic_async.messages.create(**kwargs)
        if hasattr(resp, 'usage') and resp.usage:
            self._track_usage(
                getattr(resp.usage, 'input_tokens', 0),
                getattr(resp.usage, 'output_tokens', 0),
            )
        return resp

    @staticmethod
    def _convert_tools_to_anthropic(openai_tools: List[Dict]) -> List[Dict]:
        result = []
        for tool in openai_tools:
            if tool.get("type") == "function":
                fn = tool["function"]
                result.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })
        return result

    @staticmethod
    def _split_system(messages: List[Dict]) -> tuple:
        system_msg = None
        user_msgs = []
        for m in messages:
            if m.get("role") == "system":
                content = m["content"]
                # Extract text from multimodal content array
                if isinstance(content, list):
                    system_msg = " ".join(
                        p["text"] for p in content if p.get("type") == "text"
                    )
                else:
                    system_msg = content
            else:
                user_msgs.append(m)
        return system_msg, user_msgs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_params(
        self, messages: List[Dict[str, Any]], temperature: float,
        max_tokens: Optional[int], **kwargs: Any,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "model": self.model, "messages": messages, "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        params.update(kwargs)
        return params
