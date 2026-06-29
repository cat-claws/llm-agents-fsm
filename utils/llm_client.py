"""Shared OpenAI-compatible client and LLM call helper for local model server on port 30000."""

import json
from typing import Any

import openai


def make_client() -> openai.OpenAI:
    return openai.OpenAI(base_url="http://127.0.0.1:30000/v1/", api_key="EMPTY")


def extra_body() -> dict:
    return {
        "chat_template_kwargs": {"enable_thinking": True},
        "separate_reasoning": True,
    }


def llm(
    client: openai.OpenAI,
    messages: list[dict],
    model: str,
    *,
    tools: list | None = None,
    tool_choice: dict | str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> tuple[str, list]:
    """Call the model and return (content, tool_calls).

    tool_calls is a list of dicts with keys 'id' and 'function' (with 'name' and 'arguments').
    When tools is None, tool_calls will always be empty.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": extra_body(),
    }
    if tools:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        raise RuntimeError("OpenAI-compatible chat error: %s" % exc) from exc
    try:
        msg = response.choices[0].message
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected OpenAI-compatible response: %r" % response) from exc
    content = (msg.content or "").strip()
    tool_calls = []
    for tc in msg.tool_calls or []:
        try:
            arguments = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        tool_calls.append({
            "id": tc.id,
            "function": {
                "name": tc.function.name,
                "arguments": arguments,
            },
        })
    return content, tool_calls


def tool_arguments(tool_calls: list, name: str) -> dict | None:
    """Return arguments for the first tool call with the requested function name."""
    for tool_call in tool_calls:
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if not isinstance(function, dict) or function.get("name") != name:
            continue
        arguments = function.get("arguments")
        return arguments if isinstance(arguments, dict) else {}
    return None
