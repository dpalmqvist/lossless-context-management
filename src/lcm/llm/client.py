"""Async Anthropic API wrapper for summarization and LLM operations."""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def _get_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic()


async def summarize(
    content: str,
    mode: str = "preserve_details",
    target_tokens: int = 500,
    model: str = DEFAULT_MODEL,
) -> str:
    """Summarize content using the Anthropic API.

    Modes:
      - preserve_details: Retain key details, decisions, and code references
      - bullet_points: Compress to concise bullet points
    """
    system_prompts = {
        "preserve_details": (
            f"Summarize the following conversation segment in at most {target_tokens} tokens. "
            "Preserve: key decisions, code references (file paths, function names), "
            "error messages, and action items. Use concise prose."
        ),
        "bullet_points": (
            f"Summarize the following in at most {target_tokens} tokens as bullet points. "
            "Focus on: what was done, what was decided, what files were changed. "
            "Be extremely concise."
        ),
    }
    system = system_prompts.get(mode, system_prompts["preserve_details"])

    client = _get_client()
    response = await client.messages.create(
        model=model,
        max_tokens=target_tokens * 2,  # Allow some headroom
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text


async def classify(
    item: Any,
    prompt: str,
    output_schema: dict | None = None,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Single stateless LLM call for classification/transformation (used by llm_map).

    Returns parsed JSON from the LLM response.
    """
    system = "You are a data processing assistant. Respond only with valid JSON."
    if output_schema:
        system += f"\n\nOutput must conform to this JSON schema:\n{json.dumps(output_schema)}"

    user_content = f"{prompt}\n\nInput:\n{json.dumps(item)}"

    client = _get_client()
    response = await client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )

    text = response.content[0].text.strip()
    # Try to extract JSON from response
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    return json.loads(text)


async def agent_loop(
    item: Any,
    prompt: str,
    output_schema: dict | None = None,
    tools: list[dict] | None = None,
    read_only: bool = True,
    model: str = DEFAULT_MODEL,
    max_turns: int = 10,
) -> dict:
    """Multi-turn agent loop for agentic_map.

    Each item gets multiple turns with tool use. Tools are restricted
    based on read_only flag.
    """
    system = "You are a data processing agent. Process the given item using available tools."
    if output_schema:
        system += f"\n\nFinal output must conform to this JSON schema:\n{json.dumps(output_schema)}"
    if read_only:
        system += "\n\nYou are in read-only mode. Do not modify any files."

    default_tools = []
    if tools is None:
        # Provide basic tools
        default_tools = [
            {
                "name": "read_file",
                "description": "Read the contents of a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"}
                    },
                    "required": ["path"],
                },
            },
        ]
        if not read_only:
            default_tools.append(
                {
                    "name": "bash",
                    "description": "Execute a bash command",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Command to execute",
                            }
                        },
                        "required": ["command"],
                    },
                }
            )

    api_tools = tools or default_tools
    user_content = f"{prompt}\n\nInput:\n{json.dumps(item)}"

    messages = [{"role": "user", "content": user_content}]
    client = _get_client()

    for _turn in range(max_turns):
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "system": system,
            "messages": messages,
        }
        if api_tools:
            kwargs["tools"] = api_tools

        response = await client.messages.create(**kwargs)

        # Check if we have a final text response (no tool use)
        if response.stop_reason == "end_turn":
            # Extract text content
            for block in response.content:
                if block.type == "text":
                    text = block.text.strip()
                    if text.startswith("```"):
                        lines = text.split("\n")
                        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {"result": text}
            return {"result": "No output"}

        # Handle tool use
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        tool_results = []
        for block in assistant_content:
            if block.type == "tool_use":
                result = await _execute_tool(block.name, block.input, read_only)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    # If we exhausted turns, return last response text
    return {"result": "Max turns reached", "partial": True}


async def _execute_tool(name: str, input_data: dict, read_only: bool) -> str:
    """Execute a tool call. Limited to safe operations."""
    import subprocess

    if name == "read_file":
        try:
            path = input_data.get("path", "")
            with open(path) as f:
                content = f.read(100_000)  # Cap at 100K chars
            return content
        except Exception as e:
            return f"Error reading file: {e}"

    elif name == "bash" and not read_only:
        try:
            result = subprocess.run(
                input_data.get("command", "echo 'no command'"),
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR: {result.stderr}"
            return output[:50_000]  # Cap output
        except subprocess.TimeoutExpired:
            return "Error: command timed out (30s)"
        except Exception as e:
            return f"Error: {e}"

    return f"Unknown tool: {name}"
