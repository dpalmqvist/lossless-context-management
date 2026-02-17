"""Three-level summarization escalation."""

from __future__ import annotations

from dataclasses import dataclass

from lcm.llm.client import summarize


@dataclass
class EscalationResult:
    content: str
    level: int  # 1, 2, or 3
    mode: str
    token_estimate: int


def _truncate_deterministic(text: str, max_tokens: int = 512) -> str:
    """Level 3: Deterministic truncation — no LLM call.

    Takes the first and last portions of text to fit within max_tokens.
    """
    max_chars = max_tokens * 4  # ~4 chars per token
    if len(text) <= max_chars:
        return text

    half = max_chars // 2
    return text[:half] + "\n[...truncated...]\n" + text[-half:]


async def escalated_summarize(
    content: str,
    target_tokens: int = 500,
    model: str | None = None,
) -> EscalationResult:
    """Try three escalation levels, returning the first that achieves size reduction.

    Level 1: Haiku with preserve_details, target T tokens
    Level 2: Haiku with bullet_points, target T/2 tokens
    Level 3: Deterministic truncation to 512 tokens (no LLM)
    """
    original_estimate = max(1, len(content) // 4)

    # If already small enough, return as-is
    if original_estimate <= target_tokens:
        return EscalationResult(
            content=content,
            level=0,
            mode="passthrough",
            token_estimate=original_estimate,
        )

    kwargs = {}
    if model:
        kwargs["model"] = model

    # Level 1: preserve_details
    try:
        result = await summarize(
            content, mode="preserve_details", target_tokens=target_tokens, **kwargs
        )
        estimate = max(1, len(result) // 4)
        if estimate < original_estimate:
            return EscalationResult(
                content=result,
                level=1,
                mode="preserve_details",
                token_estimate=estimate,
            )
    except Exception:
        pass  # Fall through to next level

    # Level 2: bullet_points at half target
    try:
        result = await summarize(
            content,
            mode="bullet_points",
            target_tokens=target_tokens // 2,
            **kwargs,
        )
        estimate = max(1, len(result) // 4)
        if estimate < original_estimate:
            return EscalationResult(
                content=result,
                level=2,
                mode="bullet_points",
                token_estimate=estimate,
            )
    except Exception:
        pass  # Fall through to next level

    # Level 3: Deterministic truncation
    result = _truncate_deterministic(content, max_tokens=512)
    estimate = max(1, len(result) // 4)
    return EscalationResult(
        content=result,
        level=3,
        mode="deterministic_truncate",
        token_estimate=estimate,
    )
