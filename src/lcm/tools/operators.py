"""Parallel data processing operators: llm_map, agentic_map."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from lcm.llm.client import agent_loop, classify


async def llm_map(
    input_path: str,
    prompt: str,
    output_schema: dict | None = None,
    concurrency: int = 16,
    max_retries: int = 3,
) -> dict:
    """Process each line of a JSONL file with a stateless LLM call.

    Fan out Haiku calls via asyncio with a semaphore for concurrency control.
    Validates against schema, retries failures.
    Returns path to output JSONL + stats.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        return {"error": f"Input file not found: {input_path}"}

    output_path = input_file.with_suffix(".out.jsonl")
    items = _read_jsonl(input_file)

    if not items:
        return {"error": "Empty input file or no valid JSONL lines"}

    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict | None] = [None] * len(items)
    errors: list[dict] = []

    async def process_item(idx: int, item: dict) -> None:
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    result = await classify(
                        item=item, prompt=prompt, output_schema=output_schema
                    )
                    results[idx] = result
                    return
                except Exception as e:
                    if attempt == max_retries - 1:
                        errors.append(
                            {"index": idx, "error": str(e), "item": item}
                        )

    tasks = [process_item(i, item) for i, item in enumerate(items)]
    await asyncio.gather(*tasks)

    # Write output
    successful = [r for r in results if r is not None]
    _write_jsonl(output_path, successful)

    return {
        "output_path": str(output_path),
        "total_items": len(items),
        "successful": len(successful),
        "failed": len(errors),
        "errors": errors[:10],  # Cap error list
    }


async def agentic_map(
    input_path: str,
    prompt: str,
    output_schema: dict | None = None,
    read_only: bool = True,
    concurrency: int = 4,
    max_retries: int = 3,
) -> dict:
    """Process each JSONL item with a multi-turn agent loop.

    Each item gets a multi-turn conversation with tool access.
    Lower concurrency than llm_map due to higher cost per item.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        return {"error": f"Input file not found: {input_path}"}

    output_path = input_file.with_suffix(".agent_out.jsonl")
    items = _read_jsonl(input_file)

    if not items:
        return {"error": "Empty input file or no valid JSONL lines"}

    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict | None] = [None] * len(items)
    errors: list[dict] = []

    async def process_item(idx: int, item: dict) -> None:
        async with semaphore:
            for attempt in range(max_retries):
                try:
                    result = await agent_loop(
                        item=item,
                        prompt=prompt,
                        output_schema=output_schema,
                        read_only=read_only,
                    )
                    results[idx] = result
                    return
                except Exception as e:
                    if attempt == max_retries - 1:
                        errors.append(
                            {"index": idx, "error": str(e), "item": item}
                        )

    tasks = [process_item(i, item) for i, item in enumerate(items)]
    await asyncio.gather(*tasks)

    successful = [r for r in results if r is not None]
    _write_jsonl(output_path, successful)

    return {
        "output_path": str(output_path),
        "total_items": len(items),
        "successful": len(successful),
        "failed": len(errors),
        "errors": errors[:10],
    }


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, skipping invalid lines."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def _write_jsonl(path: Path, items: list[dict]) -> None:
    """Write items to a JSONL file."""
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
