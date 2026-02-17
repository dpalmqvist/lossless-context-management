"""Type-aware file analysis for large file references."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path

from lcm.llm.client import summarize

# Extensions handled deterministically (no LLM needed)
DETERMINISTIC_TYPES = {".json", ".csv", ".tsv", ".jsonl", ".ndjson"}

# Extensions analyzed with LLM for structure extraction
CODE_TYPES = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".rb"}


async def analyze_file(
    file_path: str, model: str | None = None
) -> dict[str, str | int | None]:
    """Analyze a file and return metadata + exploration summary.

    Returns dict with keys: file_type, size_bytes, exploration_summary
    """
    path = Path(file_path)

    if not path.exists():
        return {
            "file_type": None,
            "size_bytes": None,
            "exploration_summary": f"File not found: {file_path}",
        }

    size_bytes = path.stat().st_size
    suffix = path.suffix.lower()

    if suffix in DETERMINISTIC_TYPES:
        summary = _analyze_deterministic(path, suffix)
    elif suffix in CODE_TYPES:
        summary = await _analyze_code(path, suffix, model=model)
    else:
        summary = await _analyze_generic(path, model=model)

    return {
        "file_type": suffix.lstrip("."),
        "size_bytes": size_bytes,
        "exploration_summary": summary,
    }


def _analyze_deterministic(path: Path, suffix: str) -> str:
    """Deterministic analysis for structured data files."""
    try:
        content = path.read_text(errors="replace")[:50_000]  # Cap read

        if suffix == ".json":
            return _analyze_json(content)
        elif suffix in (".csv", ".tsv"):
            delimiter = "\t" if suffix == ".tsv" else ","
            return _analyze_csv(content, delimiter)
        elif suffix in (".jsonl", ".ndjson"):
            return _analyze_jsonl(content)

    except Exception as e:
        return f"Error analyzing {path.name}: {e}"

    return f"Structured data file: {path.name}"


def _analyze_json(content: str) -> str:
    """Extract schema and shape from JSON."""
    try:
        data = json.loads(content)
        return f"JSON: {_describe_shape(data)}"
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"


def _analyze_csv(content: str, delimiter: str) -> str:
    """Extract schema and row count from CSV/TSV."""
    reader = csv.reader(io.StringIO(content), delimiter=delimiter)
    rows = list(reader)
    if not rows:
        return "Empty CSV"
    headers = rows[0]
    return f"CSV: {len(rows)-1} rows, columns: {headers}"


def _analyze_jsonl(content: str) -> str:
    """Analyze JSONL: count lines, extract schema from first entry."""
    lines = [l for l in content.strip().split("\n") if l.strip()]
    if not lines:
        return "Empty JSONL"
    try:
        first = json.loads(lines[0])
        shape = _describe_shape(first)
        return f"JSONL: {len(lines)} lines, first entry schema: {shape}"
    except json.JSONDecodeError:
        return f"JSONL: {len(lines)} lines (parse error on first)"


def _describe_shape(obj: object, depth: int = 0, max_depth: int = 3) -> str:
    """Recursively describe the shape/schema of a JSON object."""
    if depth >= max_depth:
        return "..."

    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = []
        for k, v in list(obj.items())[:10]:
            items.append(f"{k}: {_describe_shape(v, depth + 1, max_depth)}")
        suffix = ", ..." if len(obj) > 10 else ""
        return "{" + ", ".join(items) + suffix + "}"
    elif isinstance(obj, list):
        if not obj:
            return "[]"
        return f"[{_describe_shape(obj[0], depth + 1, max_depth)}] ({len(obj)} items)"
    elif isinstance(obj, str):
        return "str"
    elif isinstance(obj, bool):
        return "bool"
    elif isinstance(obj, int):
        return "int"
    elif isinstance(obj, float):
        return "float"
    elif obj is None:
        return "null"
    return type(obj).__name__


async def _analyze_code(
    path: Path, suffix: str, model: str | None = None
) -> str:
    """LLM-powered analysis for code files — extract signatures and structure."""
    content = path.read_text(errors="replace")[:30_000]

    prompt_text = (
        f"Analyze this {suffix} file and list:\n"
        "1. Function/method signatures (name, params, return type)\n"
        "2. Class names and their hierarchies\n"
        "3. Key imports\n"
        "4. Module-level constants\n"
        "Be concise — just signatures and names, no implementations."
    )

    kwargs = {}
    if model:
        kwargs["model"] = model

    try:
        result = await summarize(
            f"{prompt_text}\n\n```{suffix}\n{content}\n```",
            mode="preserve_details",
            target_tokens=600,
            **kwargs,
        )
        return result
    except Exception as e:
        # Fall back to basic line count
        line_count = content.count("\n") + 1
        return f"{suffix} file: {line_count} lines (LLM analysis failed: {e})"


async def _analyze_generic(path: Path, model: str | None = None) -> str:
    """LLM-powered analysis for other file types."""
    try:
        content = path.read_text(errors="replace")[:20_000]
    except Exception:
        return f"Binary or unreadable file: {path.name}"

    kwargs = {}
    if model:
        kwargs["model"] = model

    try:
        result = await summarize(
            f"Briefly describe the contents and purpose of this file:\n\n{content[:5000]}",
            mode="bullet_points",
            target_tokens=300,
            **kwargs,
        )
        return result
    except Exception as e:
        line_count = content.count("\n") + 1
        return f"Text file: {line_count} lines (LLM analysis failed: {e})"
