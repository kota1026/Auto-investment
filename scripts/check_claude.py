"""Standalone Claude API health check.

Run this BEFORE a contest to verify that:
  1. ANTHROPIC_API_KEY is set and valid
  2. The anthropic SDK version supports our code path
  3. The model `claude-opus-4-6` is accessible
  4. `output_config.format` + `thinking={adaptive}` work together
  5. The Pydantic DecisionBatch schema round-trips cleanly

If all 5 pass, the real contest will work. If any fail, you see EXACTLY
what's broken and can fix it without burning time on a 30-day run that
silently falls back to the heuristic.

Usage:
    PYTHONPATH=src python scripts/check_claude.py

Exit codes:
    0 = all checks passed, Claude is ready
    1 = API key missing or obviously wrong
    2 = SDK import failed
    3 = API call failed
    4 = JSON schema / structured output failed
    5 = Pydantic validation failed
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"


def main() -> int:
    print(_bold("=" * 70))
    print(_bold("  Claude API Health Check"))
    print(_bold("=" * 70))

    # ---- Check 1: API key ---------------------------------------------------
    print("\n[1/5] Checking ANTHROPIC_API_KEY ... ", end="")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(_red("FAIL"))
        print("       ANTHROPIC_API_KEY is not set in environment.")
        print("       In GitHub Actions, verify:")
        print("         1. Settings → Secrets and variables → Actions")
        print("         2. 'Repository secrets' tab (NOT Environment secrets)")
        print("         3. Name is EXACTLY 'ANTHROPIC_API_KEY' (case-sensitive)")
        return 1
    if not api_key.startswith("sk-ant-"):
        print(_yellow("WARN"))
        print(f"       Key does not start with 'sk-ant-' (got: {api_key[:10]}...)")
        print("       Continuing anyway, but this probably won't work.")
    else:
        print(_green("OK"))
        print(f"       Key prefix: {api_key[:15]}...")

    # ---- Check 2: SDK import ------------------------------------------------
    print("[2/5] Importing anthropic SDK ... ", end="")
    try:
        import anthropic  # noqa: PLC0415

        version = getattr(anthropic, "__version__", "unknown")
        print(_green("OK"))
        print(f"       anthropic version: {version}")
    except ImportError as exc:
        print(_red("FAIL"))
        print(f"       Could not import anthropic: {exc}")
        print("       Fix: pip install 'anthropic>=0.60.0'")
        return 2

    # ---- Check 3: Basic API call --------------------------------------------
    print("[3/5] Calling messages.create() with claude-opus-4-6 ... ", end="")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Respond with exactly the word 'OK' and nothing else."}
            ],
        )
        text_parts = [
            getattr(b, "text", "")
            for b in response.content
            if getattr(b, "type", None) == "text"
        ]
        text = "".join(text_parts).strip()
        print(_green("OK"))
        print(f"       Response: {text!r}")
        print(f"       Model: {response.model}")
        print(
            f"       Usage: input={response.usage.input_tokens}, "
            f"output={response.usage.output_tokens}"
        )
    except Exception as exc:  # noqa: BLE001
        print(_red("FAIL"))
        print(f"       {type(exc).__name__}: {exc}")
        print("       This is the most common failure point. Likely causes:")
        print("         - Wrong API key (revoke old, create new)")
        print("         - No credits on the account")
        print("         - Model name invalid (try 'claude-sonnet-4-6')")
        print("         - Network blocked")
        return 3

    # ---- Check 4: Structured output (the path the contest uses) -------------
    print("[4/5] Testing output_config.format + thinking=adaptive ... ", end="")
    test_schema = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["yes", "no"]},
            "confidence": {"type": "number"},
        },
        "required": ["verdict", "confidence"],
        "additionalProperties": False,
    }
    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=200,
            thinking={"type": "adaptive"},
            messages=[
                {
                    "role": "user",
                    "content": "Is 2+2=4? Return {verdict: 'yes'|'no', confidence: 0-1}",
                }
            ],
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": test_schema},
            },
        )
        text_parts = [
            getattr(b, "text", "")
            for b in response.content
            if getattr(b, "type", None) == "text"
        ]
        text = "".join(text_parts).strip()
        data = json.loads(text)
        print(_green("OK"))
        print(f"       Parsed: {data}")
    except Exception as exc:  # noqa: BLE001
        print(_red("FAIL"))
        print(f"       {type(exc).__name__}: {exc}")
        print("       This means structured outputs aren't working on this SDK version.")
        print("       Fix: upgrade anthropic ('pip install -U anthropic')")
        return 4

    # ---- Check 5: Full DecisionBatch round-trip -----------------------------
    print("[5/5] Testing DecisionBatch schema round-trip ... ", end="")
    try:
        from auto_investment.decision_agent import (  # noqa: PLC0415
            _DECISION_JSON_SCHEMA,
            DecisionBatch,
        )

        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            thinking={"type": "adaptive"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Return a DecisionBatch where every symbol in [BTC, ETH, SOL] "
                        "has action='hold', confidence=0.5, rationale='health check test'. "
                        "overall_thesis can be any short sentence."
                    ),
                }
            ],
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": _DECISION_JSON_SCHEMA},
            },
        )
        text_parts = [
            getattr(b, "text", "")
            for b in response.content
            if getattr(b, "type", None) == "text"
        ]
        text = "".join(text_parts).strip()
        data = json.loads(text)
        batch = DecisionBatch.model_validate(data)
        print(_green("OK"))
        print(f"       {len(batch.decisions)} decisions, thesis: {batch.overall_thesis[:80]}")
    except Exception as exc:  # noqa: BLE001
        print(_red("FAIL"))
        print(f"       {type(exc).__name__}: {exc}")
        return 5

    # ---- All passed ---------------------------------------------------------
    print()
    print(_bold(_green("=" * 70)))
    print(_bold(_green("  ALL 5 CHECKS PASSED — Claude is ready for the contest")))
    print(_bold(_green("=" * 70)))

    gh_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_summary:
        with open(gh_summary, "a") as f:
            f.write("# Claude API Health Check\n\n")
            f.write("✅ All 5 checks passed\n\n")
            f.write("| Check | Status |\n|---|---|\n")
            f.write("| ANTHROPIC_API_KEY set | ✅ |\n")
            f.write(f"| anthropic SDK import (v{version}) | ✅ |\n")
            f.write("| Basic messages.create() | ✅ |\n")
            f.write("| output_config.format + adaptive thinking | ✅ |\n")
            f.write("| DecisionBatch schema round-trip | ✅ |\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
