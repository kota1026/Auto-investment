"""Path A: Claude vs Heuristic — head-to-head Alpha Arena comparison.

Run this on YOUR machine (NOT in any chat agent). Requires:
  1. pip install -r requirements.txt
  2. ANTHROPIC_API_KEY in your environment
  3. Internet access for Claude API calls

What it does:
  1. Runs the heuristic baseline (free, instant) on a 30-day contest
  2. Runs the Claude Opus 4.6 agent on the SAME 30-day contest (same seed
     so the underlying market data is identical)
  3. Compares the two head-to-head
  4. Saves both results to JSON for later analysis
  5. Estimates the API cost spent

Expected runtime: 5-15 minutes for the Claude run (180 decisions, each one
sends a fresh market snapshot to Opus 4.6 with adaptive thinking).

Expected cost: ~$2-3 total (with prompt caching reducing repeat input cost
by ~90% after the first call).

Usage:
    cd ~/Auto-investment
    export ANTHROPIC_API_KEY=sk-ant-...
    PYTHONPATH=src python scripts/run_contest_path_a.py

Output is written to:
    ./contest_results/heuristic_<timestamp>.json
    ./contest_results/claude_<timestamp>.json
    ./contest_results/comparison_<timestamp>.txt

Paste ONLY the comparison summary (the numbers) back to the chat agent
that helped you build this. NEVER paste your API key.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Make src/ importable when running from the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from auto_investment.alpha_arena import run_contest  # noqa: E402


# -----------------------------------------------------------------------------
# Config — edit these if you want a different contest setup
# -----------------------------------------------------------------------------

CONTEST_CONFIG = {
    "starting_capital": float(os.environ.get("CONTEST_CAPITAL", "30")),
    "duration_days": int(os.environ.get("CONTEST_DAYS", "30")),
    "decision_interval_hours": int(os.environ.get("CONTEST_INTERVAL", "4")),
    "use_real_data": os.environ.get("CONTEST_REAL_DATA", "false").lower() == "true",
    "seed": int(os.environ.get("CONTEST_SEED", "42")),
}

OUTPUT_DIR = REPO_ROOT / "contest_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def banner(title: str, char: str = "=") -> None:
    line = char * 70
    print(f"\n{line}\n  {title}\n{line}")


def fmt_pct(x: float) -> str:
    return f"{x:+.2f}%"


def fmt_usd(x: float) -> str:
    return f"${x:,.2f}"


def short_summary(result, label: str) -> str:
    return (
        f"{label:<12} "
        f"{fmt_usd(result['starting_capital'])} -> {fmt_usd(result['final_equity'])} "
        f"({fmt_pct(result['total_return_pct'])}) "
        f"vs BTC HODL {fmt_pct(result['benchmark_return_pct'])} "
        f"alpha={fmt_pct(result['alpha_vs_benchmark'])} "
        f"Sharpe={result['sharpe']:.2f} "
        f"MaxDD={result['max_drawdown_pct']:.1f}%"
    )


def estimate_cost(n_decisions: int) -> float:
    """Rough cost estimate for Claude Opus 4.6 calls.

    Per call (with prompt caching after the first):
      Input:  ~3000 tokens, ~10% paid (cached) = ~300 effective tokens
      Output: ~500 tokens
      Cost:   300 * $5/1M + 500 * $25/1M ≈ $0.0015 + $0.0125 = $0.014
    First call (no cache): ~$0.025
    """
    if n_decisions == 0:
        return 0.0
    return 0.025 + (n_decisions - 1) * 0.014


def save_result(result_dict: dict, name: str, timestamp: str) -> Path:
    path = OUTPUT_DIR / f"{name}_{timestamp}.json"
    with path.open("w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    return path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    banner("PATH A — Claude vs Heuristic head-to-head Alpha Arena contest")

    print(f"Config: {CONTEST_CONFIG}")
    print(f"Output dir: {OUTPUT_DIR}")

    # Sanity check: API key present
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("\n[ERROR] ANTHROPIC_API_KEY not set in environment.")
        print("Set it first:")
        print("    export ANTHROPIC_API_KEY=sk-ant-...")
        print("Then re-run this script.")
        return 1
    if not api_key.startswith("sk-ant-"):
        print(f"\n[WARN] ANTHROPIC_API_KEY does not look like an Anthropic key (starts with '{api_key[:10]}...')")
        print("Continuing anyway, but the Claude run will probably fail.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # -----------------------------------------------------------------------
    # Run 1: Heuristic baseline (free, instant)
    # -----------------------------------------------------------------------
    banner("Run 1/2 — HEURISTIC baseline (no API calls, ~5 seconds)")
    t0 = time.monotonic()
    heuristic = run_contest(use_ai=False, **CONTEST_CONFIG)
    h_secs = time.monotonic() - t0
    h_dict = heuristic.to_dict()
    h_path = save_result(h_dict, "heuristic", timestamp)
    print(f"\nDone in {h_secs:.1f}s. Saved: {h_path}")
    print(short_summary(h_dict, "HEURISTIC"))

    # -----------------------------------------------------------------------
    # Run 2: Claude Opus 4.6 (slow, costs ~$2-3)
    # -----------------------------------------------------------------------
    n_expected = (CONTEST_CONFIG["duration_days"] * 24) // CONTEST_CONFIG["decision_interval_hours"]
    est_cost = estimate_cost(n_expected)
    banner(f"Run 2/2 — CLAUDE Opus 4.6 (~{n_expected} decisions, ~${est_cost:.2f}, ~10 min)")
    print("Each decision call sends a market snapshot to Opus 4.6 with adaptive thinking.")
    print("Prompt caching reduces repeat input cost by ~90% after the first call.")
    print("Be patient — this loop is sequential (no batching).\n")

    t0 = time.monotonic()
    claude_result = run_contest(use_ai=True, **CONTEST_CONFIG)
    c_secs = time.monotonic() - t0
    c_dict = claude_result.to_dict()
    c_path = save_result(c_dict, "claude", timestamp)
    actual_cost = estimate_cost(c_dict["n_decisions"])
    print(f"\nDone in {c_secs:.1f}s. Saved: {c_path}")
    print(f"Estimated API spend: ${actual_cost:.2f}")
    print(short_summary(c_dict, "CLAUDE"))

    # -----------------------------------------------------------------------
    # Comparison
    # -----------------------------------------------------------------------
    banner("HEAD-TO-HEAD COMPARISON")

    rows = [
        ("Starting capital",      fmt_usd(h_dict["starting_capital"]),  fmt_usd(c_dict["starting_capital"])),
        ("Final equity",          fmt_usd(h_dict["final_equity"]),      fmt_usd(c_dict["final_equity"])),
        ("Total return",          fmt_pct(h_dict["total_return_pct"]),  fmt_pct(c_dict["total_return_pct"])),
        ("BTC HODL benchmark",    fmt_pct(h_dict["benchmark_return_pct"]), fmt_pct(c_dict["benchmark_return_pct"])),
        ("Alpha vs BTC",          fmt_pct(h_dict["alpha_vs_benchmark"]), fmt_pct(c_dict["alpha_vs_benchmark"])),
        ("Sharpe ratio",          f"{h_dict['sharpe']:.3f}",            f"{c_dict['sharpe']:.3f}"),
        ("Sortino ratio",         f"{h_dict['sortino']:.3f}",           f"{c_dict['sortino']:.3f}"),
        ("Max drawdown",          fmt_pct(h_dict["max_drawdown_pct"]),  fmt_pct(c_dict["max_drawdown_pct"])),
        ("Decisions made",        str(h_dict["n_decisions"]),           str(c_dict["n_decisions"])),
        ("Trades executed",       str(h_dict["n_trades"]),              str(c_dict["n_trades"])),
        ("Liquidations",          str(h_dict["n_liquidations"]),        str(c_dict["n_liquidations"])),
        ("Fees paid",             fmt_usd(h_dict["fees_paid"]),         fmt_usd(c_dict["fees_paid"])),
        ("Funding paid",          fmt_usd(h_dict["funding_paid"]),      fmt_usd(c_dict["funding_paid"])),
    ]

    col1_w = max(len(r[0]) for r in rows) + 2
    print(f"\n  {'Metric':<{col1_w}}{'Heuristic':>16}{'Claude Opus 4.6':>20}")
    print(f"  {'-' * col1_w}{'-' * 16}{'-' * 20}")
    for label, h_val, c_val in rows:
        print(f"  {label:<{col1_w}}{h_val:>16}{c_val:>20}")

    # Verdict
    banner("VERDICT", char="-")
    delta_return = c_dict["total_return_pct"] - h_dict["total_return_pct"]
    delta_sharpe = c_dict["sharpe"] - h_dict["sharpe"]
    delta_dd = c_dict["max_drawdown_pct"] - h_dict["max_drawdown_pct"]

    if delta_return > 0:
        print(f"  [+] Claude beat heuristic by {delta_return:+.2f} pp on total return")
    else:
        print(f"  [-] Claude lost to heuristic by {delta_return:+.2f} pp on total return")

    if delta_sharpe > 0:
        print(f"  [+] Claude improved Sharpe by {delta_sharpe:+.2f}")
    else:
        print(f"  [-] Claude degraded Sharpe by {delta_sharpe:+.2f}")

    if delta_dd > 0:  # less negative is better
        print(f"  [+] Claude reduced max drawdown by {abs(delta_dd):.2f} pp")
    else:
        print(f"  [-] Claude deepened max drawdown by {abs(delta_dd):.2f} pp")

    if c_dict["alpha_vs_benchmark"] > 0:
        print(f"  [+] Claude beat BTC buy-and-hold by {c_dict['alpha_vs_benchmark']:+.2f} pp")
    else:
        print(f"  [-] Claude lost to BTC buy-and-hold by {c_dict['alpha_vs_benchmark']:+.2f} pp")

    print(f"\n  Cost spent on Claude API: ~${actual_cost:.2f}")
    print(f"  Heuristic runtime: {h_secs:.1f}s")
    print(f"  Claude runtime: {c_secs / 60:.1f} min")

    # Save the comparison summary so it's easy to paste back to the chat
    summary_path = OUTPUT_DIR / f"comparison_{timestamp}.txt"
    summary_lines = ["PATH A — Claude vs Heuristic (synthetic data, $30, 30 days)\n"]
    summary_lines.append(f"Config: {CONTEST_CONFIG}\n")
    summary_lines.append(f"\n{'Metric':<{col1_w}}{'Heuristic':>16}{'Claude':>20}\n")
    summary_lines.append(f"{'-' * col1_w}{'-' * 16}{'-' * 20}\n")
    for label, h_val, c_val in rows:
        summary_lines.append(f"{label:<{col1_w}}{h_val:>16}{c_val:>20}\n")
    summary_lines.append(f"\nClaude vs heuristic delta: {delta_return:+.2f} pp return, {delta_sharpe:+.3f} Sharpe\n")
    summary_lines.append(f"Estimated API cost: ${actual_cost:.2f}\n")
    summary_path.write_text("".join(summary_lines))

    # If running under GitHub Actions, also write a Markdown summary that
    # shows up in the workflow run page directly
    gh_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_summary:
        with open(gh_summary, "a") as f:
            f.write("# Alpha Arena Contest — Path A Results\n\n")
            f.write(f"**Config:** `{CONTEST_CONFIG}`\n\n")
            f.write("| Metric | Heuristic | Claude Opus 4.6 |\n")
            f.write("|---|---:|---:|\n")
            for label, h_val, c_val in rows:
                f.write(f"| {label} | {h_val} | {c_val} |\n")
            f.write("\n## Verdict\n\n")
            arrow = "[+]" if delta_return > 0 else "[-]"
            f.write(f"- {arrow} Return delta: **{delta_return:+.2f} pp**\n")
            arrow = "[+]" if delta_sharpe > 0 else "[-]"
            f.write(f"- {arrow} Sharpe delta: **{delta_sharpe:+.3f}**\n")
            arrow = "[+]" if delta_dd > 0 else "[-]"
            f.write(f"- {arrow} Max DD delta: **{abs(delta_dd):.2f} pp** ({'better' if delta_dd > 0 else 'worse'})\n")
            f.write(f"- Estimated API cost: **${actual_cost:.2f}**\n")
            f.write(f"- Claude runtime: **{c_secs / 60:.1f} min**\n")
            print(f"\nWrote GitHub Actions summary to $GITHUB_STEP_SUMMARY")

    banner("WHAT TO DO NEXT")
    print(f"  1. Review the JSON files in {OUTPUT_DIR}/ for the full trade logs")
    print(f"  2. Paste {summary_path.name} (the SUMMARY ONLY, no API key) back to the agent")
    print(f"  3. We can then iterate on the prompt or move to Path B (real data)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
