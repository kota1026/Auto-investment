"""Auto-track contest results — append to history, update README, commit.

Called from GitHub Actions workflows after each contest run. Reads the
latest result JSON, appends a one-line summary to results/history.jsonl,
and regenerates results/README.md as a sortable Markdown table of all
historical runs. The workflow then commits and pushes the updated files.

This is what makes the automation "self-improving": every run leaves a
trace, so over time you build a longitudinal dataset of how the strategy
performs across different markets / seeds / prompt versions.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
HISTORY_FILE = RESULTS_DIR / "history.jsonl"
README_FILE = RESULTS_DIR / "README.md"


def append_to_history(entry: dict) -> None:
    """Append one entry to history.jsonl (newline-delimited JSON)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    with HISTORY_FILE.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    entries = []
    with HISTORY_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def regenerate_readme() -> None:
    """Build a Markdown table of every historical run."""
    history = load_history()
    if not history:
        README_FILE.write_text("# Contest results\n\nNo runs yet.\n")
        return

    lines = [
        "# Contest Results — auto-generated\n",
        f"_Last updated: {datetime.utcnow().isoformat()}Z_\n",
        f"_Total runs: {len(history)}_\n",
        "",
        "Each row is one Alpha Arena contest run. Click into the JSON file in",
        "this directory for the full trade log + decision log.",
        "",
        "| Date | Mode | Capital | Days | Real? | Final | Return | Sharpe | MaxDD | Trades | Cost |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    for entry in reversed(history):  # newest first
        timestamp = entry.get("timestamp", "?")
        date = timestamp.split("T")[0] if "T" in timestamp else timestamp[:10]
        mode = entry.get("mode", "?")
        capital = entry.get("starting_capital", 0)
        days = entry.get("duration_days", 0)
        real = "yes" if entry.get("use_real_data") else "syn"
        final = entry.get("final_equity", 0)
        ret = entry.get("total_return_pct", 0)
        sharpe = entry.get("sharpe", 0)
        max_dd = entry.get("max_drawdown_pct", 0)
        trades = entry.get("n_trades", 0)
        cost = entry.get("estimated_api_cost", 0)
        lines.append(
            f"| {date} | {mode} | ${capital:.0f} | {days} | {real} | "
            f"${final:.2f} | {ret:+.2f}% | {sharpe:+.2f} | {max_dd:.1f}% | "
            f"{trades} | ${cost:.2f} |"
        )

    # Aggregate stats by mode
    lines.append("")
    lines.append("## Aggregate by mode")
    lines.append("")
    lines.append("| Mode | N | Mean return | Mean Sharpe | Win rate vs BTC |")
    lines.append("|---|---:|---:|---:|---:|")

    by_mode: dict[str, list[dict]] = {}
    for e in history:
        by_mode.setdefault(e.get("mode", "?"), []).append(e)

    for mode, entries in sorted(by_mode.items()):
        if not entries:
            continue
        n = len(entries)
        mean_ret = sum(e.get("total_return_pct", 0) for e in entries) / n
        mean_sharpe = sum(e.get("sharpe", 0) for e in entries) / n
        wins = sum(
            1
            for e in entries
            if e.get("total_return_pct", 0) > e.get("benchmark_return_pct", 0)
        )
        win_rate = wins / n if n else 0
        lines.append(
            f"| {mode} | {n} | {mean_ret:+.2f}% | {mean_sharpe:+.2f} | "
            f"{win_rate * 100:.0f}% |"
        )

    README_FILE.write_text("\n".join(lines) + "\n")


def main() -> int:
    """Read latest result file from contest_results/ and add to history."""
    contest_dir = REPO_ROOT / "contest_results"
    if not contest_dir.exists():
        print(f"No {contest_dir} found; nothing to track")
        return 0

    # Find the latest claude_*.json or heuristic_*.json
    latest_claude = sorted(contest_dir.glob("claude_*.json"))
    latest_heur = sorted(contest_dir.glob("heuristic_*.json"))

    for fname, mode in [(latest_claude, "claude"), (latest_heur, "heuristic")]:
        if not fname:
            continue
        latest = fname[-1]
        try:
            data = json.loads(latest.read_text())
        except Exception as exc:
            print(f"Failed to read {latest}: {exc}")
            continue
        # Estimate cost if available
        n_decisions = data.get("n_decisions", 0)
        est_cost = (
            0.025 + (n_decisions - 1) * 0.014 if mode == "claude" and n_decisions > 0 else 0.0
        )
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "mode": mode,
            "starting_capital": data.get("starting_capital"),
            "duration_days": (data.get("duration_hours", 0) // 24) or 0,
            "use_real_data": False,  # propagated separately if needed
            "final_equity": data.get("final_equity"),
            "total_return_pct": data.get("total_return_pct"),
            "benchmark_return_pct": data.get("benchmark_return_pct"),
            "alpha_vs_benchmark": data.get("alpha_vs_benchmark"),
            "sharpe": data.get("sharpe"),
            "sortino": data.get("sortino"),
            "max_drawdown_pct": data.get("max_drawdown_pct"),
            "n_trades": data.get("n_trades"),
            "n_liquidations": data.get("n_liquidations"),
            "estimated_api_cost": round(est_cost, 4),
            "source_file": latest.name,
        }
        append_to_history(entry)
        print(f"Tracked: {mode} → {entry['final_equity']:.2f} ({entry['total_return_pct']:+.2f}%)")

    regenerate_readme()
    print(f"Updated {README_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
