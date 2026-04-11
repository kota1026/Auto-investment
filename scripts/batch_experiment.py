"""Batch experiment runner — Monte Carlo over N seeds for statistical confidence.

The single-run contest gives you ONE number — easy to fool yourself with luck.
This script runs N independent contests (different seeds) and aggregates the
distribution. With N=10, you can compute mean, std, win rate vs baseline,
and p-value of "Claude actually has edge".

Why this matters:
  Alpha Arena Season 1 ran for 17 days with N=1 trial per model. That's a
  single noisy data point. We can do better by running multiple seeded
  contests in batch and aggregating. This gives us a confidence interval
  on Claude's actual edge over the heuristic.

Usage (locally):
    PYTHONPATH=src python scripts/batch_experiment.py

Usage (CI):
    Set BATCH_N_SEEDS to control how many trials. Defaults from env.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from auto_investment.alpha_arena import run_contest  # noqa: E402


# -----------------------------------------------------------------------------
# Config from environment
# -----------------------------------------------------------------------------

N_SEEDS = int(os.environ.get("BATCH_N_SEEDS", "10"))
STARTING_CAPITAL = float(os.environ.get("BATCH_CAPITAL", "30"))
DURATION_DAYS = int(os.environ.get("BATCH_DAYS", "30"))
DECISION_INTERVAL = int(os.environ.get("BATCH_INTERVAL", "4"))
USE_REAL_DATA = os.environ.get("BATCH_REAL_DATA", "false").lower() == "true"
USE_AI = os.environ.get("BATCH_USE_AI", "false").lower() == "true"

OUTPUT_DIR = REPO_ROOT / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------------


def t_stat(values_a: list[float], values_b: list[float]) -> float:
    """Welch's t-test statistic for comparing two unequal-variance samples.

    Used to test 'is Claude's mean return significantly different from the
    heuristic's?' A |t| > 2 with N=10 each is roughly p < 0.05.
    """
    n_a, n_b = len(values_a), len(values_b)
    if n_a < 2 or n_b < 2:
        return 0.0
    mean_a = statistics.mean(values_a)
    mean_b = statistics.mean(values_b)
    var_a = statistics.variance(values_a)
    var_b = statistics.variance(values_b)
    pooled_se = math.sqrt(var_a / n_a + var_b / n_b)
    if pooled_se == 0:
        return 0.0
    return (mean_a - mean_b) / pooled_se


def summarize(label: str, values: list[float]) -> dict:
    if not values:
        return {"label": label, "n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "label": label,
        "n": len(values),
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values) if len(values) > 1 else 0.0, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "median": round(statistics.median(values), 4),
    }


def banner(title: str, ch: str = "=") -> None:
    print(f"\n{ch * 72}\n  {title}\n{ch * 72}")


def fmt_pct(x: float) -> str:
    return f"{x:+.2f}%"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run_one(seed: int, use_ai: bool) -> dict:
    """Run a single contest and return its summary dict."""
    label = "claude" if use_ai else "heuristic"
    print(f"  [{label}] seed={seed} ...", end="", flush=True)
    t0 = time.monotonic()
    result = run_contest(
        starting_capital=STARTING_CAPITAL,
        duration_days=DURATION_DAYS,
        decision_interval_hours=DECISION_INTERVAL,
        use_real_data=USE_REAL_DATA,
        use_ai=use_ai,
        seed=seed,
    )
    elapsed = time.monotonic() - t0
    print(f" {fmt_pct(result.total_return_pct)} (Sharpe {result.sharpe:+.2f}, {elapsed:.1f}s)")
    return result.to_dict()


def main() -> int:
    banner(f"BATCH EXPERIMENT — N={N_SEEDS}, days={DURATION_DAYS}, capital=${STARTING_CAPITAL}")
    print(f"  use_ai={USE_AI}  use_real_data={USE_REAL_DATA}")
    print(f"  output: {OUTPUT_DIR}")

    if USE_AI and not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n[ERROR] BATCH_USE_AI=true but ANTHROPIC_API_KEY is not set.")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Always run heuristic baseline (free, fast, gives the comparison floor)
    banner("PHASE 1 — Heuristic baseline (free)", ch="-")
    heuristic_results = []
    for seed in range(1, N_SEEDS + 1):
        heuristic_results.append(run_one(seed, use_ai=False))

    claude_results: list[dict] = []
    if USE_AI:
        banner(f"PHASE 2 — Claude runs (~${0.025 + (N_SEEDS - 1) * 0.014 + N_SEEDS * 2.5:.2f} estimated)", ch="-")
        for seed in range(1, N_SEEDS + 1):
            claude_results.append(run_one(seed, use_ai=True))
    else:
        print("\n  (Skipping Claude phase — set BATCH_USE_AI=true to enable)")

    # ---- Aggregate ---------------------------------------------------------
    h_returns = [r["total_return_pct"] for r in heuristic_results]
    h_sharpes = [r["sharpe"] for r in heuristic_results]
    h_dds = [r["max_drawdown_pct"] for r in heuristic_results]
    h_alpha = [r["alpha_vs_benchmark"] for r in heuristic_results]

    h_stats = {
        "return_pct": summarize("Heuristic return %", h_returns),
        "sharpe": summarize("Heuristic Sharpe", h_sharpes),
        "max_dd_pct": summarize("Heuristic max DD %", h_dds),
        "alpha_vs_btc": summarize("Heuristic alpha vs BTC", h_alpha),
    }

    c_stats = None
    t_return = 0.0
    t_sharpe = 0.0
    if claude_results:
        c_returns = [r["total_return_pct"] for r in claude_results]
        c_sharpes = [r["sharpe"] for r in claude_results]
        c_dds = [r["max_drawdown_pct"] for r in claude_results]
        c_alpha = [r["alpha_vs_benchmark"] for r in claude_results]

        c_stats = {
            "return_pct": summarize("Claude return %", c_returns),
            "sharpe": summarize("Claude Sharpe", c_sharpes),
            "max_dd_pct": summarize("Claude max DD %", c_dds),
            "alpha_vs_btc": summarize("Claude alpha vs BTC", c_alpha),
        }

        t_return = t_stat(c_returns, h_returns)
        t_sharpe = t_stat(c_sharpes, h_sharpes)

    # ---- Save aggregated batch result --------------------------------------
    batch_payload = {
        "timestamp": timestamp,
        "config": {
            "n_seeds": N_SEEDS,
            "starting_capital": STARTING_CAPITAL,
            "duration_days": DURATION_DAYS,
            "decision_interval": DECISION_INTERVAL,
            "use_real_data": USE_REAL_DATA,
            "use_ai": USE_AI,
        },
        "heuristic_stats": h_stats,
        "claude_stats": c_stats,
        "t_stats": {
            "return": round(t_return, 3),
            "sharpe": round(t_sharpe, 3),
        },
        "heuristic_runs": heuristic_results,
        "claude_runs": claude_results,
    }
    out_path = OUTPUT_DIR / f"batch_{timestamp}.json"
    out_path.write_text(json.dumps(batch_payload, indent=2, default=str))

    # ---- Print human summary -----------------------------------------------
    banner("BATCH RESULT SUMMARY")
    print(f"\n  Heuristic (N={len(heuristic_results)}):")
    print(f"    return:    mean={h_stats['return_pct']['mean']:+.2f}%  std={h_stats['return_pct']['std']:.2f}  range=[{h_stats['return_pct']['min']:+.2f}, {h_stats['return_pct']['max']:+.2f}]")
    print(f"    Sharpe:    mean={h_stats['sharpe']['mean']:+.3f}  std={h_stats['sharpe']['std']:.3f}")
    print(f"    Max DD:    mean={h_stats['max_dd_pct']['mean']:.2f}%")
    print(f"    Alpha BTC: mean={h_stats['alpha_vs_btc']['mean']:+.2f}%")

    if c_stats:
        print(f"\n  Claude (N={len(claude_results)}):")
        print(f"    return:    mean={c_stats['return_pct']['mean']:+.2f}%  std={c_stats['return_pct']['std']:.2f}  range=[{c_stats['return_pct']['min']:+.2f}, {c_stats['return_pct']['max']:+.2f}]")
        print(f"    Sharpe:    mean={c_stats['sharpe']['mean']:+.3f}  std={c_stats['sharpe']['std']:.3f}")
        print(f"    Max DD:    mean={c_stats['max_dd_pct']['mean']:.2f}%")
        print(f"    Alpha BTC: mean={c_stats['alpha_vs_btc']['mean']:+.2f}%")

        delta_return = c_stats["return_pct"]["mean"] - h_stats["return_pct"]["mean"]
        delta_sharpe = c_stats["sharpe"]["mean"] - h_stats["sharpe"]["mean"]

        print(f"\n  CLAUDE EDGE:")
        print(f"    Return delta:  {delta_return:+.2f} pp  (t={t_return:+.2f})")
        print(f"    Sharpe delta:  {delta_sharpe:+.3f}     (t={t_sharpe:+.2f})")

        if abs(t_return) >= 2.0:
            verdict = "STATISTICALLY SIGNIFICANT" if delta_return > 0 else "SIGNIFICANTLY WORSE"
            print(f"    Verdict:       {verdict} (|t| >= 2.0, ~p < 0.05)")
        else:
            print(f"    Verdict:       INCONCLUSIVE (|t| < 2.0, can't reject null)")

    # ---- Write GitHub Actions Markdown summary -----------------------------
    gh_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if gh_summary:
        with open(gh_summary, "a") as f:
            f.write(f"# Batch Experiment — N={N_SEEDS} seeds\n\n")
            f.write(f"**Config:** capital=${STARTING_CAPITAL}, days={DURATION_DAYS}, interval={DECISION_INTERVAL}h, real_data={USE_REAL_DATA}\n\n")
            f.write("## Heuristic baseline\n\n")
            f.write("| Metric | Mean | Std | Min | Max |\n|---|---:|---:|---:|---:|\n")
            for key, label in [("return_pct", "Total return %"), ("sharpe", "Sharpe"), ("max_dd_pct", "Max DD %"), ("alpha_vs_btc", "Alpha vs BTC %")]:
                s = h_stats[key]
                f.write(f"| {label} | {s['mean']:+.3f} | {s['std']:.3f} | {s['min']:+.3f} | {s['max']:+.3f} |\n")

            if c_stats:
                f.write("\n## Claude Opus 4.6\n\n")
                f.write("| Metric | Mean | Std | Min | Max |\n|---|---:|---:|---:|---:|\n")
                for key, label in [("return_pct", "Total return %"), ("sharpe", "Sharpe"), ("max_dd_pct", "Max DD %"), ("alpha_vs_btc", "Alpha vs BTC %")]:
                    s = c_stats[key]
                    f.write(f"| {label} | {s['mean']:+.3f} | {s['std']:.3f} | {s['min']:+.3f} | {s['max']:+.3f} |\n")

                delta_return = c_stats["return_pct"]["mean"] - h_stats["return_pct"]["mean"]
                delta_sharpe = c_stats["sharpe"]["mean"] - h_stats["sharpe"]["mean"]
                f.write("\n## Statistical comparison (Claude vs Heuristic)\n\n")
                f.write(f"- **Return delta:** {delta_return:+.2f} pp (t = {t_return:+.2f})\n")
                f.write(f"- **Sharpe delta:** {delta_sharpe:+.3f} (t = {t_sharpe:+.2f})\n")
                if abs(t_return) >= 2.0:
                    if delta_return > 0:
                        f.write("- **Verdict:** Claude significantly beat the heuristic (|t| >= 2.0, ~p < 0.05)\n")
                    else:
                        f.write("- **Verdict:** Claude significantly underperformed the heuristic (|t| >= 2.0, ~p < 0.05)\n")
                else:
                    f.write("- **Verdict:** Inconclusive — N too small or effect too noisy to call (|t| < 2.0)\n")

    print(f"\n  Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
