"""Post contest results to a GitHub Issue (auto-creates the tracking issue).

Used by GitHub Actions after each scheduled contest run. The workflow gives
us GITHUB_TOKEN automatically, so we use the GitHub REST API directly via
httpx (already a dependency) — no `gh` CLI needed.

The flow:
  1. Find an issue labeled "contest-tracking" in the current repo
  2. If it doesn't exist, create it (one-time)
  3. Append the latest contest result as a comment
  4. Update the issue body with a rolling summary table

This means you get one persistent issue that accumulates all contest runs
as comments, with a header that shows aggregate stats. Zero manual setup —
the workflow handles everything.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent

GH_API = "https://api.github.com"
TRACKING_LABEL = "contest-tracking"
ISSUE_TITLE_PREFIX = "Auto-Investment Contest Results"


def gh_headers(token: str) -> dict:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def find_or_create_issue(repo: str, token: str) -> int:
    """Return the issue number for the tracking issue, creating it if needed."""
    headers = gh_headers(token)
    with httpx.Client(timeout=30.0, headers=headers) as client:
        # Search for existing tracking issue
        r = client.get(
            f"{GH_API}/repos/{repo}/issues",
            params={"labels": TRACKING_LABEL, "state": "open", "per_page": 1},
        )
        r.raise_for_status()
        issues = r.json()
        if issues:
            return issues[0]["number"]

        # Ensure label exists
        try:
            client.post(
                f"{GH_API}/repos/{repo}/labels",
                json={"name": TRACKING_LABEL, "color": "0e8a16", "description": "Automated contest result tracking"},
            )
        except Exception:
            pass

        # Create the issue
        body = (
            "This issue is auto-maintained by the GitHub Actions contest workflow. "
            "Each contest run adds a new comment with the result. The body below "
            "is regenerated on every run with aggregate stats.\n\n"
            "_No manual edits — they will be overwritten._"
        )
        r = client.post(
            f"{GH_API}/repos/{repo}/issues",
            json={
                "title": f"{ISSUE_TITLE_PREFIX} (auto-tracking)",
                "body": body,
                "labels": [TRACKING_LABEL],
            },
        )
        r.raise_for_status()
        return r.json()["number"]


def post_comment(repo: str, token: str, issue_number: int, body: str) -> None:
    headers = gh_headers(token)
    with httpx.Client(timeout=30.0, headers=headers) as client:
        r = client.post(
            f"{GH_API}/repos/{repo}/issues/{issue_number}/comments",
            json={"body": body},
        )
        r.raise_for_status()


def update_issue_body(repo: str, token: str, issue_number: int, body: str) -> None:
    headers = gh_headers(token)
    with httpx.Client(timeout=30.0, headers=headers) as client:
        r = client.patch(
            f"{GH_API}/repos/{repo}/issues/{issue_number}",
            json={"body": body},
        )
        r.raise_for_status()


def build_comment(result: dict, mode: str) -> str:
    """Build a Markdown comment from a contest result dict."""
    rid = result.get("starting_capital", 0)
    final = result.get("final_equity", 0)
    ret = result.get("total_return_pct", 0)
    bench = result.get("benchmark_return_pct", 0)
    alpha = result.get("alpha_vs_benchmark", 0)
    sharpe = result.get("sharpe", 0)
    dd = result.get("max_drawdown_pct", 0)
    n_dec = result.get("n_decisions", 0)
    n_tr = result.get("n_trades", 0)
    n_liq = result.get("n_liquidations", 0)
    fees = result.get("fees_paid", 0)
    funding = result.get("funding_paid", 0)

    arrow = "📈" if ret > 0 else "📉"
    alpha_arrow = "🟢" if alpha > 0 else "🔴"

    return f"""## {arrow} Contest run — `{mode}` — {datetime.utcnow().isoformat()}Z

| Metric | Value |
|---|---:|
| Starting capital | ${rid:.2f} |
| **Final equity** | **${final:.2f}** |
| Total return | {ret:+.2f}% |
| BTC HODL benchmark | {bench:+.2f}% |
| {alpha_arrow} Alpha vs BTC | **{alpha:+.2f}%** |
| Sharpe ratio | {sharpe:+.3f} |
| Max drawdown | {dd:.2f}% |
| Decisions made | {n_dec} |
| Trades executed | {n_tr} |
| Liquidations | {n_liq} |
| Fees paid | ${fees:.2f} |
| Funding paid | ${funding:+.2f} |

<details><summary>Full result JSON (expand)</summary>

```json
{json.dumps({k: v for k, v in result.items() if k not in ('equity_curve', 'trade_log', 'decision_log')}, indent=2, default=str)}
```

Equity curve, trade log, and decision log are in the workflow artifact.
</details>
"""


def build_issue_body(history: list[dict]) -> str:
    """Aggregate stats summary that lives in the issue body."""
    if not history:
        return "No runs yet."

    n = len(history)
    by_mode: dict[str, list[dict]] = {}
    for e in history:
        by_mode.setdefault(e.get("mode", "?"), []).append(e)

    lines = [
        "# Auto-Investment Contest Tracker",
        "",
        f"**Total runs tracked:** {n}",
        f"**Last update:** {datetime.utcnow().isoformat()}Z",
        "",
        "## Aggregate by mode",
        "",
        "| Mode | N | Mean return | Mean Sharpe | Win vs BTC | Mean MaxDD |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for mode, entries in sorted(by_mode.items()):
        if not entries:
            continue
        nm = len(entries)
        mean_ret = sum(e.get("total_return_pct", 0) for e in entries) / nm
        mean_sharpe = sum(e.get("sharpe", 0) for e in entries) / nm
        wins = sum(
            1 for e in entries
            if e.get("total_return_pct", 0) > e.get("benchmark_return_pct", 0)
        )
        wr = wins / nm * 100
        mean_dd = sum(e.get("max_drawdown_pct", 0) for e in entries) / nm
        lines.append(
            f"| `{mode}` | {nm} | {mean_ret:+.2f}% | {mean_sharpe:+.2f} | "
            f"{wr:.0f}% | {mean_dd:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Recent runs (last 5)",
            "",
            "| Date | Mode | Final | Return | Sharpe | vs BTC |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for e in list(reversed(history))[:5]:
        date = e.get("timestamp", "?")[:10]
        lines.append(
            f"| {date} | `{e.get('mode', '?')}` | "
            f"${e.get('final_equity', 0):.2f} | "
            f"{e.get('total_return_pct', 0):+.2f}% | "
            f"{e.get('sharpe', 0):+.2f} | "
            f"{e.get('alpha_vs_benchmark', 0):+.2f}% |"
        )

    lines.extend(
        [
            "",
            "_This issue is auto-maintained. Click any comment for the full result of one run._",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        print("[ERROR] GITHUB_REPOSITORY and GITHUB_TOKEN must be set")
        print("        These are provided automatically inside GitHub Actions")
        return 1

    contest_dir = REPO_ROOT / "contest_results"
    if not contest_dir.exists():
        print(f"No {contest_dir}; nothing to post")
        return 0

    issue_number = find_or_create_issue(repo, token)
    print(f"Tracking issue: #{issue_number}")

    posted = 0
    for fname in sorted(contest_dir.glob("*.json")):
        mode = "claude" if "claude" in fname.name else "heuristic"
        try:
            data = json.loads(fname.read_text())
        except Exception as exc:
            print(f"  Skipping {fname.name}: {exc}")
            continue
        comment = build_comment(data, mode)
        post_comment(repo, token, issue_number, comment)
        posted += 1
        print(f"  Posted {mode} result from {fname.name}")

    # Update issue body with the latest aggregate
    try:
        from auto_track import load_history  # noqa: PLC0415

        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        history = load_history()
        body = build_issue_body(history)
        update_issue_body(repo, token, issue_number, body)
        print(f"Updated issue body with {len(history)} historical runs")
    except Exception as exc:
        print(f"Couldn't update issue body: {exc}")

    print(f"Done — posted {posted} comments to issue #{issue_number}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
