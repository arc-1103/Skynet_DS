# demo/run_demo.py
"""Single entry point tying the three demo surfaces together against a
real GitHub PR: run the multi-agent graph (traced via Langfuse), print
the result, optionally post it back to GitHub as a review, and generate
an Evidently dashboard for any data file the PR touched.

    python -m demo.run_demo owner/repo 42 --post

This is separate from src/cli.py on purpose - src/cli.py is the actual
product's CLI entry point (mlcop debug --run-id ...) and has no
dependency on evidently/langfuse/langchain. This script only exists to
make the three demo pieces easy to run together for a presentation; the
CLI is unaffected whether or not this file exists.

--post actually writes a review comment to a real pull request - a
visible, external side effect - so it is off by default. Only pass it
against a repo/PR you intend to demo against."""

from __future__ import annotations

import argparse
import sys

from demo.evidently_dashboard import dashboards_for_pr
from demo.github_pr_comment import post_review_to_github
from demo.langfuse_tracing import langfuse_is_configured, run_traced_pr_review
from src.core.config import settings
from src.github_integration.pr_client import fetch_pr
from src.github_integration.repo_client import clone_or_checkout


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the multi-agent PR review demo end-to-end")
    parser.add_argument("repo_full_name", help="e.g. acme/fraud-clf")
    parser.add_argument("pr_number", type=int)
    parser.add_argument("--post", action="store_true", help="Post the review back to the real PR on GitHub")
    parser.add_argument(
        "--event",
        default="COMMENT",
        choices=["COMMENT", "APPROVE", "REQUEST_CHANGES"],
        help="Review event to post when --post is set (default: COMMENT)",
    )
    args = parser.parse_args()

    if not settings.github_token:
        print("GITHUB_TOKEN is not set - fetching a public PR anonymously (rate-limited, read-only).")

    pr = fetch_pr(args.repo_full_name, args.pr_number, settings.github_token)
    # extra_refs: on an existing cached clone, only `ref` gets fetched - if
    # the base branch has moved on since the clone was made, pr.base_sha
    # (needed by `git show <base_sha>:<path>` for the drift/dashboard
    # comparisons below) may not be present locally without this. Same
    # fix as src/worker/jobs.py's call.
    repo_path = clone_or_checkout(
        args.repo_full_name,
        pr.commit_sha,
        settings.github_token,
        extra_refs=(pr.base_sha,) if pr.base_sha else (),
    )

    print(f"Running multi-agent review for {args.repo_full_name}#{args.pr_number}...")
    if not langfuse_is_configured():
        print("(Langfuse not configured - set LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY to see the trace in its UI)")
    final_state = run_traced_pr_review(pr, repo_path)
    review = final_state["final_review"]

    print(review.markdown_summary)

    dashboards = dashboards_for_pr(pr, repo_path)
    for path in dashboards:
        print(f"Evidently dashboard written to: {path}")

    if args.post:
        if not settings.github_token:
            print("Cannot post: GITHUB_TOKEN is not set.", file=sys.stderr)
            return 1
        post_review_to_github(
            args.repo_full_name, args.pr_number, pr.title, review, settings.github_token, event=args.event
        )
        print(f"Posted review to {args.repo_full_name}#{args.pr_number} (event={args.event}).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
