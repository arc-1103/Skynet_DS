# Demo layer

This folder is **not part of the core product**. `src/cli.py` (the CLI)
and `src/api/webhooks.py` (the webhook receiver) are the actual product
and have no dependency on anything in here. Everything in `demo/` exists
to answer one evaluator question - "where is the UI?" - for three
different audiences, by pointing at tools that already have a UI instead
of building one from scratch.

Install its extra dependencies separately from the core project:

```
pip install -r requirements.txt
pip install -r demo/requirements-demo.txt
```

## 1. GitHub itself, as the PR-review UI (`github_pr_comment.py`)

Takes the `FinalPRReview` produced by `src.agents.multi_agent_graph` and
posts it as a real GitHub review on a real PR:

- `build_summary_markdown()` - a top-level comment with status/severity
  badges and a findings table collapsed behind `<details>`.
- `build_inline_comment()` - one inline-anchored comment per finding that
  has a line number, with a labeled suggested-fix blockquote.
- `post_review_to_github()` - posts both in one `pull.create_review()`
  call. Defaults to `event="COMMENT"` - it will never silently
  `REQUEST_CHANGES`/`APPROVE` unless you pass that explicitly.

This needs a `GITHUB_TOKEN` with write access to the target repo and
posts a real, visible comment - only point it at a repo you intend to
demo against.

## 2. Evidently's built-in dashboard (`evidently_dashboard.py`)

Runs Evidently's `DataDriftPreset` (K-S test, PSI, distribution curves,
p-values) over the same reference/current data pairs
`src.drift.pr_drift.check_pr_drift` already compares, and writes a
standalone, interactive HTML file - open it directly in a browser, no
server needed. The CLI's actual pass/fail decision still comes from
`src/drift/` (numpy/scipy only); this is a second, presentation-only
renderer over the same inputs.

## 3. Langfuse trace UI (`langfuse_tracing.py`)

Attaches `langfuse.langchain.CallbackHandler` to the compiled LangGraph
multi-agent graph, so the Langfuse UI shows the four specialist agents
(`security`, `architecture`, `ml_debugger`, `drift`) executing
concurrently, with real timestamps and latency.

**Honest gap**: every specialist agent is currently deterministic
(regex/statistics, matching this codebase's "no LLM call without a
configured key" convention - see `Execution.md`). The trace will show
real concurrent node execution, but not the token-count/Groq-vs-Gemini
model breakdown described in the pitch - that only appears once a node
makes an actual LLM call that Langfuse can see. Wiring a real LLM call
into one agent (e.g. `security_agent` via `ChatGroq` or
`ChatGoogleGenerativeAI`) is the natural next step if that half of the
demo matters for the pitch.

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` (if
self-hosting instead of Langfuse Cloud) before running - without them the
graph still runs correctly, it just isn't traced anywhere.

## Running all three together

```
python -m demo.run_demo owner/repo 42        # prints the review + writes Evidently dashboards
python -m demo.run_demo owner/repo 42 --post # also posts the review to the real PR
```

## Tests

`demo/tests/` is a separate pytest rootdir from the core `tests/` suite
(its own `conftest.py`) so the demo dependencies never leak into the
core test run:

```
pytest demo/tests/
```
