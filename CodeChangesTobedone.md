# Code Changes To Be Done

Findings from `/code-review high` on the working-tree diff for the new "multi-agent PR review + data-drift detection" feature (src/api/webhooks.py, src/core/config.py, src/schemas/*, src/worker/jobs.py, src/agents/*.py, src/drift/*.py, src/github_integration/*.py, and corresponding tests).

## Correctness Findings (ranked most severe first)

### 1. Crash on null severity from LLM — ✅ Fixed
- **File:** `src/agents/security_agent.py:181` (duplicated in `src/agents/architecture_agent.py:67` and `src/agents/ml_debugger_agent.py:109`)
- **Issue:** `Severity(item.get("severity", "MEDIUM").upper())` crashes with `AttributeError` when the LLM returns an explicit `null` severity, escaping the narrow `except (KeyError, ValueError)` meant to skip malformed findings.
- **Failure scenario:** Groq returns `{"severity": null, ...}`. `dict.get` returns `None` (key is present, so default isn't used), then `None.upper()` raises `AttributeError`, uncaught by `except (KeyError, ValueError): continue` — crashing the specialist node and, once wired in, the entire graph run.
- **Fix:** Changed `item.get("severity", "MEDIUM").upper()` to `(item.get("severity") or "MEDIUM").upper()` in all three files, so an explicit `null` also falls back to `"MEDIUM"`.

### 2. Stale clone never fetches PR's base ref — ✅ Fixed
- **File:** `src/github_integration/repo_client.py:29`
- **Issue:** `clone_or_checkout` only fetches the ref it's given (`head_sha`); on an existing clone it never fetches the PR's base ref/branch.
- **Failure scenario:** After the first PR review for a repo, the clone exists. A later PR's `base_sha` is often not yet present if `main` advanced. `src/drift/pr_drift.py:74`'s `git show base_sha:file` fails with `CalledProcessError`, caught per-file and logged as a skip — the job "succeeds" having silently produced zero drift reports.
- **Fix:** `clone_or_checkout` now takes an `extra_refs` tuple that gets fetched alongside `ref` (but not checked out); `jobs.py` passes `pr.base_sha` as an extra ref so `git show base_sha:file` always resolves against a locally-fetched commit.

### 3. head_sha / pr.commit_sha can diverge — ✅ Fixed
- **File:** `src/worker/jobs.py:79`
- **Issue:** `clone_or_checkout` checks out the webhook-provided `head_sha`, but `pr_drift.py:75` diffs against `pr.commit_sha` from a separately, freshly re-fetched `PRPayload` — these can diverge.
- **Failure scenario:** If the PR receives a new push between webhook delivery and job execution, `fetch_pr`'s live API call returns a newer `pr.commit_sha` than the `head_sha` the job was scheduled with. The clone never fetched that commit, so `git show <pr.commit_sha>:file` fails and is silently skipped, same as finding #2.
- **Fix:** `jobs.py` now checks out `pr.commit_sha` (the fresh value from `fetch_pr`) instead of the webhook's `head_sha`, so the working tree and the drift comparison always agree on which commit is "head".

### 4. Unhandled ValueError for missing base_sha — ✅ Fixed
- **File:** `src/worker/jobs.py:80`
- **Issue:** `check_pr_drift(pr, repo_path)` has no try/except around the `ValueError` it raises when `pr.base_sha` is falsy, unlike `drift_agent.py:26` which explicitly catches this exact case.
- **Failure scenario:** `PRPayload.base_sha` is `Optional`. If a payload reaches this path without `base_sha` (malformed/older webhook payload), `run_pr_code_review` crashes with an unhandled `ValueError` after already writing the cached PR JSON — turning a should-be-partial success into a hard job failure.
- **Fix:** Wrapped the `check_pr_drift` call in `jobs.py` in `try/except ValueError`, logging a warning and returning instead of crashing the job.

### 5. Graph docstring overstates failure isolation — ✅ Fixed
- **File:** `src/schemas/graph.py:42`
- **Issue:** `AgentState`'s docstring claims the per-field split means "one slow/failing branch can't clobber another's results," but `multi_agent_graph.py` wires branches with plain `add_node`/`add_edge` and no per-node exception handling.
- **Failure scenario:** An unexpected exception in any specialist agent (e.g. finding #1) propagates out of `app.invoke()` and kills the entire graph run before `synthesize` ever executes — zero partial review, not the isolation the docstring advertises. Currently latent since `run_pr_review` isn't yet called from `webhooks.py`/`jobs.py`.
- **Fix:** Added an `_isolate` wrapper in `multi_agent_graph.py` that catches any exception from a specialist node, logs it, and returns an empty result for that branch's output key plus an `errors` entry — all four branch nodes are now registered through it, so the docstring's claim actually holds.

### 6. Unguarded exceptions from dataset drift detection — ✅ Fixed
- **File:** `src/agents/drift_agent.py:26`
- **Issue:** `run_drift_agent` only catches `ValueError` from `check_pr_drift`, but `pr_drift.py`'s `detect_dataset_drift` call runs entirely outside `pr_drift`'s own try/except, so any other exception type propagates unguarded.
- **Failure scenario:** A data file that loads successfully but has an unexpected shape (zero rows, column type mismatch) causes `detect_dataset_drift` to raise a non-`ValueError` exception, uncaught anywhere, crashing the whole multi-agent graph run instead of producing an empty drift report for that file.
- **Fix:** `check_pr_drift` in `pr_drift.py` now wraps the `detect_dataset_drift` call in its own try/except, logging a warning and skipping that file on any exception, same pattern as the existing load-failure handling.

### 7. No per-repo locking on concurrent clones — ✅ Fixed
- **File:** `src/github_integration/repo_client.py:26`
- **Issue:** `clone_or_checkout`'s destination path is derived only from `repo_full_name` with no per-job locking, so concurrent PR-review jobs for the same repo share one on-disk working tree.
- **Failure scenario:** Two `run_pr_code_review` jobs run close together for the same repo (an "opened" and "synchronize" event, or two open PRs). Both concurrently `git fetch`/`checkout`/`reset --hard` the same directory, corrupting the working tree or producing spurious git errors.
- **Fix:** Added a cross-process `_repo_lock` context manager (`O_CREAT|O_EXCL` lock file next to the repo dir, with stale-lock reclamation) and wrapped the whole clone/fetch/checkout body in it — no new dependency, works on Windows and POSIX.

### 8. Falsy-zero bug in line number rendering — ✅ Fixed
- **File:** `src/agents/synthesizer.py:40`
- **Issue:** The ternary `f"{f.file_path}:{f.line_number}" if f.line_number else f.file_path` treats a legitimate `line_number` of `0` the same as missing/`None`.
- **Failure scenario:** An LLM-backed finding returns `line_number: 0` instead of a real line or null. The synthesized markdown silently drops the line reference and shows just the bare file path, misrepresenting a line-anchored finding as file-level.
- **Fix:** Changed the condition to `if f.line_number is not None else f.file_path`.

### 9. "reopened" PR action silently ignored — ✅ Fixed
- **File:** `src/api/webhooks.py:92`
- **Issue:** The `pull_request` action allow-list only includes `"opened"` and `"synchronize"`; `"reopened"` is silently ignored despite being the same conceptual trigger as opened.
- **Failure scenario:** A closed PR is reopened with no new commits; GitHub sends `action='reopened'`. The webhook returns `status=ignored` and no code review or drift check ever runs, even though a human would expect review to (re)trigger.
- **Fix:** Added `"reopened"` to the allow-list.

### 10. False positive: unbatched training-loop detection — ✅ Fixed
- **File:** `src/agents/ml_debugger_agent.py:178`
- **Issue:** "Model `.fit()` called inside a loop" is flagged whenever a changed file's added lines contain any for-loop AND any `.fit()` call anywhere, with no check that the `fit()` call is actually nested inside that loop.
- **Failure scenario:** A diff adds an unrelated `for batch in dataloader: log(batch)` loop plus a separate, correct `model.fit(X, y, epochs=10)` call elsewhere in the same diff — `has_loop` and `has_fit` are both `True` independently, raising a false "unbatched training in a loop" finding against fine code.
- **Fix:** Added `_fit_call_inside_loop`, which walks added lines from each for-loop header and only matches a `.fit()` call whose indentation is deeper than the loop's (stopping once indentation returns to the loop's level or shallower), replacing the independent `has_loop`/`has_fit` check.

All fixes verified against the existing test suite: `pytest tests/` — 281 passed, 0 failed.

## Lower-Severity Cleanup Candidates — ✅ all resolved

> Cut from the original pass to stay within a correctness-first cap; worked through in the "Cleanup pass" section at the end of this file.

- `_llm_findings`-style LLM-response-parsing loop duplicated verbatim across `security_agent.py:159-188`, `architecture_agent.py:47-74`, `ml_debugger_agent.py:87-116`.
- `_read_table`-style CSV/Parquet loader reimplemented inconsistently in `ml_debugger_tools.py:44`, `drift/detector.py:206`, `drift/root_cause.py:275`, `drift/pr_drift.py:43` (detector.py/root_cause.py silently default unknown suffixes to `read_csv`; ml_debugger_tools.py raises `ValueError` instead).
- Duplicated "optional corroborating CLI tool" pattern in `security_agent_tools.py:122/245` and `ml_debugger_tools.py:112`.
- Over-built `AgentNode` Protocol (`schemas/graph.py:65-76`) for a single Pylance nicety; half-applied `_STATIC_ANALYSIS_NODES` registry in `multi_agent_graph.py:42` that only covers 1 of 4 branches; dead `PRReviewState` TypedDict (`schemas/graph.py`) with no live callers.
- ~~`run_pr_review`/`build_multi_agent_graph` (`multi_agent_graph.py`) have no caller anywhere in `src/` yet — the whole multi-agent feature is not wired into `webhooks.py`/`jobs.py` in this diff.~~ ✅ Wired — see "Multi-agent graph wired into the webhook path" at the end of this file.
- Sequential, blocking per-file LLM calls in each specialist agent (no batching/async), and `clone_or_checkout` doing a full working-tree checkout when `pr_drift.py` only ever needs `git show` blobs.

## Findings from `/code-review xhigh` (ranked most severe first)

> Status: all resolved. #1 and #3 were fixed in the `/code-review ultra` pass (#5 and #2 there); #4–#7 in the follow-up pass, and #2 was closed as not-a-bug — all at the end of this file.

Scope: `git diff origin/main...HEAD` combined with `git diff HEAD` (local commits on main not yet pushed, plus uncommitted working-tree changes), including untracked callee files (`src/drift/pr_drift.py`, `src/github_integration/pr_client.py`/`repo_client.py`) called by the diffed `src/worker/jobs.py`.

### 1. Drift report JSON silently omits the verdict fields — ✅ Fixed (ultra #5)
- **File:** `src/schemas/drift.py:79`
- **Issue:** `DatasetDriftReport`'s `n_columns`/`n_drifted`/`share_drifted`/`dataset_drifted` are plain `@property` methods, not pydantic fields, so `model_dump_json()` silently omits them.
- **Failure scenario:** `src/worker/jobs.py` writes each drift report to disk via `report.model_dump_json(indent=2)` so a future review-agent pipeline can read it (per the function's own docstring). The persisted JSON contains only `{columns, share_threshold, label}` — the actual drift verdict (`n_drifted`, `share_drifted`, `dataset_drifted`) is missing entirely, even though the adjacent log line in `jobs.py` prints those exact values. Any consumer reading the cached file instead of the live Python object gets no drift verdict at all.

### 2. `.gitignore` now silently ignores the real design docs — ❌ Closed, not a bug
- **File:** `.gitignore:7`
- **Issue:** Removing the trailing slash from `Architecture.md/`, `BluePrint.md/`, `PRD.md/`, `Execution.md/`, `Workflow.md/` changes them from directory-only patterns to patterns that also match plain files.
- **Failure scenario:** `Architecture.md`, `BluePrint.md`, `PRD.md`, `Execution.md`, and `Workflow.md` exist as plain files at the repo root (verified via `git check-ignore -v`, all 5 report as ignored by these exact lines) and are referenced throughout the new code's docstrings. They no longer show up in `git status` at all, so any future edit to these design docs is never committed and is invisible to anyone cloning the repo.

### 3. Blocking git/HTTP calls run directly on the shared event loop — ✅ Fixed (ultra #2)
- **File:** `src/worker/jobs.py:65`
- **Issue:** `async def run_pr_code_review` calls synchronous, blocking `fetch_pr` (PyGithub HTTP) and `clone_or_checkout` (subprocess git clone/fetch) directly, with no executor/thread offload.
- **Failure scenario:** The job is scheduled via `DelayedTaskQueue.schedule` → `asyncio.create_task` on the same event loop uvicorn/FastAPI uses to serve `/webhooks/github` (a single process, per `webhooks.py`'s own "ack fast, thin layer" docstring). A slow GitHub API call or a large git clone triggered by one PR's drift check blocks that one event loop, stalling every other concurrent webhook request and every other scheduled delayed task (e.g. `run_deployment_regression_check`'s `asyncio.sleep` timer) until it finishes.

### 4. `clone_or_checkout`/`fetch_pr` errors uncaught, masked by blanket handler — ✅ Fixed
- **File:** `src/worker/jobs.py:79`
- **Issue:** `clone_or_checkout` (and `fetch_pr`) are not wrapped in any try/except, unlike `check_pr_drift` a few lines later which specifically catches `ValueError`.
- **Failure scenario:** A transient git/network failure (`subprocess.CalledProcessError`) or a missing git executable (`FileNotFoundError`, raised by `pr_drift.py`'s `load_at_ref` and caught only as `(CalledProcessError, ValueError)` internally, so `FileNotFoundError` still escapes) propagates out of `run_pr_code_review` uncaught. It's only masked by `dispatcher.py`'s blanket `except Exception: logger.exception("Delayed task failed")`, even though the PR metadata was already successfully fetched and cached moments earlier — operators get a generic failure log with no indication the PR cache actually succeeded and only drift-checking broke.

### 5. Drift-report filename slug doesn't sanitize backslashes — ✅ Fixed
- **File:** `src/worker/jobs.py:92`
- **Issue:** `slug = str(report.label).replace("/", "_")` only strips forward slashes before being used to build a filename via `out_path.with_name(...)`, leaving backslashes or other separator characters unsanitized.
- **Failure scenario:** On Windows (this repo's dev platform), `pathlib.PurePath.with_name()` raises `ValueError` if the new name contains the OS separator. Git can track a path containing a literal backslash byte (only `/` is git's own separator), so a PR touching such a data file produces a slug still containing `\`, crashing `with_name()` inside the loop; the exception is swallowed by `dispatcher.py`'s blanket handler and every remaining drift report for that PR silently fails to write.

### 6. Decorated large-class chunks drop the decorator line — ✅ Fixed
- **File:** `src/indexing/chunker.py:153`
- **Issue:** For a decorated large class (>1500 chars), `header_line` is taken from the inner `class_definition` node instead of the enclosing `decorated_definition`, so `class_header` drops the decorator line.
- **Failure scenario:** Every per-method chunk for that class gets `# Class Context: class Foo:` without its `@dataclass`/`@register_model`/etc. decorator, unlike (a) the whole-class chunk path for small classes, which includes the decorator, and (b) each member's own decorator, which is preserved. A decorated large class (e.g. an `@dataclass` config or a registered model) loses exactly the annotation a RAG/embedding search would need to identify it — existing tests don't exercise a decorated class, so this went unnoticed.

### 7. Nested functions no longer chunked as distinct symbols — ✅ Fixed
- **File:** `src/indexing/chunker.py:103`
- **Issue:** `PythonDSChunker` only walks `tree.root_node.children` plus one extra level into class bodies, so a function nested inside another function (a closure, e.g. `train_step` inside `train`) is no longer chunked as its own symbol — a real regression from the old `SemanticChunker`, which recursed to any depth.
- **Failure scenario:** A common DS pattern like `def train(): ... def train_step(x): ...` has `train_step`'s code silently absorbed into `train`'s single chunk instead of becoming its own `CodeChunk`. `CodeIndex.find_symbol('train_step')`/`.search('train_step')` can no longer resolve it as a distinct symbol, degrading retrieval for exactly the nested-training-loop code this chunker is designed to optimize for. The new test suite only added a regression test for nested classes, not nested functions, so this gap is untested.

## Additional Cleanup Candidates from `/code-review xhigh` — ✅ all resolved

> See the "Cleanup pass" section at the end of this file.

- `src/api/webhooks.py:65` — the new `pull_request` branch duplicates the exact validate/check-condition/queue.schedule/return shape of the `deployment_status` branch immediately above it, growing `github_webhook` into an ever-larger if/elif chain instead of a data-driven event router.
- `src/indexing/chunker.py:43` — `CodeChunk(...)` is constructed inline with near-identical boilerplate in four separate places (top-level script flush, function case, standalone-class case, and `_append_class_member_chunk`), differing only in `symbol_name`/`content`/`chunk_type`.
- `src/core/config.py:5` — `load_dotenv()` runs unconditionally at module import time, and `Settings`' fields call `os.environ.get(...)` once at class-body execution, freezing the process-wide settings singleton to whatever `.env`/env state existed at first import; this is why `tests/conftest.py`'s autouse fixture has to monkeypatch the already-built settings object via `dataclasses.replace` rather than simply setting env vars.

## Findings from `/code-review ultra` — all fixed

Scope: `git diff HEAD` plus the untracked new modules (`src/agents/*`, `src/drift/*`, `src/github_integration/*`, `src/schemas/{drift,root_cause}.py`). Every finding below is fixed and covered by a regression test; the suite went 281 → 315 passing.

This section supersedes two entries in the xhigh list above: xhigh #1 (drift verdict omitted from JSON) is fixed as #5 below, and xhigh #3 (blocking calls on the event loop) as #2 below.

### 1. `eval()` regex fires on `model.eval()` — ✅ Fixed
- **File:** `src/agents/security_agent.py:52`
- **Issue:** The word boundary in the eval/exec patterns matches after a `.`, so every attribute call was treated as a bare builtin call.
- **Failure scenario:** A diff adding `model.eval()` or `df.eval("a + b")` yields HIGH "Use of eval() — code-injection vector" findings. Because `synthesizer` computes `overall_passed` as "no HIGH/CRITICAL finding", any PR calling `model.eval()` — the exact pattern `ml_debugger_agent` itself hunts for — is auto-marked CHANGES REQUESTED. The AST visitor in `security_agent_tools.py:171` already got this right via `isinstance(node.func, ast.Name)`.
- **Fix:** Both patterns now use a negative lookbehind for a word character or dot, so only a bare `eval(`/`exec(` matches.

### 2. Blocking I/O and CPU work run on the FastAPI event loop — ✅ Fixed
- **File:** `src/worker/jobs.py:52`
- **Issue:** `queue.schedule` hands these coroutines to `asyncio.create_task` on the API server's own loop, and neither job body contained a single `await` — all of the work is synchronous.
- **Failure scenario:** `fetch_pr` (PyGithub/requests), `clone_or_checkout` (git via `subprocess.run`, plus a sleep-poll on the repo lock) and the drift detectors (pandas + RandomForest 5-fold CV + 200 MMD permutations per column) each block the loop. One PR against a large repo holds it for minutes, so every concurrent webhook delivery exceeds GitHub's 10s timeout and is retried or dropped.
- **Fix:** Both jobs are split into a sync `_run_*` body and a thin `async def` that does `await asyncio.to_thread(...)`.

### 3. Webhook-supplied `head_sha` reaches a filesystem path unvalidated — ✅ Fixed
- **File:** `src/worker/jobs.py:69`, `src/schemas/webhook.py:16`
- **Issue:** `out_path = out_dir / f"pr_{pr_number}_{head_sha}.json"` interpolates `pull_request.head.sha` straight from the payload. `repo_full_name` is neutralized with `.replace("/", "_")`; `head_sha` was not.
- **Failure scenario:** A payload with `"sha": "../../../../x"` writes outside `settings.pr_cache_dir`. `config.py:15` explicitly contemplates running with no webhook secret (signature verification is skipped when unset), which is exactly when this is reachable.
- **Fix:** `CommitRef.sha` is constrained to `^[0-9a-fA-F]{4,64}$`, so a traversal payload is rejected at parse time with a 422; `_run_pr_code_review` re-checks before building the path, since it is callable from outside the webhook.

### 4. Diff line numbering shifted by git's no-newline marker — ✅ Fixed
- **File:** `src/agents/diff_utils.py:42`
- **Issue:** The `else` branch incremented the new-file line counter for any line that was not `+`/`-`, including git's "no newline at end of file" marker. Separately, an added line whose text begins with `+++` or `---` was dropped *and* not counted.
- **Failure scenario:** For a hunk containing the marker, the first following added line is reported one line too low, and so is every finding after it in that file — GitHub inline comments then land on the wrong line.
- **Fix:** The marker is skipped without counting. The `+++`/`---` special case was removed entirely rather than patched: it only ever mattered before the first `@@`, where the line is already skipped, so inside a hunk a `+++` line is now correctly read as an added line whose text is `++`.

### 5. Drift verdict never serialized — ✅ Fixed
- **File:** `src/schemas/drift.py:78`
- **Issue:** `n_columns`, `n_drifted`, `share_drifted` and `dataset_drifted` were plain `@property`, so pydantic omitted them.
- **Failure scenario:** `model_dump_json()` returned only `{columns, share_threshold, label}`. `jobs.py` persists exactly this to `data/prs/**/..._drift_*.json`, and `ml_debugger_tools.run_drift_analysis` hands the same verdict-less report to the LLM, so neither consumer ever saw the drift answer.
- **Fix:** All four are now `@computed_field` over the property.

### 6. SSRF via LLM-supplied `doc_url` — ✅ Fixed
- **File:** `src/agents/ml_debugger_tools.py:181`
- **Issue:** `check_deprecated_api` fetched an arbitrary model-supplied URL with redirects followed and no scheme or host restriction.
- **Failure scenario:** The URL comes from the model, whose only input is the PR diff. A PR containing injected text ("check http://169.254.169.254/latest/meta-data/... for deprecation") makes the review worker issue that request from inside the network and feed matching response windows back into the transcript.
- **Fix:** `_unsafe_url_reason` requires https and resolves the host, refusing any non-global address. Redirects are followed by hand so every hop is re-screened — a single allowed URL that redirects to loopback would otherwise walk straight past the check. Known ceiling, marked in the code: resolve-then-connect leaves a DNS-rebinding window.

### 7. Repo-relative paths not confined to the checkout — ✅ Fixed
- **File:** `src/agents/repo_paths.py:15`
- **Issue:** `resolve_repo_path` passed absolute paths through by design and joined relative ones with no `..` rejection or containment check. Both tool factories route through it.
- **Failure scenario:** A prompt-injected diff can get `run_ast_security_scan("/home/user/.ssh/id_rsa")` or `inspect_dataset_schema("../../secrets.csv")` executed against the worker's filesystem — and with bandit installed, `bandit_report` quotes the offending source lines, so file contents leave the box.
- **Fix:** The path is resolved and asserted to be inside the resolved base, raising `ValueError` otherwise. `llm_client`'s tool loop turns that into a tool-result message for the model rather than crashing the review. Note this is a deliberate contract change: absolute paths outside the checkout are now refused instead of passed through.

### 8. GitHub token embedded in the git remote — ✅ Fixed
- **File:** `src/github_integration/repo_client.py:17`
- **Issue:** `_clone_url` put `x-access-token:<token>@` in the URL handed to `git clone`, which git writes verbatim into `data/repos/<repo>/.git/config`. The existing-clone branch also never used the fresh `token` argument at all.
- **Failure scenario:** The PAT sits in plaintext on disk and in any `git remote -v` or error output; and after a token rotation every already-cloned repo keeps authenticating with the stale embedded credential and fails until the directory is deleted.
- **Fix:** Credentials are passed per invocation as `git -c http.extraHeader=Authorization: Basic ...` on the clone and fetch commands, so nothing is persisted and a rotated token reaches existing clones. Known ceiling, marked in the code: the token is visible in this process's argv for the duration of the call.

### 9. Default `groq_model` is not a real model id — ✅ Fixed
- **File:** `src/core/config.py:26`
- **Issue:** `"qwen/qwen3.8-27b"` (also in `.env.example` and `security_agent.py`'s docstring) matches no Groq model — Qwen3 has no 3.8 version and no 27B size.
- **Failure scenario:** A deployment with a valid `GROQ_API_KEY` gets a 404 on every call; `llm_client` catches everything and returns `None`, so the system silently degrades to regex-only review with only a `logger.warning`. Separately, `GROQ_MODEL=` left empty in a copied `.env` yields `""`, not the default, because `os.environ.get` only falls back on an absent key.
- **Fix:** Default is `qwen/qwen3-32b` in all three places, resolved with `or` instead of a `get()` default so an empty value also falls back.

### 10. Drift never affected the pass/fail verdict — ✅ Fixed
- **File:** `src/agents/synthesizer.py:69`
- **Issue:** `overall_passed` was computed from inline findings only, ignoring the drift branch, despite the module docstring saying this node decides pass/fail.
- **Failure scenario:** A PR where every data file is flagged `dataset_drifted=True` rendered `**Overall: PASSED**` while simultaneously carrying a CONFIG label from `_labels_for`.
- **Fix:** `drift_flagged` is folded into `overall_passed`.

### 11. Stale-lock reclaim could hand one working tree to two jobs — ✅ Fixed
- **File:** `src/github_integration/repo_client.py:22`
- **Issue:** `_repo_lock` reclaimed any lock file older than `stale_after` without checking whether the holder was still alive, and the `finally` block unlinked the lock file unconditionally.
- **Failure scenario:** A first clone of a large repo can legitimately exceed the timeout while still holding the lock. Job B reclaims it and runs `git checkout`/`reset --hard` against the tree job A is mid-clone on; then job A's `finally` deletes the lock file job B now owns.
- **Fix:** The holder runs a 30s heartbeat thread touching the lock's mtime, so "stale" now means the holder stopped running rather than that its clone is slow (`stale_after` accordingly dropped to 120s). Each holder also writes a random token and only unlinks a lock still carrying its own. No pid-liveness check: `os.kill(pid, 0)` terminates the process on Windows.

### 12. `synthesize` was the one node not isolated — ✅ Fixed
- **File:** `src/agents/multi_agent_graph.py:112`
- **Issue:** Every branch was wrapped in `_isolate` but the fan-in was not.
- **Failure scenario:** Any raise inside `run_synthesizer_agent` — a `ReviewFinding` with an unexpected severity, or a missing `pr_payload` — propagates out of `app.invoke()` and discards all four branches' completed work.
- **Fix:** `_isolate` gained an `empty` argument (branches pass `[]`, synthesize passes `None` for `final_review`) and now wraps the fan-in too.

### 13. Explicit `columns` not validated against both frames — ✅ Fixed
- **File:** `src/drift/detector.py:159`
- **Issue:** An explicitly passed `columns` list skipped the shared-column intersection that the default path performs.
- **Failure scenario:** `columns` is LLM-supplied via `ml_debugger_tools.run_drift_analysis`. The model names a column it saw in the diff but that is not in the loaded file, and indexing it raises `KeyError`, so the tool returns `{"error": "KeyError..."}` instead of the intended `DriftInputError` message.
- **Fix:** An explicit `columns` is intersected with both frames, so an unknown name is skipped like any other unshared column.

## Multi-agent graph wired into the webhook path — ✅ Done

`run_pr_review`/`build_multi_agent_graph` previously had no caller in `src/` (noted in the cleanup list above), so the whole multi-agent feature was unreachable from the webhook. `_run_pr_code_review` now runs it: fetch the PR, cache it, clone **only** if the PR touches a tabular data file, then `run_pr_review(pr, repo_path)`, then persist the drift reports and the synthesized review as `pr_<n>_<sha>_review.json`.

- Drift moved into the graph rather than running twice: `jobs.py` no longer calls `check_pr_drift` itself, the drift branch does.
- The clone stays conditional, now for two reasons: drift is the one specialist that cannot work off the diff alone, and `repo_path` is also what lets the security and ml_debugger tool passes read real files. A code-only PR runs the graph with `repo_path=None` and skips only drift, instead of skipping the whole review.
- `state["errors"]` is logged per branch; a `final_review` of `None` (isolated synthesizer failure) logs an error rather than failing the job.

**Still open:** nothing is posted back to GitHub. `FinalPRReview.markdown_summary` and `inline_findings` are written to disk ready for a PR-comment step, which needs a decision on app identity and permissions first.

**Deferred at the time:** xhigh #5 (drift-report filename slug) was left alone in this pass as out of scope, and fixed in the follow-up pass below.

## Follow-up pass on the remaining xhigh findings

Suite: 315 → 321 passing. Statuses on the xhigh findings above are updated in place; the detail is here.

### xhigh #2 — ❌ Closed, not a bug
`.gitignore` ignoring `Architecture.md`, `BluePrint.md`, `PRD.md`, `Execution.md` and `Workflow.md` as plain files is deliberate: the same edit also changed `.env/` to `.env`, the commit is titled "editied .gitignore for markdown files", and none of the five have ever been tracked (`git ls-files` lists only `README.md`). The reviewer assumed they belonged in the repo. Confirmed with the repo owner and left as is.

### xhigh #4 — ✅ Fixed
- **File:** `src/worker/jobs.py`
- **Fix:** `clone_or_checkout` is wrapped in `try/except (subprocess.CalledProcessError, OSError, TimeoutError)` — that covers a transient git failure, a missing git executable (the `FileNotFoundError` that escapes `pr_drift.load_at_ref`'s narrower handler), and a `_repo_lock` timeout. On failure the job logs what happened and reviews with `repo_path=None`: the PR is already cached and the static-analysis branches work off the diff alone, so only drift is lost. `fetch_pr` is deliberately left to propagate — if it fails there is no PR to review, and the traceback names it.

### xhigh #5 — ✅ Fixed
- **File:** `src/worker/jobs.py`
- **Fix:** the drift-report filename slug is now `re.sub(r"[^A-Za-z0-9._-]+", "_", label)` instead of `.replace("/", "_")`. That covers the backslash `with_name()` raises on, and a colon or any other character Windows rejects in a filename. The dot stays in the allowed set, so existing cache filenames are unchanged.

### xhigh #6 — ✅ Fixed
- **File:** `src/indexing/chunker.py`
- **Fix:** the class header injected into each method chunk is built from `child.start_point[0]` (the `decorated_definition`, when there is one) through the `class X:` line, instead of from the inner `class_definition` alone. A decorated large class keeps its `@dataclass`/`@register_model` in every method chunk, matching what the whole-class path already did. The lines are joined with a space so the header stays a single `# Class Context:` comment.

### xhigh #7 — ✅ Fixed
- **File:** `src/indexing/chunker.py`
- **Fix:** new `_append_nested_function_chunks` walks a function body and emits a chunk per nested function, recursing to any depth, with a qualified `parent.child` symbol name and `chunk_type: "nested_function"`. `CodeIndex.find_symbol` already matches on the trailing name, so `find_symbol("train_step")` resolves `train.train_step`. It is called for top-level functions *and* for the methods of a split class — a closure inside a method was the same bug, and fixing only the flagged path would have left it.

**Still open after this pass:** the two cleanup-candidate sections above, and posting the review back to GitHub. Of the cleanup items, the one worth treating as a latent bug rather than duplication is `_read_table`: `drift/detector.py:211` and `drift/root_cause.py:275` silently `read_csv` an unknown suffix, while `agents/ml_debugger_tools.py:48` raises `ValueError` for the same input.

## Cleanup pass — both candidate lists

Suite: 321 → 339 passing. Every item from "Lower-Severity Cleanup Candidates" and "Additional Cleanup Candidates from `/code-review xhigh`" is resolved below, including the two I judged should *not* change.

### Duplicated LLM-response parsing across the three specialists — ✅ Done
- **Was:** `security_agent.py`, `architecture_agent.py` and `ml_debugger_agent.py` each held a verbatim copy of `_llm_findings`.
- **Now:** `src/agents/llm_findings.py` holds `parse_findings` (raw items → `ReviewFinding`, skipping anything it can't build) and `collect_llm_findings` (per-file prompt + call + parse). The only real difference between the three was the call itself, so that is the `ask` argument: architecture passes `call_llm_findings`, security and ml_debugger pass a lambda closing over their own `TOOL_SCHEMAS`/tool implementations. 94 lines of duplication deleted.
- Ordering is unchanged in effect: regex findings are still appended before LLM findings, so `dedupe_findings` still resolves a collision in favour of the deterministic one.

### Sequential per-file LLM calls — ✅ Done
- **Was:** each specialist looped over changed files making one blocking LLM round trip at a time; a 20-file PR meant 20 sequential network calls on a webhook job's critical path.
- **Now:** `collect_llm_findings` fans the per-file calls out over a `ThreadPoolExecutor` capped at 4 workers (`pool.map`, so findings stay in file order). The cap is a fixed number, not a rate-limit strategy — marked in the code as such, since the constraint is Groq's API and not local CPU. A single-file PR skips the pool entirely.

### Four divergent `_read_table` implementations — ✅ Done
- **Was:** `agents/ml_debugger_tools.py`, `drift/detector.py`, `drift/root_cause.py` and `drift/pr_drift.py` each had their own reader, and they disagreed: the two drift CLIs fed *any* non-`.parquet` suffix to `read_csv`, while ml_debugger_tools raised `ValueError`.
- **Now:** one `src/drift/tables.py::read_table`, raising for an unsupported suffix everywhere. Raising is the behaviour worth keeping — a `.json` or `.xlsx` read as CSV yields a garbage DataFrame that the drift detectors will then happily report drift on. It also takes an open buffer plus an explicit `suffix`, which is what `pr_drift.load_at_ref` needs for blobs coming out of `git show`.

### Duplicated "optional corroborating CLI" blocks — ✅ Done
- **Was:** `security_agent_tools.py` repeated the same shape for detect-secrets and bandit: `shutil.which` → note if absent → run → parse JSON → note on failure.
- **Now:** one `_corroborate_with_cli(result, tool, args, ...)`. The contract that was implicit in both copies is now stated once: the project's own analysis is the source of truth, the exit code is ignored (both tools exit non-zero merely for *finding* something), and any failure of the tool becomes a note, because these run inside an LLM tool call that must always return something.
- The third instance named in the list, `ml_debugger_tools.py:112`, is the optional **evidently import**, not a CLI. Nothing to share with the other two, left alone.

### Dead `PRReviewState` TypedDict — ✅ Deleted
Removed from `schemas/graph.py` and from the package's imports and `__all__`, along with the now-unused `Dict`/`PRLabel`/`AgentReviewReport` imports it pulled in. Its only reference anywhere was the re-export. The `AgentState` docstring no longer contrasts itself against it. The package's two-products NOTE is otherwise untouched — that flag is about telemetry vs PR-review and is still open.

### Half-applied `_STATIC_ANALYSIS_NODES` registry — ✅ Deleted
The registry covered 1 of 4 branches while the other three were added by hand. `build_multi_agent_graph` now builds one `branches` dict (architecture, security, ml_debugger, plus drift when `repo_path` is given) and loops over it once, so every branch is registered the same way and `branch_names` falls out of it.

### Over-built `AgentNode` Protocol — ❌ Left as is
This one is not over-built. Its docstring gives the reason and it checks out: langgraph's `StateNode` type requires a callable whose parameter is *named* `state`, and `typing.Callable` erases parameter names, so a `Callable[[AgentState], dict]` return annotation fails that structural check in Pylance/pyright even though the function satisfies it at runtime. Replacing the Protocol would reintroduce the type error it was written to fix.

### `CodeChunk(...)` boilerplate in four places — ✅ Done
`chunker.py` now has one `_make_chunk(file_path, symbol_name, content, start_line, end_line, chunk_type, meta, **extra_metadata)`; every call site passes tree-sitter's 0-based lines and the helper does the 1-based conversion and the `chunk_id` hashing. One `CodeChunk(...)` construction left in the module, inside the helper.

### `webhooks.py` if/elif event chain — ✅ Done
Each branch repeated validate → check-condition → schedule → respond. Now there is an `_EVENT_HANDLERS` dict mapping `X-GitHub-Event` to a handler, plus shared `_validated` (schema mismatch → 422) and `_ignored` helpers. `github_webhook` ends in a dict lookup; adding an event is a handler plus one line, and the unhandled-event path is unchanged.

### `config.py` settings frozen at import — ✅ Done
- **Was:** `load_dotenv()` ran at import and each field default was an `os.environ.get(...)` evaluated once in the class body, so the singleton was pinned to whatever the environment looked like at first import and a bare `Settings()` in a test silently picked up the developer's real `.env`.
- **Now:** defaults are plain values and the environment is read in `Settings.from_env()` (which also calls `load_dotenv`), with `settings = Settings.from_env()` at module level. `Settings(...)` builds exactly what it is passed. `tests/conftest.py`'s `dataclasses.replace` fixture still works unchanged; it just isn't the only way in any more.

### `clone_or_checkout` doing a full working-tree checkout — ❌ No longer applicable
The premise was that `pr_drift.py` only ever needs `git show` blobs, so a checkout was wasted work. That stopped being true when the multi-agent graph was wired in: `repo_path` is now also what lets the security (`run_ast_security_scan`) and ml_debugger (`inspect_dataset_schema`, `run_drift_analysis`) tool passes read real files. The working tree is used. A `--filter=blob:none` partial clone is still worth considering for transfer size, but that is an optimisation, not this cleanup.

**Everything in this file is now resolved except:** posting the review back to GitHub, and the unrelated two-products question flagged at the top of `src/schemas/__init__.py`.
