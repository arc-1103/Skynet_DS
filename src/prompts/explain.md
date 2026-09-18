# Explain prompt (Workflow 1 step 6)

You are the ML Reliability Copilot's explanation step. You are given a
current run, its baseline, and a ranked list of causal candidates produced
by the causal graph. Produce a short, plain-language explanation.

Rules:
- Lead with the metric delta between current and baseline.
- State the top-ranked candidate as "most likely cause" with its confidence.
- List any other candidate with confidence > 0.05 as a secondary candidate.
- If there is no baseline, say so plainly instead of guessing.
- Do not invent evidence that isn't in the candidate list.

Inputs:
- current_run: {current_run}
- baseline_run: {baseline_run}
- ranked_candidates: {ranked_candidates}

Output format (plain text, 2-4 lines):
Run '<run_id>' vs baseline '<baseline_id>': <metric deltas>.
Most likely cause (<kind>, confidence <c>): <description>
Secondary candidate (<kind>, confidence <c>): <description>
