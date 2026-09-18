// Sample data shaped to match src/schemas/review.py and src/schemas/drift.py
// exactly (field names, enum values). This is presentation-only data for
// the demo UI - the real values come from the Python multi-agent graph
// (src/agents/multi_agent_graph.py) and drift engine (src/drift/).

export type Severity = "INFO" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export interface ReviewFinding {
  file_path: string;
  line_number: number | null;
  title: string;
  severity: Severity;
  description: string;
  suggestion: string | null;
}

export interface FinalPRReview {
  pr_id: number;
  pr_title: string;
  repo_full_name: string;
  labels: string[];
  overall_passed: boolean;
  markdown_summary: string;
  inline_findings: ReviewFinding[];
}

export const samplePrReview: FinalPRReview = {
  pr_id: 128,
  pr_title: "Add batch feature-scaling step + retrain fraud-clf on refreshed data",
  repo_full_name: "arc-1103/Skynet_DS",
  labels: ["security", "performance", "config"],
  overall_passed: false,
  markdown_summary:
    "## Automated Review — PR #128: Add batch feature-scaling step + retrain fraud-clf\n\n" +
    "**Overall: CHANGES REQUESTED**",
  inline_findings: [
    {
      file_path: "src/pipeline/train.py",
      line_number: 47,
      title: "Hardcoded credential",
      severity: "CRITICAL",
      description:
        "A literal secret is checked directly into source control, where it stays in git history even if removed later.",
      suggestion:
        "Move the value to an environment variable or secrets manager and load it at runtime.",
    },
    {
      file_path: "src/pipeline/features.py",
      line_number: 112,
      title: "Row-wise DataFrame iteration via .iterrows()",
      severity: "HIGH",
      description:
        "iterrows() is orders of magnitude slower than a vectorized operation because it boxes every row into a Series.",
      suggestion:
        "Replace with a vectorized column expression, or Series.map()/apply() only if a true row-wise callback is unavoidable.",
    },
    {
      file_path: "src/pipeline/infer.py",
      line_number: 63,
      title: "Inference call without torch.no_grad()",
      severity: "MEDIUM",
      description:
        "model.eval() switches off dropout/batchnorm training behavior but does not stop autograd - the forward pass still builds a graph it doesn't need for inference.",
      suggestion: "Wrap the inference calls in `with torch.no_grad():`.",
    },
    {
      file_path: "src/pipeline/train.py",
      line_number: 21,
      title: "Bare except clause",
      severity: "MEDIUM",
      description:
        "except: with no exception type silently swallows every error, including KeyboardInterrupt and SystemExit.",
      suggestion: "Catch a specific exception type and log or re-raise it.",
    },
    {
      file_path: "src/pipeline/config.py",
      line_number: 9,
      title: "print() used instead of logging",
      severity: "LOW",
      description:
        "print() output isn't leveled or capturable the way this codebase's logging setup is.",
      suggestion: "Use the logging module (e.g. logger.info(...)) instead of print().",
    },
  ],
};

export type DriftMethod =
  | "ks"
  | "psi"
  | "page_hinkley"
  | "euclidean"
  | "cosine"
  | "model_based"
  | "mmd";

export interface MethodResult {
  method: DriftMethod;
  statistic: number;
  threshold: number;
  drifted: boolean;
  p_value: number | null;
  detail: string | null;
}

export interface ColumnDriftReport {
  column: string;
  kind: "numerical" | "categorical";
  n_reference: number;
  n_current: number;
  results: MethodResult[];
  skipped: Record<string, string>;
  drifted: boolean;
  vote: string;
}

export interface DatasetDriftReport {
  label: string;
  share_threshold: number;
  columns: ColumnDriftReport[];
}

export const sampleDriftReport: DatasetDriftReport = {
  label: "data/features/transactions.csv",
  share_threshold: 0.5,
  columns: [
    {
      column: "transaction_amount",
      kind: "numerical",
      n_reference: 5000,
      n_current: 4200,
      drifted: true,
      vote: "majority",
      skipped: {},
      results: [
        { method: "ks", statistic: 0.187, threshold: 0.05, drifted: true, p_value: 0.0008, detail: "max CDF gap" },
        { method: "psi", statistic: 0.34, threshold: 0.2, drifted: true, p_value: null, detail: "significant shift" },
      ],
    },
    {
      column: "merchant_category",
      kind: "categorical",
      n_reference: 5000,
      n_current: 4200,
      drifted: true,
      vote: "majority",
      skipped: { ks: "not applicable to categorical columns" },
      results: [
        { method: "psi", statistic: 0.28, threshold: 0.2, drifted: true, p_value: null, detail: "significant shift" },
        { method: "cosine", statistic: 0.14, threshold: 0.1, drifted: true, p_value: null, detail: "shape changed" },
      ],
    },
    {
      column: "account_age_days",
      kind: "numerical",
      n_reference: 5000,
      n_current: 4200,
      drifted: false,
      vote: "majority",
      skipped: {},
      results: [
        { method: "ks", statistic: 0.021, threshold: 0.05, drifted: false, p_value: 0.61, detail: "max CDF gap" },
        { method: "psi", statistic: 0.04, threshold: 0.2, drifted: false, p_value: null, detail: "stable" },
      ],
    },
    {
      column: "is_flagged_fraud",
      kind: "categorical",
      n_reference: 5000,
      n_current: 4200,
      drifted: false,
      vote: "majority",
      skipped: { ks: "not applicable to categorical columns" },
      results: [
        { method: "psi", statistic: 0.03, threshold: 0.2, drifted: false, p_value: null, detail: "stable" },
      ],
    },
  ],
};

export interface FeatureAttribution {
  feature: string;
  share: number;
  mean_abs_shap: number;
  direction: "increased" | "decreased" | "reshaped";
  reference_mean: number | null;
  current_mean: number | null;
}

export interface DriftRootCause {
  label: string;
  surrogate_auc: number;
  auc_threshold: number;
  separable: boolean;
  attributions: FeatureAttribution[];
}

export const sampleRootCause: DriftRootCause = {
  label: "data/features/transactions.csv",
  surrogate_auc: 0.81,
  auc_threshold: 0.55,
  separable: true,
  attributions: [
    { feature: "transaction_amount", share: 0.52, mean_abs_shap: 0.412, direction: "increased", reference_mean: 84.3, current_mean: 131.7 },
    { feature: "merchant_category", share: 0.31, mean_abs_shap: 0.245, direction: "reshaped", reference_mean: null, current_mean: null },
    { feature: "account_age_days", share: 0.1, mean_abs_shap: 0.079, direction: "decreased", reference_mean: 412.0, current_mean: 398.2 },
    { feature: "is_flagged_fraud", share: 0.07, mean_abs_shap: 0.055, direction: "reshaped", reference_mean: null, current_mean: null },
  ],
};
