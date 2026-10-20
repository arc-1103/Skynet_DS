import { Injectable } from "@nestjs/common";
import { readdirSync, readFileSync, statSync } from "fs";
import { basename, join } from "path";
import {
  FinalPRReview,
  sampleDriftReport,
  samplePrReview,
  sampleRootCause,
} from "./sample-data";

// Where live data comes from. With SKYNET_API_URL set (e.g.
// http://webhook:8000), the Python service's read-only /results API - so the
// dashboard can run on another host. Unset, its data dir on disk: ../data in
// dev, or SKYNET_DATA_DIR.
const API_URL = process.env.SKYNET_API_URL?.replace(/\/+$/, "") || null;
const API_TOKEN = process.env.SKYNET_API_TOKEN || null;
const API_TIMEOUT_MS = 5000;
const DATA_DIR = process.env.SKYNET_DATA_DIR ?? join(process.cwd(), "..", "data");
export const DATA_VIA: "api" | "disk" = API_URL ? "api" : "disk";

// auto: live output when present, else the bundled sample (labelled as such).
// live: never serve a sample - a panel with no output yet is a 404.
// sample: always the bundled sample, for demos.
export type DataMode = "auto" | "live" | "sample";
export const DATA_MODE: DataMode = (["auto", "live", "sample"] as const).find(
  (m) => m === process.env.SKYNET_DATA_MODE,
) ?? "auto";

// The monitoring sweep rewrites its output every interval; older than this
// and the panel is flagged stale. PR reviews are point-in-time, never stale.
const STALE_AFTER_MS = Number(process.env.SKYNET_STALE_AFTER_SECONDS || 7200) * 1000;

export interface Served<T> {
  data: T;
  source: "live" | "sample";
  updated?: Date;
  stale?: boolean;
}

export class NoLiveData extends Error {}
export class UnreadableLiveData extends Error {}

interface Found<T> {
  data: T;
  updated?: Date;
}

// ---------- live from the backend API ----------

// 404 is "nothing written yet"; every other failure - unreachable, timeout,
// 401, 503 - is an error to show, never a reason to quietly serve the sample.
async function fromApi<T>(path: string): Promise<Found<T> | null> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/results/${path}`, {
      headers: API_TOKEN ? { Authorization: `Bearer ${API_TOKEN}` } : {},
      signal: AbortSignal.timeout(API_TIMEOUT_MS),
    });
  } catch {
    throw new UnreadableLiveData("The backend results API is unreachable.");
  }
  if (res.status === 404) return null;
  if (res.status === 401) throw new UnreadableLiveData("The backend rejected the dashboard's token (check SKYNET_API_TOKEN).");
  if (!res.ok) {
    let detail = "";
    try {
      detail = String((await res.json()).detail ?? "");
    } catch { /* keep the status alone */ }
    throw new UnreadableLiveData(`The backend results API answered HTTP ${res.status}${detail ? `: ${detail}` : "."}`);
  }
  let data: T;
  try {
    data = (await res.json()) as T;
  } catch {
    throw new UnreadableLiveData("The backend results API returned invalid JSON.");
  }
  const updated = res.headers.get("x-data-updated");
  return { data, updated: updated ? new Date(updated) : undefined };
}

// ---------- live from disk ----------

// Newest file under `dir` (one level of subdirs allowed - data/prs/<repo>/)
// whose name ends with `suffix`. A missing dir is "nothing yet"; any other
// filesystem error propagates.
function newestFile(dir: string, suffix: string): { path: string; mtime: number } | null {
  let entries;
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw err;
  }
  let best: { path: string; mtime: number } | null = null;
  for (const entry of entries) {
    const path = join(dir, entry.name);
    const candidates = entry.isDirectory() ? readdirSync(path).map((n) => join(path, n)) : [path];
    for (const p of candidates) {
      if (!p.endsWith(suffix)) continue;
      const mtime = statSync(p).mtimeMs;
      if (!best || mtime > best.mtime) best = { path: p, mtime };
    }
  }
  return best;
}

// The Python FinalPRReview carries no PR identity; it sits next to the cached
// PRPayload (pr_<n>_<sha>.json) that does. Missing sibling -> review as-is.
// (The /results API does this same join server-side.)
function withPrIdentity(review: FinalPRReview, reviewPath: string): FinalPRReview {
  try {
    const pr = JSON.parse(readFileSync(reviewPath.replace(/_review\.json$/, ".json"), "utf-8"));
    return {
      ...review,
      pr_id: pr.pr_id,
      pr_title: pr.title,
      repo_full_name: pr.repo_full_name,
      commit_sha: pr.commit_sha,
      base_ref: pr.base_ref,
    };
  } catch {
    return review;
  }
}

function fromDisk<T>(dir: string, suffix: string, enrich?: (data: T, path: string) => T): Found<T> | null {
  const file = newestFile(dir, suffix);
  if (!file) return null;
  let data: T;
  try {
    data = JSON.parse(readFileSync(file.path, "utf-8")) as T;
  } catch (err) {
    // A live file that exists but won't parse is an error to show, never a
    // reason to quietly swap in the sample.
    throw new UnreadableLiveData(`Live file ${basename(file.path)} could not be read: ${(err as Error).message}`);
  }
  return { data: enrich ? enrich(data, file.path) : data, updated: new Date(file.mtime) };
}

// ---------- one policy over either source ----------

interface Kind<T> {
  apiPath: string;
  dir: string;
  suffix: string;
  sample: T;
  monitoring?: boolean;
  enrich?: (data: T, path: string) => T;
}

async function serve<T>(kind: Kind<T>): Promise<Served<T>> {
  if (DATA_MODE === "sample") return { data: kind.sample, source: "sample" };
  const found = API_URL ? await fromApi<T>(kind.apiPath) : fromDisk(kind.dir, kind.suffix, kind.enrich);
  if (!found) {
    if (DATA_MODE === "live") throw new NoLiveData(`No live ${kind.apiPath} output yet.`);
    return { data: kind.sample, source: "sample" };
  }
  return {
    data: found.data,
    source: "live",
    updated: found.updated,
    stale: Boolean(kind.monitoring && found.updated) && Date.now() - found.updated!.getTime() > STALE_AFTER_MS,
  };
}

@Injectable()
export class ReviewService {
  getPrReview() {
    return serve({ apiPath: "pr-review", dir: join(DATA_DIR, "prs"), suffix: "_review.json", sample: samplePrReview, enrich: withPrIdentity });
  }

  getDriftReport() {
    return serve({ apiPath: "drift", dir: join(DATA_DIR, "monitoring"), suffix: ".drift.json", sample: sampleDriftReport, monitoring: true });
  }

  getRootCause() {
    return serve({ apiPath: "root-cause", dir: join(DATA_DIR, "monitoring"), suffix: ".root_cause.json", sample: sampleRootCause, monitoring: true });
  }
}
