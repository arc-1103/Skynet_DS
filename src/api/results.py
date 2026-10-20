"""Read-only results API for the dashboard (demo-ui). Serves the newest
output the worker already writes to disk - the PR review cache and the
monitoring sweep's drift / root-cause files - so the dashboard can run on
a different host than this service. Thin layer: no analysis here, only
find, validate, return.

Every response carries X-Data-Updated (the file's mtime, ISO 8601). 404
means nothing has been written yet; 503 means a file exists but does not
parse as its schema, which the dashboard must show as an error rather than
swap for sample data."""

import hmac
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from src.core.config import Settings, get_settings
from src.schemas.drift import DatasetDriftReport
from src.schemas.review import FinalPRReview
from src.schemas.root_cause import DriftRootCause

logger = logging.getLogger(__name__)


def _authorized(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> Settings:
    token = settings.results_api_token
    if token and not (
        authorization
        and authorization.startswith("Bearer ")
        and hmac.compare_digest(authorization.removeprefix("Bearer "), token)
    ):
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")
    return settings


router = APIRouter(prefix="/results", dependencies=[Depends(_authorized)])


def _newest(directory: str, pattern: str) -> Path | None:
    files = [p for p in Path(directory).glob(pattern) if p.is_file()]
    return max(files, key=lambda p: p.stat().st_mtime, default=None)


def _load(path: Path | None, model: type[BaseModel], what: str) -> tuple[dict, Path]:
    if path is None:
        raise HTTPException(status_code=404, detail=f"No {what} has been written yet.")
    try:
        return model.model_validate_json(path.read_bytes()).model_dump(mode="json"), path
    except (OSError, ValidationError) as exc:
        logger.warning("unreadable %s at %s: %s", what, path, exc)
        raise HTTPException(
            status_code=503, detail=f"The latest {what} ({path.name}) could not be read."
        ) from exc


def _respond(payload: dict, path: Path) -> JSONResponse:
    updated = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return JSONResponse(payload, headers={"X-Data-Updated": updated, "Cache-Control": "no-store"})


@router.get("/pr-review")
def latest_pr_review(settings: Settings = Depends(get_settings)) -> JSONResponse:
    review, path = _load(_newest(settings.pr_cache_dir, "*/*_review.json"), FinalPRReview, "PR review")
    # FinalPRReview carries no PR identity; the cached PRPayload beside it
    # (pr_<n>_<sha>.json) does. Missing or unreadable -> review as-is.
    try:
        pr = json.loads(path.with_name(path.name.removesuffix("_review.json") + ".json").read_text("utf-8"))
        review |= {
            "pr_id": pr.get("pr_id"),
            "pr_title": pr.get("title"),
            "repo_full_name": pr.get("repo_full_name"),
            "commit_sha": pr.get("commit_sha"),
            "base_ref": pr.get("base_ref"),
        }
    except (OSError, ValueError):
        pass
    return _respond(review, path)


@router.get("/drift")
def latest_drift(settings: Settings = Depends(get_settings)) -> JSONResponse:
    return _respond(*_load(_newest(settings.monitoring_dir, "*.drift.json"), DatasetDriftReport, "drift report"))


@router.get("/root-cause")
def latest_root_cause(settings: Settings = Depends(get_settings)) -> JSONResponse:
    return _respond(*_load(_newest(settings.monitoring_dir, "*.root_cause.json"), DriftRootCause, "root cause"))
