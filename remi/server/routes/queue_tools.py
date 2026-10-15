"""Queue page tools on top of ``/jobs``: duplicate, edit a queued job's
parameters, move a queued job next to another, star, clear the history, and
import the queues of the extensions the Queue replaces (Agent Scheduler,
Prompt Queue).

Server paths of the imported files are left out under
``--hide-ui-dir-config``."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import Field

from remi.server import models as m
from remi.server.deps import ServicesDep, UserDep
from remi.server.routes.jobs import job_json

router = APIRouter()

ERRORS = {404: {"model": m.Detail}, 409: {"model": m.Detail}}
ImportSource = Literal["agent-scheduler", "prompt-queue"]


class JobParams(m.Model):
    core: dict[str, Any] = {}
    scripts: dict[str, Any] = {}


class StarRequest(m.Model):
    ids: list[str] = Field(..., min_length=1, max_length=500)
    starred: bool = True


class MoveRequest(m.Model):
    id: str
    after: str | None = Field(None, description="the queued job it now follows (same priority group)")
    before: str | None = Field(None, description="the queued job it now precedes, used when `after` is not given or not queued any more")
    priority: int | None = Field(None, ge=-1000, le=1000, description="its new priority (the group of its neighbours)")


class ClearHistoryRequest(m.Model):
    keep_starred: bool = True


class JobCount(m.Model):
    count: int


class ImportSourceInfo(m.Model):
    id: ImportSource
    label: str
    available: bool
    count: int = Field(0, description="records found (pending prompts for the Prompt Queue)")
    path: str | None = Field(None, description="where it was found (not under --hide-ui-dir-config)")


class ImportItem(m.Model):
    source_id: str
    title: str
    outcome: Literal["queued", "history", "refused", "invalid", "skipped", "error"]
    job_id: str | None = None
    notes: list[m.Note] = []


class ImportReport(m.Model):
    source: ImportSource
    path: str | None = Field(None, description="the file imported (not under --hide-ui-dir-config)")
    total: int
    queued: int
    history: int
    refused: int = Field(description="pending tasks with arguments that cannot be mapped: kept in the history, not replayable")
    invalid: int = Field(description="pending tasks whose values the current form rejects: kept in the history to edit")
    skipped: int = Field(description="imported before (or finished prompts)")
    errors: int
    paused: bool = Field(description="the queue was paused so the imported jobs can be reviewed")
    items: list[ImportItem]


def _hide_paths() -> bool:
    from modules import shared

    return bool(getattr(shared.cmd_opts, "hide_ui_dir_config", False))


@router.post("/jobs/{job_id}/duplicate", response_model=m.JobCreated, status_code=201, responses={**ERRORS, 422: {"model": m.ValidationErrorBody}}, summary="Queue a copy of a job (right after it while it is queued)")
def duplicate_job(job_id: str, services: ServicesDep, user: UserDep) -> dict[str, Any]:
    job = services.queue.duplicate(job_id, user=user)
    return {"job_id": job.id, "job_ids": [job.id], "seq": job.seq}


@router.put("/jobs/{job_id}/params", response_model=m.JobDetail, responses={**ERRORS, 422: {"model": m.ValidationErrorBody}}, summary="Replace a queued job's parameters (validated)")
def update_job_params(job_id: str, body: JobParams, services: ServicesDep) -> dict[str, Any]:
    return job_json(services.queue.update_params(job_id, body.core, body.scripts), detail=True)


@router.post("/queue/move", response_model=m.QueueStatus, responses=ERRORS, summary="Move a queued job next to another (and into its priority group)")
def move_job(body: MoveRequest, services: ServicesDep) -> dict[str, Any]:
    services.queue.move(body.id, after=body.after, before=body.before, priority=body.priority)
    return services.queue.status()


@router.post("/jobs/star", response_model=JobCount, summary="Star or unstar jobs (starred jobs survive clearing the history)")
def star_jobs(body: StarRequest, services: ServicesDep) -> dict[str, Any]:
    return {"count": services.queue.set_starred(body.ids, body.starred)}


@router.post("/jobs/clear-history", response_model=JobCount, summary="Delete finished jobs (their images stay in the gallery)")
def clear_history(body: ClearHistoryRequest, services: ServicesDep) -> dict[str, Any]:
    return {"count": services.queue.clear_history(keep_starred=body.keep_starred)}


@router.get("/queue/import", response_model=list[ImportSourceInfo], summary="Queues of replaced extensions found on this machine")
def import_sources() -> list[dict[str, Any]]:
    from remi.jobs.importers import sources

    found = sources()
    if _hide_paths():
        for source in found:
            source["path"] = None
    return found


@router.post("/queue/import/{source}", response_model=ImportReport, responses={404: {"model": m.Detail}, 422: {"model": m.Detail}}, summary="Import a replaced extension's queue and history (idempotent)")
async def import_queue(source: ImportSource, services: ServicesDep, user: UserDep) -> dict[str, Any]:
    from remi.jobs.importers import Importer, ImportFailed

    def run() -> dict[str, Any]:
        report = Importer(services.queue, user=user).run(source).json()
        if _hide_paths():
            report["path"] = None
        return report

    try:
        return await run_in_threadpool(run)
    except ImportFailed as error:
        raise HTTPException(status_code=error.status, detail=str(error)) from error
