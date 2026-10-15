"""The generation queue: ``/jobs``, ``/queue`` and ``/interrupt``."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Response

from remi.jobs import Job, Submission
from remi.jobs.store import RUNNING
from remi.realtime import progress_snapshot
from remi.server import models as m
from remi.server.deps import ServicesDep, UserDep, iso

router = APIRouter()

ERRORS = {404: {"model": m.Detail}, 409: {"model": m.Detail}}


def job_json(job: Job, *, detail: bool = False, running: bool = False) -> dict[str, Any]:
    data: dict[str, Any] = {
        "id": job.id,
        "seq": job.seq,
        "tab": job.tab,
        "status": job.status,
        "created_at": iso(job.created),
        "started_at": iso(job.started),
        "finished_at": iso(job.finished),
        "progress": None,
        "outputs": job.outputs,
        "error": job.error,
        "errors": job.errors,
        "summary": job.summary,
        "priority": job.priority,
        "forever": job.forever,
        "retry_of": job.retry_of,
        "infotext": job.infotext,
    }
    if running and job.status == RUNNING:
        from modules import shared

        data["progress"] = progress_snapshot(shared.state)
    if detail:
        data.update(core=job.core, scripts=job.scripts, layout_hash=job.layout_hash, schema_version=job.schema_version)
    return data


@router.post("/jobs", response_model=m.JobCreated, status_code=201, responses={422: {"model": m.ValidationErrorBody}}, summary="Queue a generation (validated first)")
def create_job(body: m.JobRequest, services: ServicesDep, user: UserDep) -> dict[str, Any]:
    jobs = services.queue.submit(Submission(body.tab, body.core, body.scripts, priority=body.priority, repeat=body.repeat, forever=body.forever), user=user)
    return {"job_id": jobs[0].id, "job_ids": [job.id for job in jobs], "seq": jobs[0].seq}


@router.get("/jobs", response_model=m.Page[m.Job], summary="Queue and history, newest first")
def list_jobs(services: ServicesDep, cursor: str | None = None, limit: int = Query(50, ge=1, le=200), status: str | None = Query(None, description="comma-separated statuses")) -> dict[str, Any]:
    jobs, next_cursor, total = services.store.history(cursor=cursor, limit=limit, status=status)
    return {"items": [job_json(job, running=True) for job in jobs], "next_cursor": next_cursor, "total": total}


@router.get("/jobs/{job_id}", response_model=m.JobDetail, responses=ERRORS, summary="One job with its parameters")
def get_job(job_id: str, services: ServicesDep) -> dict[str, Any]:
    return job_json(services.queue.get(job_id), detail=True, running=True)


@router.patch("/jobs/{job_id}", response_model=m.Job, responses=ERRORS, summary="Change a job's priority or its generate-forever flag")
def update_job(job_id: str, body: m.JobPatch, services: ServicesDep) -> dict[str, Any]:
    return job_json(services.queue.update(job_id, priority=body.priority, forever=body.forever))


@router.delete("/jobs/{job_id}", status_code=204, responses=ERRORS, summary="Cancel a running job, remove a queued or finished one")
def delete_job(job_id: str, services: ServicesDep) -> Response:
    services.queue.delete(job_id)
    return Response(status_code=204)


@router.post("/jobs/{job_id}/interrupt", response_model=m.Job, responses=ERRORS, summary="Stop a running job (keeping its results) or cancel a queued one")
def interrupt_job(job_id: str, services: ServicesDep) -> dict[str, Any]:
    return job_json(services.queue.interrupt(job_id))


@router.post("/jobs/{job_id}/skip", status_code=204, responses=ERRORS, summary="Skip the running job's current image")
def skip_job(job_id: str, services: ServicesDep) -> Response:
    services.queue.skip(job_id)
    return Response(status_code=204)


@router.post("/jobs/{job_id}/retry", response_model=m.JobCreated, status_code=201, responses=ERRORS, summary="Queue a finished job again")
def retry_job(job_id: str, services: ServicesDep, user: UserDep) -> dict[str, Any]:
    job = services.queue.retry(job_id, user=user)
    return {"job_id": job.id, "job_ids": [job.id], "seq": job.seq}


@router.get("/queue", response_model=m.QueueStatus, summary="Queue state")
def queue_status(services: ServicesDep) -> dict[str, Any]:
    return services.queue.status()


@router.get("/queue/items", response_model=list[m.Job], summary="Queued jobs in run order")
def queued_jobs(services: ServicesDep) -> list[dict[str, Any]]:
    return [job_json(job) for job in services.store.queued()]


@router.post("/queue/reorder", response_model=m.QueueStatus, summary="Put queued jobs in this order (within each priority; at most 500 ids)")
def reorder_queue(body: m.ReorderRequest, services: ServicesDep) -> dict[str, Any]:
    services.queue.reorder(body.order)
    return services.queue.status()


@router.post("/queue/pause", response_model=m.QueueStatus, summary="Do not start queued jobs")
def pause_queue(services: ServicesDep) -> dict[str, Any]:
    services.queue.set_paused(True)
    return services.queue.status()


@router.post("/queue/resume", response_model=m.QueueStatus, summary="Start queued jobs again")
def resume_queue(services: ServicesDep) -> dict[str, Any]:
    services.queue.set_paused(False)
    return services.queue.status()


@router.post("/queue/stop-forever", response_model=m.QueueStatus, summary="Stop every generate-forever job after its current run")
def stop_forever(services: ServicesDep) -> dict[str, Any]:
    services.queue.stop_forever()
    return services.queue.status()


@router.post("/interrupt", response_model=m.QueueStatus, summary="Interrupt whatever generates (also /sdapi requests)")
def interrupt_all(services: ServicesDep, body: m.InterruptRequest | None = None) -> dict[str, Any]:
    services.queue.interrupt_all(clear_queue=bool(body and body.clear_queue))
    return services.queue.status()
