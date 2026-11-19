"""FastAPI routes: MES/FDC data surface + agent task surface."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from fabpilot.api.schemas import (
    AnomalyRequest,
    ErrorResponse,
    ForecastRequest,
    SearchRequest,
    TaskRequest,
    TaskResponse,
)
from fabpilot.api.service import FabPilotService, FabPilotServiceError


router = APIRouter(prefix="/api/v1", tags=["fabpilot"])


def _service() -> FabPilotService:
    # Bound lazily so the app factory can inject a custom service in tests.
    from fabpilot.api.app import current_service  # noqa: PLC0415

    return current_service()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---- equipment / FDC ------------------------------------------------------ #

@router.get("/equipment")
def list_equipment() -> dict[str, list[dict]]:
    return {"equipment": _service().list_equipment()}


@router.get("/equipment/{equipment_id}")
def get_equipment(equipment_id: str) -> dict:
    try:
        return _service().get_equipment(equipment_id)
    except FabPilotServiceError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/equipment/{equipment_id}/sensor-data")
def sensor_data(
    equipment_id: str,
    channel: str | None = Query(default=None),
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=100000),
) -> dict:
    from datetime import datetime

    def _dt(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None

    try:
        return _service().sensor_data(equipment_id, channel=channel, start=_dt(start), end=_dt(end), limit=limit)
    except FabPilotServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/equipment/{equipment_id}/alarms")
def alarms(
    equipment_id: str,
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=10000),
) -> dict:
    from datetime import datetime

    def _dt(value: str | None):
        return datetime.fromisoformat(value) if value else None

    try:
        return _service().alarms(equipment_id, start=_dt(start), end=_dt(end), limit=limit)
    except FabPilotServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/equipment/{equipment_id}/maintenance")
def maintenance(
    equipment_id: str,
    start: str | None = Query(default=None),
    end: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=10000),
) -> dict:
    from datetime import datetime

    def _dt(value: str | None):
        return datetime.fromisoformat(value) if value else None

    try:
        return _service().maintenance(equipment_id, start=_dt(start), end=_dt(end), limit=limit)
    except FabPilotServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/fault-cases")
def fault_cases(equipment_id: str | None = Query(default=None), limit: int = Query(default=50, ge=1, le=1000)) -> dict:
    return _service().fault_cases(equipment_id=equipment_id, limit=limit)


# ---- analytics ------------------------------------------------------------ #

@router.post("/forecast")
def forecast(body: ForecastRequest) -> dict:
    try:
        return _service().forecast(
            body.equipment_id,
            target=body.target,
            horizon=body.horizon,
            lookback=body.lookback,
            model=body.model,
        )
    except FabPilotServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/anomaly-detection")
def detect_anomalies(body: AnomalyRequest) -> dict:
    try:
        return _service().detect_anomalies(
            body.equipment_id,
            channel=body.channel,
            method=body.method,
            window=body.window,
            threshold=body.threshold,
        )
    except FabPilotServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/search")
def search(body: SearchRequest) -> dict:
    try:
        return _service().search_knowledge(body.query, k=body.k, doc_type=body.doc_type)
    except FabPilotServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---- agent tasks ---------------------------------------------------------- #

@router.post("/tasks", response_model=TaskResponse, responses={400: {"model": ErrorResponse}})
def run_task(body: TaskRequest) -> TaskResponse:
    try:
        state = _service().run_task(body.user_task, session_id=body.session_id, thread_id=body.thread_id)
    except FabPilotServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return TaskResponse(
        thread_id=state["thread_id"],
        session_id=state["session_id"],
        status=state.get("status") or "unknown",
        final_report=state.get("final_report", ""),
        delegated_agents=[r.get("agent") for r in state.get("delegation_records", [])],
        n_tool_calls=len(state.get("tool_history", [])),
        hypotheses=state.get("hypotheses", []),
        errors=state.get("errors", []),
    )


@router.get("/tasks/{thread_id}", response_model=TaskResponse)
def get_task(thread_id: str) -> TaskResponse:
    record = _service().get_task(thread_id)
    if record is None:
        raise HTTPException(status_code=404, detail="task not found")
    return TaskResponse(**record)


@router.get("/tasks")
def list_tasks() -> dict[str, list]:
    return {"tasks": _service().list_tasks()}
