"""Data-access tools: read-only views over the MES / FDC business database."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field
from sqlalchemy import text

from fabpilot.data import queries
from fabpilot.tools.registry import ToolContext, tool
from fabpilot.tools.schemas import (
    AlarmLogResult,
    EquipmentInfo,
    FaultCaseResult,
    MaintenanceResult,
    SensorDataResult,
)
from fabpilot.utils import ToolValidationError

# --------------------------------------------------------------------------- #
# SQL query (read-only)
# --------------------------------------------------------------------------- #

_FORBIDDEN = ("insert", "update", "delete", "drop", "create", "alter", "truncate",
              "grant", "revoke", "merge", "replace", "call", "copy", "vacuum")


class SqlQueryInput(BaseModel):
    query: str = Field(description="Read-only SQL SELECT statement over the fabpilot schema.")
    limit: int = Field(default=200, description="Maximum number of rows to return.")


def _assert_read_only(query: str) -> None:
    stripped = query.strip().rstrip(";").strip().lower()
    first_word = stripped.split()[0] if stripped else ""
    if first_word not in ("select", "with", "show", "explain"):
        raise ToolValidationError("only read-only SELECT / WITH / SHOW statements are allowed")
    for word in _FORBIDDEN:
        if word in stripped:
            # allow 'select' containing 'create' etc. only when word is not the statement
            raise ToolValidationError(f"statement contains forbidden keyword '{word}'")
    if ";" in query:
        remainder = query.split(";", 1)[1].strip()
        if remainder:
            raise ToolValidationError("multiple statements are not allowed")


@tool(
    name="sql_query",
    description=(
        "Run a read-only SQL SELECT query against the fabpilot MES/FDC database. "
        "Tables include: equipment, sensor_readings, equipment_alarms, "
        "maintenance_records, fault_cases, process_configurations, anomaly_labels. "
        "Use for arbitrary aggregates / joins that no dedicated tool covers."
    ),
    input_model=SqlQueryInput,
    tags=["data", "sql"],
)
def sql_query(ctx: ToolContext, query: str, limit: int = 200) -> dict:
    _assert_read_only(query)
    with ctx.session_scope() as session:  # type: ignore[misc]
        result = session.execute(text(query))
        rows = result.fetchmany(limit + 1)
        keys = list(result.keys())
    data = [dict(zip(keys, r)) for r in rows[:limit]]
    return {"columns": keys, "rows": data, "truncated": len(rows) > limit, "n": len(data)}


# --------------------------------------------------------------------------- #
# Equipment status
# --------------------------------------------------------------------------- #


class EquipmentInput(BaseModel):
    equipment_id: str | None = Field(
        default=None, description="Equipment code (e.g. 'ETCH-01'). Omit to list all."
    )


def _equipment_to_dict(eq) -> EquipmentInfo:
    return EquipmentInfo(
        code=eq.code, name=eq.name, equipment_type=eq.type,
        status=eq.status, location=eq.chamber or "",
    )


@tool(
    name="equipment_status",
    description="Return the status / metadata of one equipment or list all equipment.",
    input_model=EquipmentInput,
    tags=["data"],
)
def equipment_status(ctx: ToolContext, equipment_id: str | None = None):
    with ctx.session_scope() as session:  # type: ignore[misc]
        if equipment_id:
            eq = queries.get_equipment(session, equipment_id)
            if eq is None:
                raise ToolValidationError(f"equipment '{equipment_id}' not found")
            return _equipment_to_dict(eq)
        return {"equipment": [_equipment_to_dict(e) for e in queries.list_equipment(session)]}


# --------------------------------------------------------------------------- #
# Sensor data
# --------------------------------------------------------------------------- #


class SensorDataInput(BaseModel):
    equipment_id: str = Field(description="Equipment code, e.g. 'ETCH-01'.")
    channel: str | None = Field(
        default=None, description="Single channel to select; omit for all channels."
    )
    start: datetime | None = Field(default=None, description="Inclusive start timestamp (ISO-8601).")
    end: datetime | None = Field(default=None, description="Inclusive end timestamp (ISO-8601).")
    limit: int = Field(default=500, description="Maximum number of raw points to return.")
    summarize: bool = Field(default=False, description="Return per-channel summary statistics instead of raw rows.")


@tool(
    name="sensor_data",
    description=(
        "Query historical sensor readings (pressure, temperature, gas_flow, rf_power, "
        "vacuum_pressure, bias_voltage, helium_flow) for an equipment. Returns raw rows "
        "and/or per-channel summary statistics."
    ),
    input_model=SensorDataInput,
    tags=["data"],
)
def sensor_data(
    ctx: ToolContext,
    equipment_id: str,
    channel: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 500,
    summarize: bool = False,
) -> SensorDataResult:
    with ctx.session_scope() as session:  # type: ignore[misc]
        if queries.get_equipment(session, equipment_id) is None:
            raise ToolValidationError(f"equipment '{equipment_id}' not found")
        df = queries.query_sensor_readings(session, equipment_id, start, end)

    if df.empty:
        return SensorDataResult(equipment_id=equipment_id, n_points=0)

    channels = [channel] if channel else list(df.columns.drop("ts"))
    if channel and channel not in df.columns:
        raise ToolValidationError(f"unknown channel '{channel}'")

    summary: dict[str, dict[str, float]] = {}
    for c in channels:
        s = df[c].dropna()
        summary[c] = {
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std()), 6),
            "min": round(float(s.min()), 6),
            "max": round(float(s.max()), 6),
        }

    last = df.tail(limit)[["ts", *channels]]
    last["ts"] = last["ts"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    rows = last.to_dict(orient="records")

    return SensorDataResult(
        equipment_id=equipment_id,
        start_ts=df["ts"].min(),
        end_ts=df["ts"].max(),
        n_points=len(df),
        channels=channels,
        summary=summary,
        last_10_points=rows[-10:],
    )


# --------------------------------------------------------------------------- #
# Alarm log
# --------------------------------------------------------------------------- #


class AlarmInput(BaseModel):
    equipment_id: str = Field(description="Equipment code, e.g. 'ETCH-01'.")
    start: datetime | None = None
    end: datetime | None = None
    limit: int = Field(default=100, description="Maximum number of alarms to return.")


def _alarm_dict(a) -> dict:
    return {
        "ts": a.ts.isoformat(),
        "alarm_code": a.alarm_code,
        "alarm_type": a.alarm_type,
        "severity": a.severity,
        "message": a.message,
    }


@tool(
    name="alarm_log",
    description="Query the equipment alarm log (code, severity, description, timestamp).",
    input_model=AlarmInput,
    tags=["data"],
)
def alarm_log(
    ctx: ToolContext, equipment_id: str, start: datetime | None = None,
    end: datetime | None = None, limit: int = 100,
) -> AlarmLogResult:
    with ctx.session_scope() as session:  # type: ignore[misc]
        if queries.get_equipment(session, equipment_id) is None:
            raise ToolValidationError(f"equipment '{equipment_id}' not found")
        alarms = queries.query_alarms(session, equipment_id, start, end, limit)
    return AlarmLogResult(
        equipment_id=equipment_id, n_alarms=len(alarms),
        alarms=[_alarm_dict(a) for a in alarms],
    )


# --------------------------------------------------------------------------- #
# Maintenance record
# --------------------------------------------------------------------------- #


class MaintenanceInput(BaseModel):
    equipment_id: str = Field(description="Equipment code, e.g. 'ETCH-01'.")
    start: datetime | None = None
    end: datetime | None = None
    limit: int = Field(default=100, description="Maximum number of records to return.")


def _maint_dict(m) -> dict:
    return {
        "ts": m.ts.isoformat(),
        "type": m.maintenance_type,
        "description": m.description,
        "technician": m.technician,
        "downtime_hours": m.downtime_hours,
    }


@tool(
    name="maintenance_record",
    description="Query maintenance records (preventive / corrective) for an equipment.",
    input_model=MaintenanceInput,
    tags=["data"],
)
def maintenance_record(
    ctx: ToolContext, equipment_id: str, start: datetime | None = None,
    end: datetime | None = None, limit: int = 100,
) -> MaintenanceResult:
    with ctx.session_scope() as session:  # type: ignore[misc]
        if queries.get_equipment(session, equipment_id) is None:
            raise ToolValidationError(f"equipment '{equipment_id}' not found")
        records = queries.query_maintenance(session, equipment_id, start, end, limit)
    return MaintenanceResult(
        equipment_id=equipment_id, n_records=len(records),
        records=[_maint_dict(m) for m in records],
    )


# --------------------------------------------------------------------------- #
# Fault case
# --------------------------------------------------------------------------- #


class FaultCaseInput(BaseModel):
    equipment_id: str | None = Field(default=None, description="Filter by equipment; omit for all.")
    limit: int = Field(default=50, description="Maximum number of cases to return.")


def _fault_dict(f) -> dict:
    return {
        "case_id": f.id,
        "equipment_id": f.equipment_id,
        "title": f.title,
        "root_cause": f.root_cause,
        "symptoms": f.symptoms,
        "related_alarm_codes": f.related_alarm_codes,
        "occurred_at": f.occurred_at.isoformat() if f.occurred_at else None,
        "resolution": f.resolution,
    }


@tool(
    name="fault_case",
    description="Query historical fault cases with known root causes and resolutions.",
    input_model=FaultCaseInput,
    tags=["data"],
)
def fault_case(
    ctx: ToolContext, equipment_id: str | None = None, limit: int = 50,
) -> FaultCaseResult:
    with ctx.session_scope() as session:  # type: ignore[misc]
        cases = queries.query_fault_cases(session, equipment_id, limit)
    return FaultCaseResult(n_cases=len(cases), cases=[_fault_dict(c) for c in cases])
