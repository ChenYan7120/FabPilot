"""数据访问工具：对 MES / FDC 业务数据库的只读视图。"""
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
# SQL 查询（只读）
# --------------------------------------------------------------------------- #

_FORBIDDEN = ("insert", "update", "delete", "drop", "create", "alter", "truncate",
              "grant", "revoke", "merge", "replace", "call", "copy", "vacuum")


class SqlQueryInput(BaseModel):
    query: str = Field(description="对 fabpilot schema 的只读 SQL SELECT 语句。")
    limit: int = Field(default=200, description="最多返回的行数。")


def _assert_read_only(query: str) -> None:
    stripped = query.strip().rstrip(";").strip().lower()
    first_word = stripped.split()[0] if stripped else ""
    if first_word not in ("select", "with", "show", "explain"):
        raise ToolValidationError("only read-only SELECT / WITH / SHOW statements are allowed")
    for word in _FORBIDDEN:
        if word in stripped:
            # 仅当单词不是语句本身时，才允许 'select' 中包含 'create' 等字样
            raise ToolValidationError(f"statement contains forbidden keyword '{word}'")
    if ";" in query:
        remainder = query.split(";", 1)[1].strip()
        if remainder:
            raise ToolValidationError("multiple statements are not allowed")


@tool(
    name="sql_query",
    description=(
        "对 fabpilot MES/FDC 数据库执行只读的 SQL SELECT 查询。"
        "表包括：equipment、sensor_readings、equipment_alarms、"
        "maintenance_records、fault_cases、process_configurations、anomaly_labels。"
        "用于没有专用工具覆盖的任意聚合 / 连接查询。"
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
# 设备状态
# --------------------------------------------------------------------------- #


class EquipmentInput(BaseModel):
    equipment_id: str | None = Field(
        default=None, description="设备编码（如 'ETCH-01'）。省略则列出全部。"
    )


def _equipment_to_dict(eq) -> EquipmentInfo:
    return EquipmentInfo(
        code=eq.code, name=eq.name, equipment_type=eq.type,
        status=eq.status, location=eq.chamber or "",
    )


@tool(
    name="equipment_status",
    description="返回某台设备的状态 / 元数据，或列出全部设备。",
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
# 传感器数据
# --------------------------------------------------------------------------- #


class SensorDataInput(BaseModel):
    equipment_id: str = Field(description="设备编码，如 'ETCH-01'。")
    channel: str | None = Field(
        default=None, description="要选择的单个通道；省略则选择全部通道。"
    )
    start: datetime | None = Field(default=None, description="起始时间戳（含该时刻，ISO-8601）。")
    end: datetime | None = Field(default=None, description="结束时间戳（含该时刻，ISO-8601）。")
    limit: int = Field(default=500, description="最多返回的原始数据点数。")
    summarize: bool = Field(default=False, description="返回各通道的汇总统计，而不是原始数据行。")


@tool(
    name="sensor_data",
    description=(
        "查询某台设备的历史 sensor 读数（pressure、temperature、gas_flow、rf_power、"
        "vacuum_pressure、bias_voltage、helium_flow）。返回原始数据行"
        "和 / 或各通道的汇总统计。"
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
# 告警日志
# --------------------------------------------------------------------------- #


class AlarmInput(BaseModel):
    equipment_id: str = Field(description="设备编码，如 'ETCH-01'。")
    start: datetime | None = None
    end: datetime | None = None
    limit: int = Field(default=100, description="最多返回的告警数量。")


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
    description="查询设备告警日志（含 code、severity、description、timestamp 字段）。",
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
# 维护记录
# --------------------------------------------------------------------------- #


class MaintenanceInput(BaseModel):
    equipment_id: str = Field(description="设备编码，如 'ETCH-01'。")
    start: datetime | None = None
    end: datetime | None = None
    limit: int = Field(default=100, description="最多返回的记录数。")


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
    description="查询某台设备的维护记录（预防性 / 纠正性）。",
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
# 故障案例
# --------------------------------------------------------------------------- #


class FaultCaseInput(BaseModel):
    equipment_id: str | None = Field(default=None, description="按设备过滤；省略则返回全部。")
    limit: int = Field(default=50, description="最多返回的案例数量。")


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
    description="查询带有已知根因与解决方案的历史故障案例。",
    input_model=FaultCaseInput,
    tags=["data"],
)
def fault_case(
    ctx: ToolContext, equipment_id: str | None = None, limit: int = 50,
) -> FaultCaseResult:
    with ctx.session_scope() as session:  # type: ignore[misc]
        cases = queries.query_fault_cases(session, equipment_id, limit)
    return FaultCaseResult(n_cases=len(cases), cases=[_fault_dict(c) for c in cases])
