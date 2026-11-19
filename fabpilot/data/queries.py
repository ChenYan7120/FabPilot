"""供数据工具与 MES/FDC API 共用的可复用读侧查询。"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from fabpilot.data.models import (
    CHANNELS,
    AnomalyLabel,
    Equipment,
    EquipmentAlarm,
    FaultCase,
    MaintenanceRecord,
    SensorReading,
)


def get_equipment(session: Session, equipment_id: str) -> Equipment | None:
    return session.get(Equipment, equipment_id)


def list_equipment(session: Session) -> list[Equipment]:
    return session.query(Equipment).order_by(Equipment.code).all()


def query_sensor_readings(
    session: Session,
    equipment_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    q = session.query(SensorReading).filter(SensorReading.equipment_id == equipment_id)
    if start is not None:
        q = q.filter(SensorReading.ts >= start)
    if end is not None:
        q = q.filter(SensorReading.ts <= end)
    rows = q.order_by(SensorReading.ts.asc()).all()
    return pd.DataFrame(
        [{"ts": r.ts, **{c: getattr(r, c) for c in CHANNELS}} for r in rows],
        columns=["ts", *CHANNELS],
    )


def query_alarms(
    session: Session,
    equipment_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 500,
) -> list[EquipmentAlarm]:
    q = session.query(EquipmentAlarm).filter(EquipmentAlarm.equipment_id == equipment_id)
    if start is not None:
        q = q.filter(EquipmentAlarm.ts >= start)
    if end is not None:
        q = q.filter(EquipmentAlarm.ts <= end)
    return q.order_by(EquipmentAlarm.ts.desc()).limit(limit).all()


def query_maintenance(
    session: Session,
    equipment_id: str,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int = 200,
) -> list[MaintenanceRecord]:
    q = session.query(MaintenanceRecord).filter(MaintenanceRecord.equipment_id == equipment_id)
    if start is not None:
        q = q.filter(MaintenanceRecord.ts >= start)
    if end is not None:
        q = q.filter(MaintenanceRecord.ts <= end)
    return q.order_by(MaintenanceRecord.ts.desc()).limit(limit).all()


def query_fault_cases(
    session: Session,
    equipment_id: str | None = None,
    limit: int = 50,
) -> list[FaultCase]:
    q = session.query(FaultCase)
    if equipment_id is not None:
        q = q.filter(FaultCase.equipment_id == equipment_id)
    return q.order_by(FaultCase.occurred_at.desc()).limit(limit).all()


def query_anomaly_labels(session: Session, equipment_id: str) -> list[AnomalyLabel]:
    return (
        session.query(AnomalyLabel)
        .filter(AnomalyLabel.equipment_id == equipment_id)
        .order_by(AnomalyLabel.start_ts)
        .all()
    )
