"""模拟半导体 MES/FDC 领域的 SQLAlchemy ORM 模型。

所有数据均为合成数据，不代表任何真实的企业系统。
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# 刻蚀设备仿真中的标准多元传感器通道。
CHANNELS = [
    "pressure",
    "temperature",
    "gas_flow",
    "rf_power",
    "vacuum_pressure",
    "bias_voltage",
    "helium_flow",
]
DEFAULT_TARGET = "gas_flow"


class Base(DeclarativeBase):
    pass


class Equipment(Base):
    __tablename__ = "equipment"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    type: Mapped[str] = mapped_column(String(64), default="Etch")
    chamber: Mapped[str] = mapped_column(String(64), default="A")
    manufacturer: Mapped[str] = mapped_column(String(64), default="FabPilot-Sim")
    status: Mapped[str] = mapped_column(String(32), default="RUN")
    description: Mapped[str] = mapped_column(Text, default="")

    readings: Mapped[list["SensorReading"]] = relationship(back_populates="equipment")
    alarms: Mapped[list["EquipmentAlarm"]] = relationship(back_populates="equipment")
    maintenance: Mapped[list["MaintenanceRecord"]] = relationship(back_populates="equipment")
    fault_cases: Mapped[list["FaultCase"]] = relationship(back_populates="equipment")


class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    equipment_id: Mapped[str] = mapped_column(ForeignKey("equipment.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    pressure: Mapped[float] = mapped_column(Float)
    temperature: Mapped[float] = mapped_column(Float)
    gas_flow: Mapped[float] = mapped_column(Float)
    rf_power: Mapped[float] = mapped_column(Float)
    vacuum_pressure: Mapped[float] = mapped_column(Float)
    bias_voltage: Mapped[float] = mapped_column(Float)
    helium_flow: Mapped[float] = mapped_column(Float)

    equipment: Mapped[Equipment] = relationship(back_populates="readings")


class EquipmentAlarm(Base):
    __tablename__ = "equipment_alarms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    equipment_id: Mapped[str] = mapped_column(ForeignKey("equipment.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    alarm_code: Mapped[str] = mapped_column(String(32), index=True)
    alarm_type: Mapped[str] = mapped_column(String(64))
    severity: Mapped[str] = mapped_column(String(16), default="WARN")
    message: Mapped[str] = mapped_column(Text, default="")
    acknowledged: Mapped[bool] = mapped_column(Integer, default=0)

    equipment: Mapped[Equipment] = relationship(back_populates="alarms")


class MaintenanceRecord(Base):
    __tablename__ = "maintenance_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    equipment_id: Mapped[str] = mapped_column(ForeignKey("equipment.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    maintenance_type: Mapped[str] = mapped_column(String(32))
    description: Mapped[str] = mapped_column(Text, default="")
    technician: Mapped[str] = mapped_column(String(64), default="")
    downtime_hours: Mapped[float] = mapped_column(Float, default=0.0)

    equipment: Mapped[Equipment] = relationship(back_populates="maintenance")


class FaultCase(Base):
    __tablename__ = "fault_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    equipment_id: Mapped[str] = mapped_column(ForeignKey("equipment.id"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    symptoms: Mapped[str] = mapped_column(Text, default="")
    root_cause: Mapped[str] = mapped_column(Text, default="")
    resolution: Mapped[str] = mapped_column(Text, default="")
    occurred_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    related_alarm_codes: Mapped[list] = mapped_column(JSON, default=list)
    doc_refs: Mapped[list] = mapped_column(JSON, default=list)

    equipment: Mapped[Equipment] = relationship(back_populates="fault_cases")


class ProcessConfiguration(Base):
    __tablename__ = "process_configurations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    equipment_id: Mapped[str] = mapped_column(ForeignKey("equipment.id"), index=True)
    recipe_name: Mapped[str] = mapped_column(String(64))
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    effective_from: Mapped[datetime] = mapped_column(DateTime)
    effective_to: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AnomalyLabel(Base):
    """由仿真器注入的 ground-truth 标签，仅用于评估。

    该表记录 *已知* 的异常窗口及其真实根因，使评估框架无需猜测即可对诊断准确率打分。
    """

    __tablename__ = "anomaly_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    equipment_id: Mapped[str] = mapped_column(ForeignKey("equipment.id"), index=True)
    scenario: Mapped[str] = mapped_column(String(64), index=True)
    start_ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    end_ts: Mapped[datetime] = mapped_column(DateTime)
    root_cause: Mapped[str] = mapped_column(Text)
    affected_channels: Mapped[list] = mapped_column(JSON, default=list)
    severity: Mapped[str] = mapped_column(String(16), default="MAJOR")
