"""用合成 fab 环境填充业务数据库。"""
from __future__ import annotations

from fabpilot.data.database import create_all, get_engine
from fabpilot.data.generator import SimulationBundle, generate
from fabpilot.data.models import (
    AnomalyLabel,
    Equipment,
    EquipmentAlarm,
    FaultCase,
    MaintenanceRecord,
    ProcessConfiguration,
    SensorReading,
)
from fabpilot.utils import get_logger
from sqlalchemy.orm import Session

logger = get_logger(__name__)


def seed(seed: int = 42, drop: bool = False, **gen_kwargs) -> SimulationBundle:
    """生成仿真数据包并持久化到已配置的数据库。"""
    engine = get_engine()
    if drop:
        from fabpilot.data.models import Base

        Base.metadata.drop_all(engine)
    create_all()

    bundle = generate(seed=seed, **gen_kwargs)

    with Session(engine) as session:
        _insert(session, bundle)
        session.commit()

    logger.info(
        "Seeded %d equipment, %d readings, %d alarms, %d maintenance, %d fault cases, %d labels",
        len(bundle.equipment),
        len(bundle.readings),
        len(bundle.alarms),
        len(bundle.maintenance),
        len(bundle.fault_cases),
        len(bundle.labels),
    )
    return bundle


def _insert(session: Session, b: SimulationBundle) -> None:
    for _, row in b.equipment.iterrows():
        session.add(Equipment(**row.to_dict()))
    for _, row in b.readings.iterrows():
        session.add(SensorReading(**row.to_dict()))
    for _, row in b.alarms.iterrows():
        session.add(EquipmentAlarm(**row.to_dict()))
    for _, row in b.maintenance.iterrows():
        session.add(MaintenanceRecord(**row.to_dict()))
    for _, row in b.fault_cases.iterrows():
        session.add(FaultCase(**row.to_dict()))
    for _, row in b.process_configs.iterrows():
        session.add(ProcessConfiguration(**row.to_dict()))
    for _, row in b.labels.iterrows():
        session.add(AnomalyLabel(**row.to_dict()))


def seed_into(session_scope, seed: int = 42, **gen_kwargs) -> SimulationBundle:
    """生成仿真数据包并通过提供的 session scope 持久化。

    调用方负责创建 schema（例如通过 ``build_session_scope(url, create_tables=True)``）。评估层用它来填充临时数据库。
    """
    bundle = generate(seed=seed, **gen_kwargs)
    with session_scope() as session:
        _insert(session, bundle)
    logger.info("Seeded simulation bundle into provided session scope")
    return bundle


if __name__ == "__main__":
    seed(drop=True)
