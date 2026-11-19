"""Reproducible semiconductor fabrication (etch tool) simulation.

Generates multivariate sensor time series with *known* injected fault windows so
that evaluation can be scored against ground truth. Everything is synthetic and
explicitly does NOT represent any real enterprise data or a fully validated
physical process model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from fabpilot.data.models import CHANNELS


@dataclass(frozen=True)
class FaultScenario:
    name: str
    root_cause: str
    affected_channels: tuple[str, ...]
    alarm_codes: tuple[str, ...]
    description: str
    injectors: dict[str, tuple[str, float]] = field(default_factory=dict)  # channel -> (kind, magnitude)
    severity: str = "MAJOR"


SCENARIOS: dict[str, FaultScenario] = {
    "mfc_flow_drift": FaultScenario(
        name="mfc_flow_drift",
        root_cause="MFC mass flow controller drift / control valve stiction",
        affected_channels=("gas_flow", "pressure"),
        alarm_codes=("FLOW_HIGH", "MFC_DRIFT"),
        description="Gas flow setpoint tracking degrades: actual flow drifts above "
        "setpoint and chamber pressure follows.",
        injectors={"gas_flow": ("drift", 12.0), "pressure": ("drift", 3.0)},
    ),
    "sensor_bias": FaultScenario(
        name="sensor_bias",
        root_cause="Temperature sensor bias / in-situ probe drift",
        affected_channels=("temperature",),
        alarm_codes=("TEMP_BIAS", "TEMP_DEVIATION"),
        description="Temperature sensor reads high by a growing offset while true chamber "
        "temperature is stable.",
        injectors={"temperature": ("bias", 4.0)},
    ),
    "rf_power_instability": FaultScenario(
        name="rf_power_instability",
        root_cause="RF generator power instability / reflected power fluctuation",
        affected_channels=("rf_power",),
        alarm_codes=("RF_OSC", "RF_REFLECT_HIGH"),
        description="Forward RF power oscillates with growing amplitude, indicating an "
        "impedance mismatch or generator instability.",
        injectors={"rf_power": ("oscillation", 40.0)},
    ),
    "vacuum_leak": FaultScenario(
        name="vacuum_leak",
        root_cause="Vacuum system leak / slow pump degradation",
        affected_channels=("vacuum_pressure", "pressure"),
        alarm_codes=("VAC_LEAK", "PRESSURE_HIGH"),
        description="Base vacuum pressure rises slowly (leak / degraded pump); chamber "
        "pressure rises in response.",
        injectors={"vacuum_pressure": ("leak", 8e-6), "pressure": ("leak", 4.0)},
    ),
}

# channel -> (baseline, slow sinusoidal amplitude, gaussian noise std)
CHANNEL_BASELINE: dict[str, tuple[float, float, float]] = {
    "pressure": (30.0, 2.0, 0.4),
    "temperature": (60.0, 1.0, 0.15),
    "gas_flow": (100.0, 1.5, 0.6),
    "rf_power": (500.0, 5.0, 2.0),
    "vacuum_pressure": (5e-6, 5e-7, 1e-7),
    "bias_voltage": (-200.0, 2.0, 0.8),
    "helium_flow": (10.0, 0.3, 0.1),
}

# Deterministic fault schedule per equipment: (scenario, start_fraction, duration_fraction).
# ETCH-03 keeps one fault spanning the final 24h so demo scenario C has a live anomaly.
_SCHEDULE: dict[str, list[tuple[str, float, float]]] = {
    "ETCH-01": [("mfc_flow_drift", 0.10, 0.05), ("sensor_bias", 0.45, 0.05)],
    "ETCH-02": [("vacuum_leak", 0.20, 0.08), ("rf_power_instability", 0.55, 0.08)],
    "ETCH-03": [("sensor_bias", 0.30, 0.05), ("mfc_flow_drift", 1.0 - 24.0 / 168.0, 24.0 / 168.0)],
    "ETCH-04": [("rf_power_instability", 0.15, 0.10), ("vacuum_leak", 0.60, 0.08)],
    "ETCH-05": [("mfc_flow_drift", 0.40, 0.08), ("rf_power_instability", 0.70, 0.10)],
}


@dataclass
class SimulationBundle:
    equipment: pd.DataFrame
    readings: pd.DataFrame
    alarms: pd.DataFrame
    maintenance: pd.DataFrame
    fault_cases: pd.DataFrame
    process_configs: pd.DataFrame
    labels: pd.DataFrame
    scenarios: list[dict[str, Any]]


def _inject(series: np.ndarray, start: int, end: int, kind: str, magnitude: float) -> None:
    n = end - start
    if n <= 0:
        return
    t = np.linspace(0.0, 1.0, n)
    if kind == "drift":
        dev = magnitude * t
    elif kind == "bias":
        dev = magnitude * (0.6 + 0.4 * t)
    elif kind == "oscillation":
        dev = magnitude * np.sin(np.linspace(0.0, 8.0 * np.pi, n)) * t
    elif kind == "leak":
        dev = magnitude * t
    else:  # pragma: no cover - defensive
        dev = np.zeros(n)
    series[start:end] = series[start:end] + dev


def generate(
    seed: int = 42,
    n_equipment: int = 5,
    days: int = 7,
    interval_minutes: int = 5,
    start: datetime | None = None,
) -> SimulationBundle:
    """Generate the full synthetic fab environment."""
    rng = np.random.default_rng(seed)
    start = start or datetime(2026, 1, 5, 0, 0, 0)
    total_points = int(days * 24 * 60 / interval_minutes)
    freq = f"{interval_minutes}min"
    index = pd.date_range(start=start, periods=total_points, freq=freq)

    equipment_ids = [f"ETCH-{i:02d}" for i in range(1, n_equipment + 1)]

    equipment_rows, reading_rows, alarm_rows, maint_rows, case_rows, cfg_rows, label_rows = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    scenario_meta: list[dict[str, Any]] = []

    for eq_idx, eq_id in enumerate(equipment_ids):
        equipment_rows.append(
            {
                "id": eq_id,
                "code": eq_id,
                "name": f"Etch Chamber {eq_id}",
                "type": "Etch",
                "chamber": "A",
                "manufacturer": "FabPilot-Sim",
                "status": "RUN",
                "description": "Simulated plasma etch tool (synthetic data).",
            }
        )

        # Build per-channel base signals.
        series: dict[str, np.ndarray] = {}
        for c in CHANNELS:
            base, amp, noise = CHANNEL_BASELINE[c]
            phase = rng.uniform(0, 2 * np.pi)
            period = rng.choice([24.0, 12.0, 8.0])  # hours
            t_h = np.arange(total_points) * interval_minutes / 60.0
            series[c] = base + amp * np.sin(2 * np.pi * t_h / period + phase) + rng.normal(0, noise, total_points)

        # Inject scheduled faults.
        scenario_ids: dict[str, int] = {}
        case_counter = 0
        for scenario_name, start_frac, dur_frac in _SCHEDULE.get(eq_id, []):
            sc = SCENARIOS[scenario_name]
            s_idx = int(start_frac * total_points)
            e_idx = min(int((start_frac + dur_frac) * total_points), total_points)
            for channel, (kind, magnitude) in sc.injectors.items():
                _inject(series[channel], s_idx, e_idx, kind, magnitude)

            start_ts = index[s_idx].to_pydatetime()
            end_ts = index[e_idx - 1].to_pydatetime()
            scenario_ids[scenario_name] = case_counter
            case_counter += 1

            scenario_meta.append(
                {
                    "equipment_id": eq_id,
                    "scenario": scenario_name,
                    "start": start_ts,
                    "end": end_ts,
                    "root_cause": sc.root_cause,
                    "affected_channels": list(sc.affected_channels),
                    "alarm_codes": list(sc.alarm_codes),
                }
            )
            label_rows.append(
                {
                    "equipment_id": eq_id,
                    "scenario": scenario_name,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "root_cause": sc.root_cause,
                    "affected_channels": list(sc.affected_channels),
                    "severity": sc.severity,
                }
            )

            # Alarms during the fault window.
            n_alarms = int(rng.integers(2, 5))
            for _ in range(n_alarms):
                a_idx = int(rng.integers(s_idx, e_idx))
                alarm_rows.append(
                    {
                        "equipment_id": eq_id,
                        "ts": index[a_idx].to_pydatetime(),
                        "alarm_code": str(rng.choice(list(sc.alarm_codes))),
                        "alarm_type": sc.name,
                        "severity": "MAJOR" if sc.severity == "MAJOR" else "WARN",
                        "message": sc.description,
                        "acknowledged": int(rng.integers(0, 2)),
                    }
                )

            # Corrective maintenance + a fault case after the fault.
            maint_rows.append(
                {
                    "equipment_id": eq_id,
                    "ts": end_ts + timedelta(hours=1),
                    "maintenance_type": "CORRECTIVE",
                    "description": f"Corrective maintenance for {scenario_name}",
                    "technician": f"tech_{eq_idx + 1}",
                    "downtime_hours": float(rng.uniform(0.5, 2.0)),
                }
            )
            case_rows.append(
                {
                    "equipment_id": eq_id,
                    "title": f"{scenario_name} on {eq_id}",
                    "symptoms": sc.description,
                    "root_cause": sc.root_cause,
                    "resolution": f"Replace/adjust affected component; verify {', '.join(sc.affected_channels)} "
                    "returns to baseline.",
                    "occurred_at": end_ts,
                    "related_alarm_codes": list(sc.alarm_codes),
                    "doc_refs": [f"sop/{scenario_name}.md", f"fault_cases/{scenario_name}.md"],
                }
            )

        # Reading rows.
        for i in range(total_points):
            reading_rows.append(
                {"equipment_id": eq_id, "ts": index[i].to_pydatetime(), **{c: float(series[c][i]) for c in CHANNELS}}
            )

        # Periodic preventive maintenance.
        pm_every = total_points // 6
        for k in range(1, 6):
            pm_idx = k * pm_every
            maint_rows.append(
                {
                    "equipment_id": eq_id,
                    "ts": index[pm_idx].to_pydatetime(),
                    "maintenance_type": "PREVENTIVE",
                    "description": "Scheduled preventive maintenance (chamber clean, O-ring check).",
                    "technician": f"tech_{eq_idx + 1}",
                    "downtime_hours": 1.0,
                }
            )

        # Process configurations.
        cfg_rows.append(
            {
                "equipment_id": eq_id,
                "recipe_name": "ETCH_RECIPE_MAIN",
                "params": {
                    "pressure_setpoint": 30.0,
                    "temperature_setpoint": 60.0,
                    "gas_flow_setpoint": 100.0,
                    "rf_power_setpoint": 500.0,
                },
                "effective_from": start,
                "effective_to": None,
            }
        )

    return SimulationBundle(
        equipment=pd.DataFrame(equipment_rows),
        readings=pd.DataFrame(reading_rows),
        alarms=pd.DataFrame(alarm_rows),
        maintenance=pd.DataFrame(maintenance_rows),
        fault_cases=pd.DataFrame(case_rows),
        process_configs=pd.DataFrame(cfg_rows),
        labels=pd.DataFrame(label_rows),
        scenarios=scenario_meta,
    )
