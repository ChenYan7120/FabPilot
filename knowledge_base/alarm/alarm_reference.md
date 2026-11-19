# Alarm Interpretation Reference

FabPilot's synthetic fab environment raises alarms on physical channels when a
rolling z-score exceeds a threshold. This reference explains how to read them
and what each channel's alarm usually implies.

## Channel → typical meaning

| Channel | Typical alarm meaning |
|---------|-----------------------|
| `gas_flow` | MFC drift / setpoint deviation, or upstream gas supply issue |
| `pressure` | Throttle valve, pump, or gas-flow coupling issue |
| `temperature` | Chamber thermal drift, heater/ESC fault |
| `rf_power` | Match-network / generator instability, arcing |
| `bias_voltage` | Plasma-sheath / wafer chuck coupling issue |
| `vacuum_pressure` | Leak, pump degradation, outgassing |
| `helium_flow` | Backside helium leak (wafer chuck) |

## Reading an alarm correctly

- A *single* channel alarming usually points to a channel-specific fault
  (sensor or the subsystem it measures).
- *Coupled* channels alarming together (e.g. `rf_power` + `bias_voltage`, or
  `gas_flow` + `pressure`) usually points to a shared physical root cause.
- Always pull the maintenance and fault-case history for the equipment before
  assigning a root cause — correlation is not causation.

## Priority
- High: `vacuum_pressure`, `rf_power` (risk of wafer scrap / tool damage).
- Medium: `gas_flow`, `pressure` (process drift).
- Low: `temperature`, `helium_flow` (gradual degradation).
