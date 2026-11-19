# Fault Case: MFC Gas Flow Drift (ETCH-01)

## Summary
`gas_flow` on ETCH-01 drifted monotonically below its setpoint over a 24-hour
period while `pressure` followed with a lag. Etch rate degraded ~4%.

## Evidence chain
1. `gas_flow` trend: negative linear slope, no oscillation.
2. `pressure` lag-correlates with `gas_flow` (lag ≈ 20 min).
3. `rf_power` and `bias_voltage` stable → plasma not the cause.
4. Maintenance record shows the MFC last calibrated 9 months prior.

## Root cause
MFC ageing / thermal drift of the control element.

## Resolution
Zero/span calibration restored setpoint tracking; MFC flagged for replacement
at the next scheduled PM.

## Lesson
A *leading* flow deviation with a *lagging* pressure deviation isolates the MFC
from the throttle valve / pump loop. Without the lead/lag ordering, the two
channels are merely correlated.
