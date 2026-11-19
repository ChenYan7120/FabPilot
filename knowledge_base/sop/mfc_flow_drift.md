# MFC Gas Flow Drift — Troubleshooting SOP

## Applicability
Etch process chambers where mass flow controller (MFC) gas flow drifts over time.

## Background
An MFC regulates gas flow to a setpoint. Mechanical wear, contamination of the
laminar flow element, or thermal drift of the control electronics causes the
actual delivered flow to deviate from the commanded setpoint. A slow, monotonic
drift (rather than a step change) is the classic signature of MFC ageing.

## Observable signatures
- `gas_flow` slowly drifts away from its setpoint over hours/days.
- `pressure` tracks the flow drift (flow and pressure are coupled through the
  throttle valve), but lags behind.
- No corresponding alarm, and `rf_power` / `bias_voltage` remain stable.
- Process metrics (etch rate, uniformity) degrade gradually.

## Diagnostic procedure
1. Confirm the drift is monotonic, not oscillatory — plot `gas_flow` vs time.
2. Cross-check `pressure`: a *leading* flow change with a *lagging* pressure
   change points to the MFC, not the throttle valve or pump.
3. Inspect the MFC zero/span calibration record (maintenance history).
4. If the drift is step-like, suspect a setpoint command fault or a valve
   sticking, not ageing.

## Recommended actions
- Perform MFC zero/span calibration.
- Replace the MFC if calibration cannot restore setpoint tracking.
- Verify the upstream pressure regulator and gas filter.

## Correlation vs causation note
Flow and pressure are *correlated* by physics, but a pressure anomaly does not
*prove* an MFC fault. Only the lead/lag ordering and calibration history
establish causation.
