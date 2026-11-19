# RF Power Instability — Troubleshooting SOP

## Applicability
Etch chambers where delivered RF power fluctuates around its setpoint.

## Background
RF power instability is usually caused by an impedance mismatch in the match
network, an aging RF generator, or arcing in the chamber. It appears as
high-variance oscillation of `rf_power` (and the coupled `bias_voltage`),
rather than a clean step or monotonic drift.

## Observable signatures
- `rf_power` variance increases sharply relative to baseline.
- `bias_voltage` fluctuates in lock-step with `rf_power`.
- Possible intermittent `vacuum_pressure` spikes if arcing occurs.
- Etch rate and CD uniformity degrade.

## Diagnostic procedure
1. Compute the rolling standard deviation of `rf_power`; flag windows above a
   z-score threshold.
2. Check whether `bias_voltage` tracks `rf_power` (expected coupling).
3. Look for correlated `vacuum_pressure` spikes — these indicate arcing.
4. Review maintenance records for match-network or generator service history.

## Recommended actions
- Verify the match-network tuning and cable condition.
- Inspect the chamber for polymer build-up (arc precursor).
- Replace the RF generator if it cannot hold the setpoint.

## Correlation vs causation note
`rf_power` and `bias_voltage` are strongly coupled by the plasma sheath;
simultaneous instability in both is consistent with a genuine RF delivery
problem rather than a single-sensor fault.
