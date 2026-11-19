# Fault Case: RF Power Instability (ETCH-03)

## Summary
`rf_power` variance on ETCH-03 increased sharply for ~3 hours and returned to
normal, correlated with intermittent `vacuum_pressure` spikes.

## Evidence chain
1. Rolling std of `rf_power` exceeded 3σ during the window.
2. `bias_voltage` tracks `rf_power` closely (expected plasma-sheath coupling).
3. Intermittent `vacuum_pressure` spikes coincide with `rf_power` excursions →
   arcing signature.
4. Maintenance record: match network last tuned 6 months prior.

## Root cause
Polymer build-up in the chamber caused intermittent arcing and match-network
instability.

## Resolution
Chamber clean; match network re-tuned. Instability did not recur.

## Lesson
`rf_power` + `bias_voltage` instability *with* `vacuum_pressure` spikes is an
arcing signature, not a single-sensor fault.
