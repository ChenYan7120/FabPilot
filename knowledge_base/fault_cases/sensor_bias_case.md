# Fault Case: Sensor Bias on pressure (ETCH-02)

## Summary
The `pressure` channel on ETCH-02 read a constant +0.9 Pa offset from its
baseline for two consecutive days, with no physical process change.

## Evidence chain
1. `pressure` residual (observed − baseline) is constant ≈ +0.9 Pa.
2. `vacuum_pressure` (redundant) does NOT show the offset and remains coupled
   to `gas_flow` as expected.
3. No alarms on any other channel.

## Root cause
Zero-point offset of the `pressure` transducer (wiring/ground loop).

## Resolution
Re-zeroing the transducer removed the offset. Cable shield inspected and
re-terminated.

## Lesson
A constant, single-channel offset while a physically-redundant channel agrees
is the fingerprint of sensor bias, not a process shift. A real process shift
would move coupled channels together.
