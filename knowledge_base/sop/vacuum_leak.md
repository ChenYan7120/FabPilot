# Vacuum Leak Detection — Troubleshooting SOP

## Applicability
Etch chambers where base/vacuum pressure fails to reach or hold the target.

## Background
A vacuum leak (seal failure, cracked O-ring, valve seat damage, or a leak in
the gas delivery line) prevents the chamber from reaching its base pressure.
The signature is an elevated `vacuum_pressure` that rises toward atmosphere
when pumping is throttled.

## Observable signatures
- `vacuum_pressure` elevated above its normal base value.
- Pressure rises faster than normal during idle / throttled pumping (leak-up).
- Other channels (`gas_flow`, `rf_power`) remain nominal until the process is
  affected.

## Diagnostic procedure
1. Record base `vacuum_pressure` after a pump-down and compare to history.
2. Perform a rate-of-rise (leak-up) test: isolate the chamber and measure how
   fast pressure climbs.
3. Compare leak-up rate against the specification for the chamber volume.
4. Isolate the leak location (O-rings, valves, gas lines) with a helium sniffer.

## Recommended actions
- Replace suspect O-rings / seals.
- Verify valve seat integrity.
- Helium leak-check the gas delivery lines.

## Correlation vs causation note
Elevated `vacuum_pressure` is a *symptom*; the *cause* (leak vs. pump
degradation vs. outgassing) must be isolated with a leak-up test before
assigning a root cause.
