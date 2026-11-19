# Fault Case: Vacuum Leak (ETCH-04)

## Summary
ETCH-04 could not reach its normal base `vacuum_pressure`; pressure rose faster
than spec during idle. Process was unaffected initially.

## Evidence chain
1. Base `vacuum_pressure` elevated ~15% above its historical baseline.
2. Rate-of-rise (leak-up) test exceeded the chamber-volume specification.
3. `gas_flow` and `rf_power` nominal during the early phase.
4. Helium sniffer isolated the leak to a chamber lid O-ring.

## Root cause
Degraded chamber lid O-ring seal.

## Resolution
O-ring replaced; base pressure and leak-up rate returned to specification.

## Lesson
Elevated `vacuum_pressure` is a symptom. A leak-up test (rate-of-rise) is what
distinguishes a leak from pump degradation or outgassing before a root cause is
assigned.
