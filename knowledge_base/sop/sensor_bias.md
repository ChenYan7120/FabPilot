# Sensor Bias Detection — Troubleshooting SOP

## Applicability
Any process chamber where a single sensor channel reads consistently offset.

## Background
A biased sensor reports a constant offset from the true physical value. This
differs from noise (random) and drift (monotonic). Bias is typically caused by
zero-point offset, wiring/ground-loop issues, or a damaged sensing element.

## Observable signatures
- One channel is offset from its expected baseline by a constant amount.
- The offset persists across unrelated process conditions.
- Other channels are unaffected, and no physical fault is present.

## Diagnostic procedure
1. Establish the historical baseline mean for the channel.
2. Compute the residual (observed − baseline) and confirm it is *constant*.
3. Cross-check a redundant/derived channel (e.g. compare `pressure` against the
   physically-coupled `vacuum_pressure` or `gas_flow`).
4. If two independent sensors disagree by a constant offset, at least one is
   biased.

## Recommended actions
- Re-zero the sensor.
- Inspect cabling and ground integrity.
- Replace the sensor if zeroing cannot remove the offset.

## Correlation vs causation note
A constant offset in a *single* channel while redundant channels agree is the
strongest evidence of sensor bias. A real process shift would move multiple
coupled channels together.
