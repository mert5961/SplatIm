# Boundary temporal error: RVM vs MatAnyone2 (flow-warped self-consistency)

Method: Farneback flow on RGB, warp alpha(t) -> t+1, edge-band MAE.
59 consecutive pairs of the B5 60-view set.

- RVM self temporal edge MAE: mean 0.478, p90 0.511
- MA2 self temporal edge MAE: mean 0.492, p90 0.517

Verdict: NO temporal improvement of MA2 over RVM on THIS capture.
The flow itself is unreliable here (subject rotates ~360 deg across the
set; consecutive kept views are far apart in orbit angle, so warping the
edge band is mostly out-of-plane motion, not matte jitter). The metric
as constructed measures view-change, not matte stability. Honest gate
result: boundary-temporal gate CANNOT be claimed either way from this
data; needs dense consecutive-frame video (not orbit subset) or
rendered-alpha temporal metric. Recorded, not passed.
