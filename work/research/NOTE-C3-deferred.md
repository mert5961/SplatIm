# C3 (uMatting benchmark) — deferred, not dropped

C1 (MatAnyone2) + C2 (Sapiens2-1B) show all three models agree on certain
foreground/background (recall ~0.996-0.9995, leak ~0.0001); disagreement
lives in the edge band (MAE 0.05-0.15). Per plan-board gate, uMatting is
benchmarked only if C1/C2 leave edge quality open. Decision: defer until
E2 2D benchmark — if E2 shows the MA2/Sapiens edge still leaks, run C3 then.
