# Hybrid matte: measured, not chosen (E1 record)

Hybrid-vs-RVM edge MAE: mean 0.151, p90 0.216 (n=60).
The hybrid moves the edge substantially vs RVM, but E2 shows its
trimap scores == MA2 (miss 0.00039 vs 0.00052, leak 0.00155 vs 0.00161,
IoU identical 0.94274): the movement is lateral, not improvement.
Per §17 + E2 gate (prove beats best single model), hybrid is NOT
promoted to a 3D run. MA2 stays the temporal authority; hybrid + U map
retained as diagnostic only.
