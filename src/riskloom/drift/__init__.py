"""Score-distribution drift monitoring.

Informational only. Nothing in this package may influence a decision, a threshold, or a model
parameter, and it holds no session and imports no ORM, so it cannot write to any table at all. The
read-only window query lives in ``riskloom.services.drift``; everything here is pure arithmetic
over already-computed aggregates.

Drift observes. It never decides, and it never feeds back.
"""
