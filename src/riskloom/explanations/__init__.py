"""Grounded natural-language explanations for a risk decision.

This package is pure: it imports no ORM, no session, no FastAPI and nothing from the live decision
path. It accepts a typed :class:`ExplanationInput` of already-locked facts and returns a validated
:class:`LlmExplanation`. It has no capability to read or write ``risk_decisions`` -- not by
convention, but because nothing here can reach a database at all.

The LLM explains. It never decides.
"""
