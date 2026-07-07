"""Shared, backend-agnostic building blocks reused across ``ml/`` and ``dl/``.

This package exists to host code that both the NumPy traditional-ML path and the
PyTorch deep-learning path genuinely share (metric schema, scaler/split IO), so a
single change propagates to both instead of drifting across duplicated copies.
"""
