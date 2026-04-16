"""HSI ML Pipeline modules.

Submodules pull in heavy ML deps (torch, trimesh, smplx, scipy) so they are
imported on demand by callers, not eagerly here. This keeps the backend
FastAPI process startup cheap — pipeline imports only happen on the
worker thread when a job actually starts.
"""
