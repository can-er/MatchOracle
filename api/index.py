"""Vercel serverless entrypoint (ASGI) — Phase 1 of the Vercel/Supabase migration.

Vercel's Python runtime (@vercel/python) detects and serves the FastAPI ASGI app
exposed here as ``app``. ``vercel.json`` rewrites every path to this function, so
the whole existing FastAPI surface is served unchanged:

    /health · / · /api/v1/* · /metrics · /api/cron/tick

See the vault note "Déploiement (Vercel + Supabase)" for the full plan. The dep
diet (drop celery/prometheus/redis at runtime) and the LLM/DB/MPP swaps land in
phases 2–5.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The `app` package lives at the repo root; make it importable from /api.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.main import app  # noqa: E402  (path setup must precede the import)

__all__ = ["app"]
