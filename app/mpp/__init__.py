"""Mon Petit Prono (MPP) submission package (phase 4).

A self-contained client for the "Mon Petit Prono" REST API that reads game
weeks / matches and submits forecasts autonomously. It replaces the local MCP
server on the submission path.

Auth is an Auth0 refresh-token grant with rotation: the rotating refresh token is
persisted durably in the app cache (serverless-safe, no writable filesystem), so
state survives cold starts on Vercel.

Public surface:

* :class:`MppClient`        - typed read/submit client.
* :func:`get_mpp_client`    - factory returning a ready client.
* :class:`AuthError`        - raised when authentication cannot proceed.
* :class:`MppApiError`      - raised on a non-auth API failure.
"""

from __future__ import annotations

from app.mpp.auth import AuthError
from app.mpp.client import MppApiError, MppClient, get_mpp_client

__all__ = ["AuthError", "MppApiError", "MppClient", "get_mpp_client"]
