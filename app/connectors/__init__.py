"""Data connectors (Sprint 08).

Adapters that bring real external data into the platform. The first concrete
connector is :class:`~app.connectors.openligadb.OpenLigaDBConnector` (football).
"""

from app.connectors.base import BaseConnector

__all__ = ["BaseConnector"]
