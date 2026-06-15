"""Security layer (Sprint 13): passwords, JWT, principals, RBAC, secrets.

All of it is gated behind ``settings.auth_enabled`` (default off), so the open
development/demo deployment — and the existing test suite — keep working
unchanged. Flip the toggle on for the enterprise posture: JWT auth, role-based
access control, per-tenant isolation and audit logging.
"""

from __future__ import annotations
