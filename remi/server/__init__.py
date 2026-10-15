"""The Remi WebUI server: a FastAPI app serving ``/api/v2`` (REST and the
WebSocket hub), the A1111 ``/sdapi/v1`` API and the React SPA, with the
persistent job queue behind it. See docs/remi-webui/architecture.md."""

from remi.server.app import create_app
from remi.server.services import Services, default_services

__all__ = ["Services", "create_app", "default_services"]
