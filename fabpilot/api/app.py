"""FastAPI app factory.

The service is created lazily on first request so importing / creating the app
never touches the database, knowledge base or LLM. Tests can inject a custom
service via ``create_app(service=...)``.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fabpilot.api.routes import router
from fabpilot.api.service import FabPilotService

_current_service: FabPilotService | None = None


def current_service() -> FabPilotService:
    global _current_service
    if _current_service is None:
        _current_service = FabPilotService()
    return _current_service


def create_app(service: FabPilotService | None = None, title: str = "FabPilot API") -> FastAPI:
    global _current_service
    if service is not None:
        _current_service = service

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Warm the service eagerly if it was injected; otherwise stay lazy.
        if service is not None:
            current_service()
        yield

    app = FastAPI(
        title=title,
        description=(
            "FabPilot — hypothesis-driven multi-agent system for industrial "
            "time-series forecasting and fault diagnosis."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app
