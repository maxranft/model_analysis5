from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import Settings
from app.db.repository import SQLiteRepository
from app.services.inference import OnnxTriageModel
from app.services.preprocessing import ImagePreprocessor


def create_app(
    settings: Settings | None = None,
    model_service: OnnxTriageModel | None = None,
    repository: SQLiteRepository | None = None,
) -> FastAPI:
    settings = settings or Settings()
    repository = repository or SQLiteRepository(settings.db_path)
    preprocessor = ImagePreprocessor(settings.image_size, settings.max_image_bytes)
    model_service = model_service or OnnxTriageModel(
        model_path=settings.model_path,
        labels=settings.labels,
        model_version=settings.model_version,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        repository.initialize()
        app.state.settings = settings
        app.state.repository = repository
        app.state.preprocessor = preprocessor
        app.state.model_service = model_service
        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()

