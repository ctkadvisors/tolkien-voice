from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
import uvicorn

from tolkien_voice.api import create_api_router
from tolkien_voice.characters import CharacterRegistry
from tolkien_voice.lease import SparkModelLease
from tolkien_voice.runtime import PersonaPlexRuntime
from tolkien_voice.session import SessionController
from tolkien_voice.settings import RuntimeSettings, load_runtime_settings


ROOT = Path(__file__).parents[2]


def create_app(
    *,
    session_controller: SessionController | None = None,
    character_registry: CharacterRegistry | None = None,
    settings: RuntimeSettings | None = None,
) -> FastAPI:
    runtime_settings = settings or load_runtime_settings(ROOT / "configs/runtime.yaml")
    registry = character_registry or CharacterRegistry.from_directory(
        ROOT / "configs/characters"
    )
    controller = session_controller or SessionController(
        registry=registry,
        lease=SparkModelLease(
            runtime_settings.lock_path,
            runtime_settings.owner_path,
        ),
        runtime=PersonaPlexRuntime(runtime_settings),
        lease_wait_seconds=runtime_settings.lease_wait_seconds,
        disconnect_idle_seconds=runtime_settings.disconnect_idle_seconds,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await controller.initialize()
        yield

    app = FastAPI(title="Tolkien Voice", version="0.1.0", lifespan=lifespan)
    app.state.session_controller = controller

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "tolkien-voice"}

    app.include_router(
        create_api_router(
            controller,
            registry,
            personaplex_url=runtime_settings.personaplex_url,
        )
    )
    return app


def run() -> None:
    uvicorn.run(
        "tolkien_voice.main:create_app",
        factory=True,
        host="127.0.0.1",
        port=8788,
    )
