from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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
    web_dist: Path | None = None,
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
    if web_dist is not None:
        distribution = web_dist.expanduser().resolve()
        index_path = distribution / "index.html"
        assets_path = distribution / "assets"
        if not index_path.is_file() or not assets_path.is_dir():
            raise RuntimeError(f"production browser bundle is incomplete: {distribution}")
        app.mount("/assets", StaticFiles(directory=assets_path), name="web-assets")

        @app.get("/", include_in_schema=False)
        async def browser_client() -> FileResponse:
            return FileResponse(index_path)

    return app


def run() -> None:
    uvicorn.run(
        create_app(web_dist=ROOT / "web/dist"),
        host="127.0.0.1",
        port=8788,
    )
