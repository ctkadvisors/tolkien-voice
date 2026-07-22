from fastapi import FastAPI
import uvicorn


def create_app() -> FastAPI:
    app = FastAPI(title="Tolkien Voice", version="0.1.0")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "service": "tolkien-voice"}

    return app


def run() -> None:
    uvicorn.run(
        "tolkien_voice.main:create_app",
        factory=True,
        host="127.0.0.1",
        port=8788,
    )

