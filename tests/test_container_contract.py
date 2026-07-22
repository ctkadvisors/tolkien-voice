from pathlib import Path
import re


ROOT = Path(__file__).parents[1]
DOCKERFILE = ROOT / "container" / "Dockerfile.personaplex"
ENTRYPOINT = ROOT / "container" / "entrypoint.sh"
HEALTHCHECK = ROOT / "container" / "healthcheck.py"
BUILD_SCRIPT = ROOT / "scripts" / "build-runtime.sh"
PIN = "3428dfd95309a7f3c84fd93259ded0f810d1ff91"


def test_dockerfile_preserves_the_gb10_pytorch_stack() -> None:
    source = DOCKERFILE.read_text(encoding="utf-8")

    assert source.startswith("FROM nvcr.io/nvidia/pytorch:25.11-py3")
    assert "COPY vendor/personaplex/moshi" in source
    assert "libopus-dev" in source
    assert "pip install --no-deps -e" in source
    assert "torch<" not in source
    assert f'org.opencontainers.image.revision="{PIN}"' in source


def test_image_never_receives_hugging_face_token_at_build_time() -> None:
    source = DOCKERFILE.read_text(encoding="utf-8")

    assert re.search(r"^(ARG|ENV)\s+HF_TOKEN", source, re.MULTILINE) is None
    assert "/run/secrets/hf_token" not in source


def test_entrypoint_uses_only_the_runtime_secret_and_loopback() -> None:
    source = ENTRYPOINT.read_text(encoding="utf-8")

    assert "/run/secrets/hf_token" in source
    assert "--host 127.0.0.1" in source
    assert "--port 8998" in source
    assert "--static none" in source
    assert "--ssl" not in source
    assert "cat /run/secrets/hf_token" not in source


def test_healthcheck_does_not_allocate_a_chat() -> None:
    source = HEALTHCHECK.read_text(encoding="utf-8")

    assert "http://127.0.0.1:8998/" in source
    assert "/api/chat" not in source
    assert "404" in source


def test_build_script_refuses_unreviewed_upstream_revision() -> None:
    source = BUILD_SCRIPT.read_text(encoding="utf-8")

    assert PIN in source
    assert "git -C vendor/personaplex rev-parse HEAD" in source
    assert "vendor/personaplex-revision.txt" in source
    assert "tolkien-voice/personaplex:3428dfd" in source
