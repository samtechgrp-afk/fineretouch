"""Desktop engine: token gate, model discovery and the CLI entry point."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import desktop, main


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setattr(main, "ENGINE_TOKEN", "s3cret")
    return "s3cret"


def test_requests_without_the_engine_token_are_rejected(token):
    client = TestClient(main.app)
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"x-fineretouch-token": "wrong"}).status_code == 401
    assert client.get("/health", headers={"x-fineretouch-token": token}).status_code == 200


def test_rejections_still_carry_cors_headers_for_the_webview(token, monkeypatch):
    client = TestClient(main.app)
    res = client.get("/health", headers={"origin": "http://localhost:3000"})
    assert res.status_code == 401
    assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"
    # Preflights never need the token (browsers cannot attach custom headers to them).
    pre = client.options(
        "/v1/analyze",
        headers={
            "origin": "http://localhost:3000",
            "access-control-request-method": "POST",
            "access-control-request-headers": "x-fineretouch-token",
        },
    )
    assert pre.status_code == 200


def test_no_token_configured_means_open_server_mode(monkeypatch):
    monkeypatch.setattr(main, "ENGINE_TOKEN", None)
    assert TestClient(main.app).get("/health").status_code == 200


def test_models_dir_fills_only_unset_model_variables(tmp_path: Path, monkeypatch):
    for var in ("FACE_LANDMARKER_MODEL", "SEGMENTER_MODEL", "MIDAS_MODEL", "FINERETOUCH_MODELS_DIR"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / "face_landmarker.task").write_bytes(b"x")
    (tmp_path / "midas_small.onnx").write_bytes(b"x")
    monkeypatch.setenv("MIDAS_MODEL", "/custom/midas.onnx")
    desktop.configure_environment(str(tmp_path), str(tmp_path / "data"))
    assert os.environ["FACE_LANDMARKER_MODEL"] == str(tmp_path / "face_landmarker.task")
    assert os.environ["MIDAS_MODEL"] == "/custom/midas.onnx"  # explicit setting wins
    assert "SEGMENTER_MODEL" not in os.environ  # not bundled -> feature degrades gracefully
    assert os.environ["FINERETOUCH_DATA_DIR"] == str(tmp_path / "data")


def test_version_flag(capsys):
    assert desktop.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"fineretouch-engine {desktop.ENGINE_VERSION}"


def test_parent_watchdog_sees_this_process_alive():
    assert desktop._parent_alive(os.getpid())
