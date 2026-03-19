from __future__ import annotations

import importlib
import os
import subprocess
import sys


def test_cli_help_smoke() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "brainnet.main", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--subject" in result.stdout


def test_cli_describe_plan_smoke(tmp_path) -> None:
    env = os.environ.copy()
    env["BRAINNET_INSTANCE_DIR"] = str(tmp_path / "instance")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "brainnet.main",
            str(tmp_path),
            "--subject",
            "01",
            "--task",
            "rest",
            "--describe-plan",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0
    assert '"mode": "local-bids"' in result.stdout
    assert '"key": "preprocessing"' in result.stdout


def test_web_app_smoke(monkeypatch, tmp_path) -> None:
    instance_dir = tmp_path / "instance"
    monkeypatch.setenv("BRAINNET_INSTANCE_DIR", str(instance_dir))
    sys.modules.pop("brainnet.web_app", None)

    module = importlib.import_module("brainnet.web_app")
    client = module.app.test_client()
    response = client.get("/")

    routes = {rule.rule for rule in module.app.url_map.iter_rules()}

    assert response.status_code == 200
    assert "/analysis" in routes
    assert (instance_dir / "brainnet.db").exists()
    assert (instance_dir / "reports").exists()


def test_chat_route_exists(monkeypatch, tmp_path) -> None:
    instance_dir = tmp_path / "instance"
    monkeypatch.setenv("BRAINNET_INSTANCE_DIR", str(instance_dir))
    sys.modules.pop("brainnet.web_app", None)
    sys.modules.pop("brainnet.web_chat", None)

    module = importlib.import_module("brainnet.web_app")
    client = module.app.test_client()

    routes = {rule.rule for rule in module.app.url_map.iter_rules()}
    assert "/chat" in routes or "/chat/" in routes

    response = client.get("/chat")
    assert response.status_code in (200, 302)


def test_chat_new_session(monkeypatch, tmp_path) -> None:
    instance_dir = tmp_path / "instance"
    monkeypatch.setenv("BRAINNET_INSTANCE_DIR", str(instance_dir))
    sys.modules.pop("brainnet.web_app", None)
    sys.modules.pop("brainnet.web_chat", None)

    module = importlib.import_module("brainnet.web_app")
    client = module.app.test_client()

    response = client.post("/chat/new")
    assert response.status_code == 200
    data = response.get_json()
    assert "session_id" in data
