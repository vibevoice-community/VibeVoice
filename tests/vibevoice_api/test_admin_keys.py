from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_test_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, admin_token: str | None) -> tuple[TestClient, object, object, Path, str | None]:
    keystore_path = tmp_path / "keys.json"
    monkeypatch.setenv("VIBEVOICE_REQUIRE_API_KEY", "1")
    monkeypatch.setenv("VIBEVOICE_KEYSTORE", str(keystore_path))
    if admin_token is None:
        monkeypatch.delenv("VIBEVOICE_ADMIN_TOKEN", raising=False)
    else:
        monkeypatch.setenv("VIBEVOICE_ADMIN_TOKEN", admin_token)

    for mod in [
        "vibevoice_api.server",
        "vibevoice_api.config",
        "vibevoice_api.auth",
    ]:
        sys.modules.pop(mod, None)

    import vibevoice_api.config as config_mod  # noqa: F401  # ensure env is captured
    import vibevoice_api.auth as auth_mod
    import vibevoice_api.server as server_mod

    client = TestClient(server_mod.app)
    return client, server_mod, auth_mod, keystore_path, admin_token


@pytest.fixture()
def admin_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    token = "test-admin-token"
    client, server_mod, auth_mod, keystore_path, _ = _load_test_app(tmp_path, monkeypatch, admin_token=token)
    try:
        yield client, server_mod, auth_mod, keystore_path, token
    finally:
        client.close()


def _admin_headers(token: str | None) -> dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def test_admin_list_requires_token(admin_client):
    client, server_mod, _, _, token = admin_client
    base = server_mod._ADMIN_KEYS_PREFIX

    resp = client.get(base)
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "invalid_admin_token"

    resp = client.get(base, headers=_admin_headers("bad-token"))
    assert resp.status_code == 403
    assert resp.json()["error"]["type"] == "invalid_admin_token"


def test_admin_create_and_delete_flow(admin_client):
    client, server_mod, auth_mod, keystore_path, token = admin_client
    base = server_mod._ADMIN_KEYS_PREFIX
    headers = _admin_headers(token)

    # Initial list is empty
    resp = client.get(base, headers=headers)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["keys"] == []
    assert payload["count"] == 0

    # Auto-generate a key
    resp = client.post(base, headers=headers)
    assert resp.status_code == 201
    created = resp.json()
    assert created["key"].startswith("sk-")
    assert created["hash"] == auth_mod.hash_api_key(created["key"])

    # Import a known key
    manual_key = "sk-test-manual"
    resp = client.post(base, headers=headers, json={"key": manual_key})
    assert resp.status_code == 201
    imported = resp.json()
    assert imported["key"] == manual_key
    assert imported["hash"] == auth_mod.hash_api_key(manual_key)

    # List reflects both hashes
    resp = client.get(base, headers=headers)
    data = resp.json()
    assert sorted(data["keys"]) == sorted([created["hash"], imported["hash"]])
    assert data["count"] == 2

    # Keystore file contains only hashes
    stored = json.loads(keystore_path.read_text())
    assert stored["keys"] == sorted([created["hash"], imported["hash"]])

    # Delete both keys
    resp = client.delete(f"{base}/{created['hash']}", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True

    resp = client.delete(f"{base}/{imported['hash']}", headers=headers)
    assert resp.status_code == 200

    # No keys remain
    resp = client.get(base, headers=headers)
    final = resp.json()
    assert final["keys"] == []
    assert final["count"] == 0

    # Deleting a missing key returns 404
    resp = client.delete(f"{base}/{created['hash']}", headers=headers)
    assert resp.status_code == 404


def test_admin_post_requires_token(admin_client):
    client, server_mod, _, _, _ = admin_client
    base = server_mod._ADMIN_KEYS_PREFIX

    resp = client.post(base)
    assert resp.status_code == 401


def test_admin_delete_requires_token(admin_client):
    client, server_mod, _, _, token = admin_client
    base = server_mod._ADMIN_KEYS_PREFIX
    headers = _admin_headers(token)

    create_resp = client.post(base, headers=headers)
    key_hash = create_resp.json()["hash"]

    resp = client.delete(f"{base}/{key_hash}")
    assert resp.status_code == 401

    resp = client.delete(f"{base}/{key_hash}", headers=_admin_headers("bad-token"))
    assert resp.status_code == 403


def test_admin_disabled_without_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client, server_mod, _, _, _ = _load_test_app(tmp_path, monkeypatch, admin_token=None)
    try:
        base = server_mod._ADMIN_KEYS_PREFIX
        resp = client.get(base, headers=_admin_headers("any"))
        assert resp.status_code == 403
        assert resp.json()["error"]["type"] == "admin_disabled"
    finally:
        client.close()
