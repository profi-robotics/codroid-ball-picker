from __future__ import annotations

import app.settings as app_settings


_CODROID_ENV_KEYS = (
    "CODROID_HOST",
    "CODROID_ROBOT_PORT",
    "CODROID_WS_PORT",
    "CODROID_ORIGIN",
    "CODROID_TOKEN",
    "CODROID_USERNAME",
    "CODROID_USER_PASSWORD",
    "CODROID_ROBOT_LOGIN_NAME",
)


def _clear_codroid_env(monkeypatch) -> None:
    for key in _CODROID_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_build_codroid_settings_restores_lab_robot_defaults(monkeypatch) -> None:
    _clear_codroid_env(monkeypatch)
    monkeypatch.setattr(
        app_settings,
        "SETTINGS",
        app_settings.SETTINGS.model_copy(
            update={"robot_host": None, "robot_port": None, "user_port": None}
        ),
    )

    settings = app_settings.build_codroid_settings()

    assert settings.host == "192.168.101.100"
    assert settings.robot_port == 9000
    assert settings.ws_port == 9098
    assert settings.origin == "http://192.168.101.100:9098"
    assert settings.token == "user:admin"
    assert settings.username == "admin"
    assert settings.user_password == "123456"
    assert settings.robot_login_name == "web"
    assert settings.keep_user_web_session is False


def test_build_codroid_settings_ignores_blank_env_credentials(monkeypatch) -> None:
    _clear_codroid_env(monkeypatch)
    monkeypatch.setenv("CODROID_USERNAME", "")
    monkeypatch.setenv("CODROID_TOKEN", "")
    monkeypatch.setenv("CODROID_USER_PASSWORD", "")
    monkeypatch.setenv("CODROID_ROBOT_PORT", "")
    monkeypatch.setenv("CODROID_WS_PORT", "")
    monkeypatch.setattr(
        app_settings,
        "SETTINGS",
        app_settings.AppSettings().model_copy(
            update={"robot_host": None, "robot_port": None, "user_port": None}
        ),
    )

    settings = app_settings.build_codroid_settings()

    assert settings.robot_port == 9000
    assert settings.ws_port == 9098
    assert settings.token == "user:admin"
    assert settings.username == "admin"
    assert settings.user_password == "123456"


def test_build_codroid_settings_prefers_persisted_robot_endpoint(monkeypatch) -> None:
    _clear_codroid_env(monkeypatch)
    monkeypatch.setattr(
        app_settings,
        "SETTINGS",
        app_settings.SETTINGS.model_copy(
            update={
                "robot_host": "10.10.0.5",
                "robot_port": 19000,
                "user_port": 19098,
            }
        ),
    )

    settings = app_settings.build_codroid_settings()

    assert settings.host == "10.10.0.5"
    assert settings.robot_port == 19000
    assert settings.ws_port == 19098
    assert settings.origin == "http://10.10.0.5:19098"


def test_load_codroid_defaults_uses_effective_settings(monkeypatch) -> None:
    _clear_codroid_env(monkeypatch)
    monkeypatch.setattr(
        app_settings,
        "SETTINGS",
        app_settings.SETTINGS.model_copy(
            update={"robot_host": None, "robot_port": None, "user_port": None}
        ),
    )

    defaults = app_settings._load_codroid_defaults()

    assert defaults == {
        "host": "192.168.101.100",
        "robot_port": 9000,
        "ws_port": 9098,
        "robot_ws_url": "ws://192.168.101.100:9000/",
        "user_ws_url": "ws://192.168.101.100:9098/",
    }
