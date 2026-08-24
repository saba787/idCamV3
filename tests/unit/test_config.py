"""Configuration tests."""

from pathlib import Path

import pytest
import yaml

from idcam.config.loader import load_config
from idcam.config.schema import AppConfig
from idcam.exceptions import ConfigurationError


@pytest.fixture
def valid_config_dict():
    return {
        "device_id": "test-device",
        "server": {"url": "https://api.example.com"},
    }


@pytest.fixture
def config_file(tmp_path, valid_config_dict):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(valid_config_dict))
    return path


class TestConfigSchema:
    def test_valid_config(self, valid_config_dict):
        config = AppConfig.model_validate(valid_config_dict)
        assert config.device_id == "test-device"
        assert str(config.server.url) == "https://api.example.com/"

    def test_missing_device_id(self, valid_config_dict):
        del valid_config_dict["device_id"]
        with pytest.raises(Exception):
            AppConfig.model_validate(valid_config_dict)

    def test_invalid_server_url(self, valid_config_dict):
        valid_config_dict["server"]["url"] = "not-a-url"
        with pytest.raises(Exception):
            AppConfig.model_validate(valid_config_dict)

    def test_path_normalization(self, valid_config_dict):
        valid_config_dict["server"]["path"] = "api/doc"
        config = AppConfig.model_validate(valid_config_dict)
        assert config.server.path == "/api/doc"


class TestConfigLoader:
    def test_load_valid(self, config_file):
        config = load_config(config_file)
        assert config.device_id == "test-device"

    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigurationError, match="not found"):
            load_config(tmp_path / "missing.yaml")

    def test_invalid_yaml(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("{ invalid yaml")
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            load_config(path)

    def test_env_token_override(self, config_file, monkeypatch):
        monkeypatch.setenv("IDCAM_AUTH_TOKEN", "secret-token")
        config = load_config(config_file)
        assert config.auth.resolved_token() == "secret-token"
