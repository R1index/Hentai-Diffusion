from pathlib import Path

from src.app.config import ConfigManager


def test_normalize_tag():
    assert ConfigManager.normalize_tag("  Foo   Bar ") == "foo bar"


def test_resolve_paths(tmp_path: Path, monkeypatch):
    cfg_path = tmp_path / "configuration.yml"
    cfg_path.write_text("comfyui:\n  instances:\n    - url: http://localhost:8188\n")
    cm = ConfigManager(cfg_path)
    # Should create input dir relative to config
    input_dir = cm.comfy_input_dir()
    assert input_dir.exists()
