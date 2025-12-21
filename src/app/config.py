from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from logger import logger


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


def _read_yaml(path: Path) -> dict:
    """Read a YAML file trying common encodings used in the project."""

    encodings = ("utf-8", "utf-8-sig", "cp1251")
    for enc in encodings:
        try:
            with path.open("r", encoding=enc) as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError("auto", b"", 0, 1, "Unable to decode config in known encodings")


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


@dataclass(frozen=True)
class SyncConfig:
    channel_id: int
    shared_secret: str
    prefix: str = "SYNC "
    ttl_seconds: float = 3600.0


class ConfigManager:
    """Centralized configuration loader/validator."""

    def __init__(self, path: str | Path = "configuration.yml") -> None:
        self.path = Path(path)
        self.data = self._load()

    def _load(self) -> dict:
        try:
            return _read_yaml(self.path)
        except FileNotFoundError:
            logger.warning("Configuration file %s not found; using defaults", self.path)
            return {}

    def _persist(self) -> None:
        _write_yaml(self.path, self.data)

    # ---------------------- Discord ----------------------
    def discord_token(self) -> str:
        env_token = os.getenv("DISCORD_TOKEN")
        cfg_token = (self.data.get("discord") or {}).get("token")
        token = env_token or cfg_token
        if not token:
            raise ConfigError(
                "Discord token is missing. Set DISCORD_TOKEN env var or fill configuration.yml."
            )
        return str(token)

    # ---------------------- ComfyUI ----------------------
    def comfy_instances(self) -> List[dict]:
        comfy = self.data.get("comfyui") or {}
        instances = comfy.get("instances") or []
        if not isinstance(instances, list) or not instances:
            raise ConfigError("At least one comfyui.instances entry must be configured")
        return instances

    def comfy_input_dir(self) -> Path:
        comfy = self.data.get("comfyui") or {}
        raw = comfy.get("input_dir") or "input"
        path = Path(raw)
        if not path.is_absolute():
            path = self.path.parent / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ---------------------- Spoilers ----------------------
    def get_spoiler_tags(self) -> List[str]:
        return list((self.data.get("spoilers") or {}).get("tags", []) or [])

    def set_spoiler_tags(self, tags: Iterable[str]) -> None:
        spoilers = self.data.setdefault("spoilers", {})
        spoilers["tags"] = sorted({str(tag).strip() for tag in tags if str(tag).strip()}, key=str.lower)
        self._persist()

    # ---------------------- Security ----------------------
    def security_blocked(self) -> List[str]:
        return list((self.data.get("security") or {}).get("blocked_users", []) or [])

    def security_donors(self) -> List[str]:
        return list((self.data.get("security") or {}).get("donor_users", []) or [])

    def access_guild_id(self) -> Optional[str]:
        access_id = (self.data.get("security") or {}).get("access_guild_id")
        return str(access_id) if access_id else None

    def supporter_role(self) -> tuple[str, Optional[str]]:
        security = self.data.get("security") or {}
        return security.get("supporter_role_name", "Supporter"), (lambda r: str(r) if r else None)(security.get("supporter_role_id"))

    # ---------------------- Workflows & presets ----------------------
    def workflows(self) -> Dict[str, Any]:
        return self.data.get("workflows", {})

    def resolutions(self) -> List[dict]:
        return list(self.data.get("resolutions") or [])

    def prompt_presets_path(self) -> Optional[Path]:
        path = self.data.get("prompt_presets_file")
        return self._resolve_path(path)

    def model_presets_path(self) -> Optional[Path]:
        path = self.data.get("model_presets_file")
        return self._resolve_path(path)

    def lora_presets_path(self) -> Optional[Path]:
        path = self.data.get("lora_presets_file")
        return self._resolve_path(path)

    def _resolve_path(self, path: Optional[str]) -> Optional[Path]:
        if not path:
            return None
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.path.parent / candidate
        return candidate

    # ---------------------- Sync ----------------------
    def sync_config(self) -> SyncConfig:
        sync_channel = self.data.get("sync", {}) or {}
        channel_id = int(sync_channel.get("channel_id", 1406937891203977358))
        secret = str(sync_channel.get("shared_secret", "fj39f9aj2J#d9a!_38dja0d9@qwe93"))
        prefix = str(sync_channel.get("prefix", "SYNC "))
        ttl = float(sync_channel.get("ttl_seconds", 3600.0))
        return SyncConfig(channel_id=channel_id, shared_secret=secret, prefix=prefix, ttl_seconds=ttl)

    # ---------------------- Helpers ----------------------
    @staticmethod
    def normalize_tag(tag: str) -> str:
        return re.sub(r"\s+", " ", tag.strip()).lower()

    def reload(self) -> None:
        self.data = self._load()
