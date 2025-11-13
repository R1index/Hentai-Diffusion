from dataclasses import dataclass
from typing import Optional, Tuple

import discord

from logger import logger

__all__ = ["SecurityManager", "SecurityResult", "BasicSecurity"]


@dataclass(slots=True)
class SecurityResult:
    """Represents the outcome of a security check."""

    state: bool
    message: str = ""


class SecurityManager:
    """Manages security permissions for workflows and settings."""

    @staticmethod
    def _normalise_member_identifiers(member: discord.abc.User) -> Tuple[set[str], set[str]]:
        """Return lowercase identifiers and role names for a Discord user/member."""

        identifiers = {
            str(member.id),
            getattr(member, "name", ""),
            getattr(member, "global_name", "") or "",
            getattr(member, "display_name", "") or "",
        }

        identifiers = {identifier.strip().lower() for identifier in identifiers if identifier}

        role_names: set[str] = set()
        roles = getattr(member, "roles", None)
        if roles:
            role_names = {
                getattr(role, "name", "").strip().lower()
                for role in roles
                if getattr(role, "name", None)
            }

        return identifiers, role_names

    def _check_user_permissions(
        self, member: discord.abc.User, security_config: dict
    ) -> bool:
        """Check if the user satisfies the security configuration."""

        if not security_config or not security_config.get("enabled", False):
            return True

        identifiers, role_names = self._normalise_member_identifiers(member)

        allowed_users = {
            str(user).strip().lower() for user in security_config.get("allowed_users", []) if user
        }
        if identifiers & allowed_users:
            return True

        allowed_roles = {
            str(role).strip().lower() for role in security_config.get("allowed_roles", []) if role
        }
        if role_names & allowed_roles:
            return True

        return False

    def check_workflow_access(
        self, member: discord.abc.User, workflow_name: str, workflow_config: dict
    ) -> bool:
        """Check if a user has access to a workflow."""

        security_config = workflow_config.get("security", {})
        return self._check_user_permissions(member, security_config)

    def check_setting_access(
        self, member: discord.abc.User, workflow_config: dict, setting_name: str
    ) -> bool:
        """Check if a user has access to a workflow setting."""

        if setting_name in {"__before", "__after"}:
            return True  # System settings are always allowed

        settings = workflow_config.get("settings", [])
        setting_config = next((s for s in settings if s.get("name") == setting_name), None)

        if not setting_config:
            return False

        security_config = setting_config.get("security", {})
        return self._check_user_permissions(member, security_config)

    def validate_settings_string(
        self,
        member: discord.abc.User,
        workflow_config: dict,
        settings_str: Optional[str],
    ) -> Tuple[bool, str]:
        """Validate all settings contained in a semicolon separated string."""

        if not settings_str:
            return True, ""

        settings_list = [setting.strip() for setting in settings_str.split(";") if setting.strip()]

        for setting in settings_list:
            setting_name = setting.split("(")[0].strip()

            if not self.check_setting_access(member, workflow_config, setting_name):
                return False, f"You don't have permission to use the '{setting_name}' setting"

        return True, ""


class BasicSecurity:
    """Default implementation of the security hook."""

    def __init__(self, bot):
        self.security_manager = bot.security_manager
        bot.hook_manager.register_hook("is.security", self.check_security)

    async def check_security(
        self,
        interaction: discord.Interaction,
        workflow_name: str,
        workflow_type: str,
        prompt: str,
        workflow_config: dict,
        settings: Optional[str] = None,
    ) -> SecurityResult:
        """Check whether the user can run the workflow with the provided settings."""

        try:
            if not self.security_manager.check_workflow_access(
                interaction.user, workflow_name, workflow_config
            ):
                return SecurityResult(
                    False, f"You don't have permission to use the '{workflow_name}' workflow"
                )

            if settings:
                valid, error_msg = self.security_manager.validate_settings_string(
                    interaction.user, workflow_config, settings
                )

                if not valid:
                    return SecurityResult(False, error_msg)

            return SecurityResult(True)

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Error in security check: {exc}")
            return SecurityResult(False, "An error occurred while checking permissions")
