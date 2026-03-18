import json
import random
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from logger import logger


@dataclass(frozen=True)
class ModelPreset:
    name: str
    model: str
    value: str


@dataclass(frozen=True)
class PromptPreset:
    name: str
    tags: str
    value: str

    def apply(self, prompt: Optional[str]) -> str:
        """Prepend the preset tags to the provided prompt."""

        base_prompt = prompt.strip() if prompt else ""
        prefix = self.tags.strip()

        if not prefix:
            return base_prompt

        return f"{prefix} {base_prompt}".strip()


@dataclass(frozen=True)
class LoRAPreset:
    name: str
    lora: str
    value: str


class WorkflowManager:
    """Manages ComfyUI workflows and their configurations"""
    def __init__(self, config_path: str):
        self.config_path = Path(config_path).resolve()
        self.config = self._load_config(config_path)
        self.workflows = self.config['workflows']
        self.default_workflow = self.config.get('default_workflow')
        self._resolution_presets: List[Tuple[str, str]] = self._parse_resolution_presets(
            self.config.get('resolutions')
        )
        self._model_presets: List[ModelPreset] = self._load_model_presets(
            self.config.get('model_presets_file')
        )
        self._lora_presets: List[LoRAPreset] = self._load_lora_presets(
            self.config.get('lora_presets_file')
        )
        self._prompt_presets: List[PromptPreset] = self._load_prompt_presets(
            self.config.get('prompt_presets_file')
        )
        self._model_preset_lookup: Dict[str, ModelPreset] = {}
        self._model_preset_lookup_by_name: Dict[str, ModelPreset] = {}
        self._model_preset_order: Dict[str, int] = {}
        for idx, preset in enumerate(self._model_presets):
            self._model_preset_lookup[preset.value] = preset
            self._model_preset_lookup_by_name[preset.name.lower()] = preset
            self._model_preset_order[preset.value] = idx
        self._prompt_preset_lookup: Dict[str, PromptPreset] = {}
        self._prompt_preset_lookup_by_name: Dict[str, PromptPreset] = {}
        self._prompt_preset_order: Dict[str, int] = {}
        for idx, preset in enumerate(self._prompt_presets):
            self._prompt_preset_lookup[preset.value] = preset
            self._prompt_preset_lookup_by_name[preset.name.lower()] = preset
            self._prompt_preset_order[preset.value] = idx
        self._lora_preset_lookup: Dict[str, LoRAPreset] = {}
        self._lora_preset_lookup_by_name: Dict[str, LoRAPreset] = {}
        self._lora_preset_order: Dict[str, int] = {}
        for idx, preset in enumerate(self._lora_presets):
            self._lora_preset_lookup[preset.value] = preset
            self._lora_preset_lookup_by_name[preset.name.lower()] = preset
            self._lora_preset_order[preset.value] = idx

        self._config_dir = self.config_path.parent

        # Get ComfyUI input directory from config
        self.input_dir = Path(self.config.get('comfyui', {}).get('input_dir', 'input'))
        if not self.input_dir.is_absolute():
            # If relative path, make it relative to the config file location
            self.input_dir = self._config_dir / self.input_dir

        # Ensure input directory exists
        self.input_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using ComfyUI input directory: {self.input_dir}")

    def _parse_resolution_presets(self, raw_presets: Optional[list]) -> List[Tuple[str, str]]:
        presets: List[Tuple[str, str]] = []
        if not raw_presets:
            return presets

        for item in raw_presets:
            if isinstance(item, dict):
                value = item.get('value') or item.get('label') or item.get('name')
                label = item.get('label') or item.get('name') or value
            else:
                value = item
                label = item

            if not value:
                continue

            presets.append((str(label), str(value)))

        return presets

    def _load_prompt_presets(self, presets_path: Optional[str]) -> List[PromptPreset]:
        """Load prompt presets from a JSON file."""

        if not presets_path:
            logger.info("Prompt presets file not configured; skipping presets")
            return []

        path = Path(presets_path)
        if not path.is_absolute():
            path = self.config_path.parent / path

        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_presets = json.load(f)
        except FileNotFoundError:
            logger.info("Prompt presets file %s not found; no presets loaded", path)
            return []
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse prompt presets file %s: %s", path, exc)
            return []

        presets: List[PromptPreset] = []

        def add_preset(name: Optional[str], tags: Optional[object], value: Optional[str] = None) -> None:
            if not name or tags is None:
                return

            if isinstance(tags, list):
                tags_str = ", ".join(str(tag).strip() for tag in tags if str(tag).strip())
            else:
                tags_str = str(tags).strip()

            if not tags_str:
                return

            preset_value = value or self._slugify_value(name)
            presets.append(PromptPreset(str(name), tags_str, preset_value))

        if isinstance(raw_presets, dict):
            for name, value in raw_presets.items():
                if isinstance(value, dict):
                    add_preset(value.get('name') or name, value.get('tags'), value.get('value'))
                else:
                    add_preset(name, value)
        elif isinstance(raw_presets, list):
            for idx, item in enumerate(raw_presets):
                if isinstance(item, dict):
                    add_preset(item.get('name') or item.get('title'), item.get('tags'), item.get('value'))
                else:
                    add_preset(f"Preset {idx + 1}", item)

        if presets:
            logger.info("Loaded %d prompt presets from %s", len(presets), path)
        else:
            logger.info("No valid prompt presets found in %s", path)

        return presets

    def _load_model_presets(self, presets_path: Optional[str]) -> List[ModelPreset]:
        """Load model presets from a JSON file."""

        if not presets_path:
            logger.info("Model presets file not configured; skipping presets")
            return []

        path = Path(presets_path)
        if not path.is_absolute():
            path = self.config_path.parent / path

        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_presets = json.load(f)
        except FileNotFoundError:
            logger.info("Model presets file %s not found; no presets loaded", path)
            return []
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse model presets file %s: %s", path, exc)
            return []

        presets: List[ModelPreset] = []

        def add_preset(name: Optional[str], model: Optional[str], value: Optional[str] = None) -> None:
            if not name or not model:
                return

            model_name = str(model).strip()
            if not model_name:
                return

            preset_value = value or self._slugify_value(name)
            presets.append(ModelPreset(str(name), model_name, preset_value))

        if isinstance(raw_presets, dict):
            for name, value in raw_presets.items():
                if isinstance(value, dict):
                    add_preset(value.get('name') or name, value.get('model'), value.get('value'))
                else:
                    add_preset(name, value)
        elif isinstance(raw_presets, list):
            for idx, item in enumerate(raw_presets):
                if isinstance(item, dict):
                    add_preset(item.get('name') or item.get('title'), item.get('model'), item.get('value'))
                else:
                    add_preset(f"Model {idx + 1}", item)

        if presets:
            logger.info("Loaded %d model presets from %s", len(presets), path)
        else:
            logger.info("No valid model presets found in %s", path)

        return presets

    def _load_lora_presets(self, presets_path: Optional[str]) -> List[LoRAPreset]:
        """Load LoRA presets from a JSON file."""

        if not presets_path:
            logger.info("LoRA presets file not configured; skipping presets")
            return []

        path = Path(presets_path)
        if not path.is_absolute():
            path = self.config_path.parent / path

        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw_presets = json.load(f)
        except FileNotFoundError:
            logger.info("LoRA presets file %s not found; no presets loaded", path)
            return []
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LoRA presets file %s: %s", path, exc)
            return []

        presets: List[LoRAPreset] = []

        def add_preset(name: Optional[str], lora: Optional[str], value: Optional[str] = None) -> None:
            if not name or not lora:
                return

            lora_name = str(lora).strip()
            if not lora_name:
                return

            preset_value = value or self._slugify_value(name)
            presets.append(LoRAPreset(str(name), lora_name, preset_value))

        if isinstance(raw_presets, dict):
            for name, value in raw_presets.items():
                if isinstance(value, dict):
                    add_preset(value.get('name') or name, value.get('lora'), value.get('value'))
                else:
                    add_preset(name, value)
        elif isinstance(raw_presets, list):
            for idx, item in enumerate(raw_presets):
                if isinstance(item, dict):
                    add_preset(item.get('name') or item.get('title'), item.get('lora'), item.get('value'))
                else:
                    add_preset(f"LoRA {idx + 1}", item)

        if presets:
            logger.info("Loaded %d LoRA presets from %s", len(presets), path)
        else:
            logger.info("No valid LoRA presets found in %s", path)

        return presets

    def _slugify_value(self, name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        return slug or uuid.uuid4().hex

    def get_prompt_presets(self) -> List[PromptPreset]:
        """Expose loaded prompt presets."""

        return list(self._prompt_presets)

    def get_model_presets(self) -> List[ModelPreset]:
        """Expose loaded model presets."""

        return list(self._model_presets)

    def search_prompt_presets(self, query: str = "", *, limit: int = 25) -> List[PromptPreset]:
        """Return prompt presets filtered by query."""

        normalized = (query or "").strip().lower()
        if not normalized:
            return self._prompt_presets[:limit]

        def score(preset: PromptPreset) -> tuple[int, int]:
            name_l = preset.name.lower()
            tags_l = preset.tags.lower()
            if name_l.startswith(normalized):
                priority = 3
            elif normalized in name_l:
                priority = 2
            elif normalized in tags_l:
                priority = 1
            else:
                priority = 0
            return (-priority, self._prompt_preset_order.get(preset.value, 0))

        scored = [
            (score(preset), preset)
            for preset in self._prompt_presets
            if normalized in preset.name.lower() or normalized in preset.tags.lower()
        ]
        if not scored:
            return self._prompt_presets[:limit]

        scored.sort(key=lambda item: item[0])
        return [preset for _, preset in scored[:limit]]

    def apply_prompt_preset(
        self,
        preset_value: Optional[str],
        prompt: Optional[str],
    ) -> tuple[str, Optional[str], Optional[str]]:
        """Return prompt updated with preset tags, plus preset name and tags."""

        if not preset_value:
            return prompt or "", None, None

        normalized_value = preset_value.strip().lower()
        preset = (
            self._prompt_preset_lookup.get(preset_value)
            or self._prompt_preset_lookup.get(self._slugify_value(normalized_value))
            or self._prompt_preset_lookup_by_name.get(normalized_value)
        )
        if not preset:
            logger.warning("Prompt preset '%s' not found; using original prompt", preset_value)
            return prompt or "", None, None

        combined = preset.apply(prompt)
        return combined, preset.name, preset.tags

    def search_model_presets(self, query: str = "", *, limit: int = 25) -> List[ModelPreset]:
        """Return model presets filtered by query."""

        normalized = (query or "").strip().lower()
        if not normalized:
            return self._model_presets[:limit]

        def score(preset: ModelPreset) -> tuple[int, int]:
            name_l = preset.name.lower()
            model_l = preset.model.lower()
            if name_l.startswith(normalized):
                priority = 3
            elif normalized in name_l:
                priority = 2
            elif normalized in model_l:
                priority = 1
            else:
                priority = 0
            return (-priority, self._model_preset_order.get(preset.value, 0))

        scored = [
            (score(preset), preset)
            for preset in self._model_presets
            if normalized in preset.name.lower() or normalized in preset.model.lower()
        ]
        if not scored:
            return self._model_presets[:limit]

        scored.sort(key=lambda item: item[0])
        return [preset for _, preset in scored[:limit]]

    def apply_model_preset(
        self,
        preset_value: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """Return model name and preset name for the given preset value."""

        if not preset_value:
            return None, None

        normalized_value = preset_value.strip().lower()
        preset = (
            self._model_preset_lookup.get(preset_value)
            or self._model_preset_lookup.get(self._slugify_value(normalized_value))
            or self._model_preset_lookup_by_name.get(normalized_value)
        )
        if not preset:
            logger.warning("Model preset '%s' not found; ignoring", preset_value)
            return None, None

        return preset.model, preset.name

    def get_lora_presets(self) -> List[LoRAPreset]:
        """Expose loaded LoRA presets."""

        return list(self._lora_presets)

    def search_lora_presets(self, query: str = "", *, limit: int = 25) -> List[LoRAPreset]:
        """Return LoRA presets filtered by query."""

        normalized = (query or "").strip().lower()
        if not normalized:
            return self._lora_presets[:limit]

        def score(preset: LoRAPreset) -> tuple[int, int]:
            name_l = preset.name.lower()
            lora_l = preset.lora.lower()
            if name_l.startswith(normalized):
                priority = 3
            elif normalized in name_l:
                priority = 2
            elif normalized in lora_l:
                priority = 1
            else:
                priority = 0
            return (-priority, self._lora_preset_order.get(preset.value, 0))

        scored = [
            (score(preset), preset)
            for preset in self._lora_presets
            if normalized in preset.name.lower() or normalized in preset.lora.lower()
        ]
        if not scored:
            return self._lora_presets[:limit]

        scored.sort(key=lambda item: item[0])
        return [preset for _, preset in scored[:limit]]

    def apply_lora_preset(
        self,
        preset_value: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """Return LoRA model name and preset name for the given preset value."""

        if not preset_value:
            return None, None

        normalized_value = preset_value.strip().lower()
        preset = (
            self._lora_preset_lookup.get(preset_value)
            or self._lora_preset_lookup.get(self._slugify_value(normalized_value))
            or self._lora_preset_lookup_by_name.get(normalized_value)
        )
        if not preset:
            logger.warning("LoRA preset '%s' not found; ignoring", preset_value)
            return None, None

        return preset.lora, preset.name

    def get_resolution_presets(self) -> List[Tuple[str, str]]:
        """Return configured resolution presets as (label, value) tuples."""

        return list(self._resolution_presets)

    def is_resolution_allowed(self, resolution: str) -> bool:
        """Return True if resolution is in configured presets or presets are not set."""

        if not self._resolution_presets:
            return True

        return any(value == resolution for _, value in self._resolution_presets)

    def update_workflow_nodes(self, workflow_json: dict, workflow_config: dict,
                              prompt: str = None, image_data: bytes = None) -> dict:
        """Update workflow nodes with prompt and/or image data"""
        modified_workflow = workflow_json.copy()

        # Update prompt if provided and node is configured
        if prompt and 'text_prompt_node_id' in workflow_config:
            node_id = str(workflow_config['text_prompt_node_id'])
            if node_id in modified_workflow:
                node = modified_workflow[node_id]
                if 'inputs' in node and 'text' in node['inputs']:
                    node['inputs']['text'] = prompt
                    logger.debug(f"Updated prompt in node {node_id}: {prompt}")

        # Update image if provided and node is configured
        if image_data and 'image_input_node_id' in workflow_config:
            file_path: Optional[Path] = None
            try:
                node_id = str(workflow_config['image_input_node_id'])
                if node_id not in modified_workflow:
                    raise ValueError(f"Node ID {node_id} not found in workflow")

                # Create a unique filename
                filename = f"input_{uuid.uuid4()}.png"
                file_path = self.input_dir / filename

                # Save the image
                file_path.write_bytes(image_data)
                logger.debug(f"Saved input image to: {file_path}")

                # Update node with image path
                node = modified_workflow[node_id]
                if 'inputs' in node and 'image' in node['inputs']:
                    class_type = str(node.get('class_type', ''))

                    # VHS_LoadImagePath expects a full file path string.
                    # Standard ComfyUI LoadImage-like nodes expect only a filename.
                    if class_type == 'VHS_LoadImagePath':
                        image_value = str(file_path)
                        logger.debug(
                            "Updated image in node %s (%s) with full path: %s",
                            node_id,
                            class_type,
                            image_value,
                        )
                    else:
                        image_value = filename
                        logger.debug(
                            "Updated image in node %s (%s) with filename: %s",
                            node_id,
                            class_type or 'unknown',
                            image_value,
                        )

                    node['inputs']['image'] = image_value
                else:
                    raise ValueError(f"Node {node_id} does not have 'image' input")

            except Exception as e:
                logger.error(f"Error updating image node: {e}")
                if file_path and file_path.exists():
                    try:
                        file_path.unlink()  # Clean up the file if there was an error
                    except:
                        pass
                raise ValueError(f"Failed to process input image: {str(e)}")

        return modified_workflow

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            # Если файл с BOM
            with open(config_path, 'r', encoding='utf-8-sig') as f:
                return yaml.safe_load(f)

    def get_workflow(self, name: str) -> dict:
        """Get workflow configuration by name"""
        return self.workflows.get(name, {})

    def get_selectable_workflows(self, workflow_type: str = None) -> Dict[str, dict]:
        """Get all workflows that are marked as selectable and match the specified type"""
        workflows = {k: v for k, v in self.workflows.items()
                     if v.get('selectable', True)}

        if workflow_type:
            workflows = {k: v for k, v in workflows.items()
                         if v.get('type', 'txt2img') == workflow_type}

        return workflows

    def get_default_workflow(self, workflow_type: str) -> str:
        """Get default workflow for the specified type"""
        for name, workflow in self.workflows.items():
            if workflow.get('type', 'txt2img') == workflow_type and workflow.get('default', False):
                return name
        # Return first workflow of the specified type if no default is set
        for name, workflow in self.workflows.items():
            if workflow.get('type', 'txt2img') == workflow_type:
                return name
        return None

    def load_workflow_file(self, workflow_path: str) -> dict:
        """Load workflow JSON file."""

        path = Path(workflow_path)
        if not path.is_absolute():
            path = self._config_dir / path

        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _apply_setting(self, workflow_json: dict, setting_name: str, setting_def: dict, params: list[str] = None):
        """Apply a single setting to the workflow"""
        try:
            if 'code' in setting_def:
                code = setting_def['code']
                # Create function from code string
                exec(code)
                if params:
                    locals()[setting_name](workflow_json, *params)
                else:
                    locals()[setting_name](workflow_json)
                logger.debug(f"Applied setting: {setting_name}")
        except Exception as e:
            logger.error(f"Error applying setting {setting_name}: {e}")

    def _find_setting_def(self, workflow: dict, setting_name: str) -> Optional[dict]:
        """Find setting definition in workflow settings"""
        if 'settings' not in workflow:
            return None

        for setting_def in workflow['settings']:
            if setting_def.get('name') == setting_name:
                return setting_def
        return None

    def apply_settings(self, workflow_json: dict, workflow_config: dict, settings_str: str = None) -> dict:
        """Apply settings to a workflow including __before and __after"""
        workflow = workflow_config

        if not workflow:
            return workflow_json

        try:
            # Apply __before settings if they exist
            before_setting = self._find_setting_def(workflow, '__before')
            if before_setting:
                logger.debug("Applying __before settings...")
                self._apply_setting(workflow_json, '__before', before_setting)

            # Apply custom settings if provided
            if settings_str:
                settings_list = settings_str.split(';')
                for setting in settings_list:
                    if not setting:
                        continue

                    # Parse setting name and parameters
                    if '(' in setting and ')' in setting:
                        func_name = setting.split('(')[0]
                        params_str = setting[len(func_name) + 1:-1]
                        params = [p.strip() for p in params_str.split(',') if p.strip()]
                    else:
                        func_name = setting
                        params = []

                    # Find and apply the setting
                    setting_def = self._find_setting_def(workflow, func_name)
                    if setting_def:
                        self._apply_setting(workflow_json, func_name, setting_def, params)
                    else:
                        logger.warning(f"Setting '{func_name}' not found in workflow configuration")

            # Apply __after settings if they exist
            after_setting = self._find_setting_def(workflow, '__after')
            if after_setting:
                logger.debug("Applying __after settings...")
                self._apply_setting(workflow_json, '__after', after_setting)

            return workflow_json

        except Exception as e:
            logger.error(f"Error applying settings: {e}")
            return workflow_json

    def apply_resolution(self, workflow_json: dict, workflow_config: dict, workflow_name: str,
                         resolution: Optional[str]) -> dict:
        """Apply the selected resolution to the workflow if supported."""

        if not resolution:
            return workflow_json

        node_id = workflow_config.get('resolution_node_id')
        if node_id is None:
            logger.debug(
                "Workflow '%s' does not define 'resolution_node_id'; skipping resolution override",
                workflow_name,
            )
            return workflow_json

        node_key = str(node_id)
        node = workflow_json.get(node_key)
        if not node:
            logger.warning(
                "Resolution node '%s' not found in workflow '%s'",
                node_key,
                workflow_name,
            )
            return workflow_json

        inputs = node.setdefault('inputs', {})
        if not self.is_resolution_allowed(resolution):
            logger.debug(
                "Resolution '%s' is not in configured presets; applying regardless",
                resolution,
            )

        # Different workflow nodes expect different input names for resolution.
        # Examples:
        # - Some nodes read plain "resolution"
        # - `SDXL Empty Latent Image (rgthree)` reads "dimensions" and validates
        #   it strictly against a fixed list (including exact spacing).
        if 'resolution' in inputs:
            inputs['resolution'] = resolution

        normalized_resolution = str(resolution).strip().lower()
        resolution_match = re.search(r'(\d+)\s*x\s*(\d+)', normalized_resolution)
        if resolution_match:
            width = int(resolution_match.group(1))
            height = int(resolution_match.group(2))

            if width == height:
                orientation = "square"
            elif width > height:
                orientation = "landscape"
            else:
                orientation = "portrait"

            # Keep formatting compatible with rgthree enum values.
            # Examples:
            # - "1216 x 832   (landscape)"
            # - " 832 x 1216  (portrait)"
            # - "1024 x 1024  (square)"
            if 'dimensions' in inputs:
                inputs['dimensions'] = f"{width:>4} x {height:<4}  ({orientation})"

            if 'width' in inputs:
                inputs['width'] = width
            if 'height' in inputs:
                inputs['height'] = height

        logger.debug(
            "Applied resolution '%s' to node '%s' in workflow '%s'",
            resolution,
            node_key,
            workflow_name,
        )
        return workflow_json

    def _find_seed_node(self, workflow_json: dict) -> Optional[str]:
        """Attempt to find the first node that accepts a 'seed' input."""

        for key, node in workflow_json.items():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs")
            if isinstance(inputs, dict) and "seed" in inputs:
                return key
        return None

    def apply_seed(self, workflow_json: dict, workflow_config: dict, workflow_name: str,
                   seed: Optional[int]) -> dict:
        """Apply a fixed or random seed."""

        node_id = workflow_config.get("seed_node_id")
        if node_id is None:
            node_id = self._find_seed_node(workflow_json)

        if node_id is None:
            logger.debug("No seed node found for workflow '%s'; skipping seed override", workflow_name)
            return workflow_json

        node_key = str(node_id)
        node = workflow_json.get(node_key)
        if not node or "inputs" not in node:
            logger.debug("Seed node '%s' missing or has no inputs in workflow '%s'", node_key, workflow_name)
            return workflow_json

        seed_to_apply = int(seed) if seed is not None else random.randint(0, 2**32 - 1)
        node["inputs"]["seed"] = seed_to_apply
        logger.debug("Applied seed '%s' to node '%s' in workflow '%s'", seed_to_apply, node_key, workflow_name)
        return workflow_json


    def apply_controlnet_strength(self, workflow_json: dict, workflow_config: dict, workflow_name: str,
                                  controlnet_strength: Optional[float]) -> dict:
        """Apply ControlNet strength if workflow exposes a target node."""

        if controlnet_strength is None:
            return workflow_json

        node_id = workflow_config.get("controlnet_strength_node_id")
        if node_id is None:
            logger.debug(
                "Workflow '%s' does not define 'controlnet_strength_node_id'; skipping ControlNet strength override",
                workflow_name,
            )
            return workflow_json

        node_key = str(node_id)
        node = workflow_json.get(node_key)
        if not node or "inputs" not in node:
            logger.debug(
                "ControlNet node '%s' missing or has no inputs in workflow '%s'",
                node_key,
                workflow_name,
            )
            return workflow_json

        node["inputs"]["strength"] = float(controlnet_strength)
        logger.debug(
            "Applied ControlNet strength '%s' to node '%s' in workflow '%s'",
            controlnet_strength,
            node_key,
            workflow_name,
        )
        return workflow_json

    def prepare_workflow(self, workflow_name: str, prompt: str = None,
                         settings: Optional[str] = None,
                         resolution: Optional[str] = None,
                         image_data: Optional[bytes] = None,
                         seed: Optional[int] = None,
                         controlnet_strength: Optional[float] = None) -> dict:
        """Prepare a workflow with prompt, settings, and image data"""
        try:
            workflow_config = self.get_workflow(workflow_name)
            if not workflow_config:
                raise ValueError(f"Workflow '{workflow_name}' not found")

            # Load workflow file
            workflow_json = self.load_workflow_file(workflow_config['workflow'])

            # Update nodes with prompt and image
            workflow_json = self.update_workflow_nodes(
                workflow_json,
                workflow_config,
                prompt,
                image_data
            )

            # Apply resolution override if requested
            workflow_json = self.apply_resolution(
                workflow_json,
                workflow_config,
                workflow_name,
                resolution,
            )

            # Apply settings
            workflow_json = self.apply_settings(workflow_json, workflow_config, settings)

            # Apply optional ControlNet strength override.
            workflow_json = self.apply_controlnet_strength(
                workflow_json,
                workflow_config,
                workflow_name,
                controlnet_strength,
            )

            # Apply explicit seed last so user-provided seed is never overridden by settings hooks.
            workflow_json = self.apply_seed(
                workflow_json,
                workflow_config,
                workflow_name,
                seed,
            )

            return workflow_json
        except Exception as e:
            logger.error(f"Error preparing workflow: {e}")
            raise
