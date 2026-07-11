# RBG Smart Seed Variance 🌱 — sd-webui-forge-classic (Neo) / A1111 extension script
# Copyright (C) 2025  Ramon Guthrie
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Forge (Neo/Classic) port of the ComfyUI "RBG Smart Seed Variance" node.

All noise mathematics (direction-shift patterns, fade envelopes, protection
masks, Krea2 band rebalancing) is reused directly from
nodes/RBG_Smart_Seed_Variance.py so both frontends share one engine.

The webui integration follows the structure of sd-forge-sve (by the Forge
Neo author), which is proven to work on the Neo backend:
- per-run state lives in class attributes, configured in
  `before_process_batch` (seeds arrive through its kwargs),
- optional prompt encoding happens in `process_batch`,
- the conditioning is replaced out-of-place inside an
  `on_cfg_denoiser` callback running under `torch.inference_mode()`.

The positive text conditioning is rebuilt from the cached schedules on every
sampling step, so per-step replacement never contaminates later steps. This
also means step scheduling is exact and the ComfyUI node's "total_steps"
estimate input is not needed — the real step count is used.
"""

import importlib.util
import os

import gradio as gr
import numpy as np
import torch
from PIL import Image

from modules import errors, prompt_parser, script_callbacks, scripts, shared
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser
from modules.ui_components import InputAccordion


def _load_core():
    """Import the ComfyUI node module by path; it only needs torch (its
    node_helpers import fails gracefully and is never used from here)."""
    path = os.path.join(scripts.basedir(), "nodes", "RBG_Smart_Seed_Variance.py")
    spec = importlib.util.spec_from_file_location("rbg_ssv_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RBG_Smart_Seed_Variance


RBGCore = _load_core()
core = RBGCore()  # stateless noise engine; only its helper methods are used

FORGE_PORT_VERSION = "3.3"

AUTO_MODEL = "🤖 Auto-Detect"

# Forge diffusion-engine class name -> node model_type
ENGINE_TO_MODEL = {
    "ZImage": "⚡ Z-Image Turbo",
    "Krea2": "📸 Krea2 (SingleStream)",
    "QwenImage": "🖼️ Qwen-Image",
    "Flux": "🔮 Flux (Dev/Schnell)",
    "Flux2": "🔮 Flux (Dev/Schnell)",
    "Chroma": "🎨 Chroma HD",
    "ErnieImage": "🧧 ERNIE-Image",
    "StableDiffusionXL": "🖌️ SDXL",
    "Wan": "🎬 Wan2.2",
}

PRESET_CHOICES = [k for k in RBGCore.PRESETS if k != "❌ Disabled"]
PROTECT_CHOICES = ["🚫 None", "First Quarter", "First Half", "Last Quarter", "Last Half", "⚙️ Custom Regions", "🎲 Random Regions"]
SCHEDULE_CHOICES = ["constant", "decreasing", "step_cutoff", "hard_lock", "tiered_release"]


def _log(message):
    try:
        print(message)
    except UnicodeEncodeError:  # legacy Windows console codepages can't print emoji
        print(message.encode("ascii", "ignore").decode())


def _detect_model_type():
    engine = type(shared.sd_model).__name__
    if engine in ENGINE_TO_MODEL:
        return ENGINE_TO_MODEL[engine]
    if getattr(shared.sd_model, "is_sdxl", False):
        return "🖌️ SDXL"
    return "⚙️ Other"


def _schedule_multipliers(cfg, step, total):
    """Per-step (strength_mult, randomize_mult, seed_offset), mirroring the
    ComfyUI node's start/end_percent segmentation on a real step timeline."""
    total = max(total, 1)
    progress = step / total
    schedule = cfg["variance_schedule"]

    if schedule != "constant":
        # Composition Lock overrides noise_injection, same as the node
        cutoff = min(1.0, max(0.0, cfg["cutoff_step"] / total))
        cs = cfg["cutoff_strength"]

        if schedule == "step_cutoff":
            return (1.0, 1.0, 0) if progress < cutoff else (cs, 1.0, 0)

        if schedule == "decreasing":
            if progress < cutoff and cutoff > 0:
                segment = min(4, int(progress / cutoff * 5))
                return (1.0 - (segment / 5) * (1.0 - cs), 1.0, 0)
            return (cs, 1.0, 0)

        if schedule == "hard_lock":
            return (0.0, 1.0, 0) if progress < cutoff else (cs, 1.0, 0)

        if schedule == "tiered_release":
            phase2_end = cutoff + (1.0 - cutoff) * 0.25
            if progress < cutoff:
                return (cs, max(cs, 0.1), 0)
            if progress < phase2_end:
                return (0.6, 0.7, 1)
            return (1.0, 1.0, 0)

    injection = cfg["noise_injection"]
    switchover = 0.20
    if injection == "Beginning Steps":
        return (1.0, 1.0, 0) if progress < switchover else (0.0, 1.0, 0)
    if injection == "Ending Steps":
        return (0.0, 1.0, 0) if progress < switchover else (1.0, 1.0, 0)
    return (1.0, 1.0, 0)  # "🚫 None" / "All Steps"


def _apply_to_tensor(cfg, tensor, strength_mult, randomize_mult, seed_offset):
    """Run the shared noise engine on one crossattn tensor (2D or 3D).
    Everything is out-of-place with respect to the input tensor."""
    orig_dtype = tensor.dtype
    work = tensor.to(torch.float32)

    num_tokens = work.shape[1] if work.dim() == 3 else work.shape[0]
    seed = (cfg["seed"] + seed_offset) & 0xFFFFFFFFFFFFFFFF

    protect_mode = cfg["protect_mode"]
    if protect_mode == "⚙️ Custom Regions":
        mask = core._parse_protection_regions(cfg["protect_regions"], num_tokens)
    elif protect_mode == "🎲 Random Regions":
        mask = core._generate_random_protection_mask(seed, num_tokens, work.device)
    else:
        mask = core._legacy_protection_to_mask(protect_mode, num_tokens)

    target = cfg["vibe_cond"]
    if target is not None and target.shape[-1] != work.shape[-1]:
        target = None  # embedding width mismatch — vibe cannot steer this model

    rebalanced = core._apply_base_rebalance(work, cfg["pattern"], cfg["model_type"])
    modified = core._apply_noise(
        rebalanced,
        cfg["randomize_percent"] * randomize_mult,
        cfg["strength"] * strength_mult,
        mask,
        cfg["direction_config"],
        target,
        cfg["fade_curve"],
        seed,
        cfg["vibe_blend"],
        model_type=cfg["model_type"],
    )
    return modified.to(dtype=orig_dtype), work, modified


class RBGSmartSeedVarianceScript(scripts.Script):
    sorting_priority = 1120

    # per-run state, SVE-style: configured in before_process_batch,
    # consumed by the module-level cfg-denoiser callback
    enable: bool = False
    config: dict = None
    seed: int = 0
    vibe_prompt: str = ""
    heatmap = None
    invoked: bool = False   # callback entered at least once this run
    fired: bool = False     # noise actually written to the conditioning
    diagnosed: bool = False  # one-time text_cond/schedule dump for troubleshooting

    def title(self):
        return "RBG Smart Seed Variance 🌱"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        id_part = "img2img" if is_img2img else "txt2img"
        with InputAccordion(False, label=self.title(), elem_id=f"rbg_ssv_{id_part}_enabled") as enabled:
            with gr.Row():
                variance_preset = gr.Dropdown(PRESET_CHOICES, value="🌿 Balanced", label="Variance preset")
                fine_tune = gr.Slider(0, 100, value=50, step=1, label="Fine-tune variance (Custom preset only)")
            with gr.Row():
                model_type = gr.Dropdown([AUTO_MODEL] + list(RBGCore.MODEL_ADJUSTMENTS), value=AUTO_MODEL, label="Model type")
                fade_curve = gr.Dropdown(RBGCore.FADE_CURVES, value="Instant", label="Fade curve")
            with gr.Row():
                direction_shift = gr.Dropdown(list(RBGCore.DIRECTION_SHIFTS), value="🚫 None", label="Direction shift")
                shift_strength = gr.Slider(0, 200, value=100, step=1, label="Shift strength %")
            with gr.Row():
                noise_injection = gr.Dropdown(RBGCore.NOISE_INJECTION, value="Beginning Steps", label="Noise injection timing")
                variance_schedule = gr.Dropdown(SCHEDULE_CHOICES, value="constant", label="Variance schedule (Composition Lock 🔒)")
            with gr.Row():
                cutoff_step = gr.Slider(0, 100, value=8, step=1, label="Cutoff step")
                cutoff_strength = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Cutoff strength")
            with gr.Row():
                protect_mode = gr.Dropdown(PROTECT_CHOICES, value="🚫 None", label="Protect prompt tokens")
                protect_regions = gr.Textbox(value="", label="Custom regions (e.g. 0-5,15-20)", placeholder="Only used with ⚙️ Custom Regions")
            with gr.Row():
                seed = gr.Number(value=-1, precision=0, label="Variance seed (-1 = follow image seed)")
                show_heatmap = gr.Checkbox(False, label="Output variance heatmap")
            with gr.Accordion("Target Vibe (optional)", open=False):
                vibe_prompt = gr.Textbox(value="", label="Vibe prompt", placeholder="Optional prompt whose embedding steers the variance direction")
                vibe_blend = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Vibe blend")

        components = [
            enabled, variance_preset, fine_tune, model_type, fade_curve,
            direction_shift, shift_strength, noise_injection, variance_schedule,
            cutoff_step, cutoff_strength, protect_mode, protect_regions,
            seed, show_heatmap, vibe_prompt, vibe_blend,
        ]

        infotext_names = [
            "enabled", "preset", "fine tune", "model type", "fade curve",
            "direction shift", "shift strength", "noise injection", "schedule",
            "cutoff step", "cutoff strength", "protect mode", "protect regions",
            "seed", "heatmap", "vibe prompt", "vibe blend",
        ]
        self.infotext_fields = [(component, f"RBG SSV {name}") for component, name in zip(components, infotext_names)]

        return components

    def before_process_batch(self, p, enabled, variance_preset, fine_tune, model_type, fade_curve,
                             direction_shift, shift_strength, noise_injection, variance_schedule,
                             cutoff_step, cutoff_strength, protect_mode, protect_regions,
                             seed, show_heatmap, vibe_prompt, vibe_blend, **kwargs):
        cls = RBGSmartSeedVarianceScript
        cls.enable = bool(enabled)
        if not cls.enable:
            cls.config = None
            cls.vibe_prompt = ""
            cls.heatmap = None
            return

        requested_model = model_type
        if model_type == AUTO_MODEL:
            model_type = _detect_model_type()

        preset_config = RBGCore.PRESETS.get(variance_preset)
        if preset_config is None:  # ⚙️ Custom
            randomize_percent = (fine_tune / 100.0) * 5.0
            strength = (fine_tune / 100.0) * 50.0
        else:
            randomize_percent, strength = preset_config

        strength_mult, randomize_mult = RBGCore.MODEL_ADJUSTMENTS.get(model_type, (1.0, 1.0))
        randomize_percent *= randomize_mult
        strength *= strength_mult
        if strength <= 0 or randomize_percent <= 0:
            cls.enable = False
            cls.config = None
            return

        direction_config = RBGCore.DIRECTION_SHIFTS.get(direction_shift)
        if direction_config is not None:
            pattern, preset_mult = direction_config
            direction_config = (pattern, preset_mult * (shift_strength / 100.0))
        else:
            pattern = "random"

        seed = int(seed)
        if seed < 0:
            seeds = kwargs.get("seeds") or getattr(p, "seeds", None) or [0]
            seed = int(seeds[0])
        cls.seed = seed

        cls.vibe_prompt = (vibe_prompt or "").strip()
        cls.heatmap = None
        cls.invoked = False
        cls.fired = False
        cls.diagnosed = False

        # Re-register the callback for this run. This is defensive: it repairs
        # a registration lost to callback clearing / UI reloads, and forces
        # script_callbacks' length-validated ordered cache to rebuild so the
        # dispatch list is guaranteed to contain us.
        try:
            registry = script_callbacks.callback_map.get("callbacks_cfg_denoiser", [])
            was_registered = any(getattr(c, "callback", None) == cls.on_cfg for c in registry)
            names_before = [getattr(c, "name", "?") for c in registry]
            script_callbacks.remove_callbacks_for_function(cls.on_cfg)
            on_cfg_denoiser(cls.on_cfg)
            if not was_registered:
                _log(f"[RBG SSV] callback was MISSING from the cfg_denoiser registry (had: {names_before}); re-registered")
        except Exception:
            errors.report("RBG Smart Seed Variance: callback registry check failed", exc_info=True)

        cls.config = {
            "randomize_percent": randomize_percent,
            "strength": strength,
            "model_type": model_type,
            "fade_curve": fade_curve,
            "pattern": pattern,
            "direction_config": direction_config,
            "noise_injection": noise_injection,
            "variance_schedule": variance_schedule,
            "cutoff_step": int(cutoff_step),
            "cutoff_strength": float(cutoff_strength),
            "protect_mode": protect_mode,
            "protect_regions": protect_regions or "",
            "seed": seed,
            "show_heatmap": bool(show_heatmap),
            "vibe_cond": None,  # encoded in process_batch
            "vibe_blend": float(vibe_blend),
        }
        _log(f"[RBG SSV v{FORGE_PORT_VERSION}] enabled: model={model_type}, strength={strength:.2f}, randomize={randomize_percent:.2f}%, seed={seed}")

        p.extra_generation_params.update({
            "RBG SSV enabled": True,
            "RBG SSV preset": variance_preset,
            "RBG SSV fine tune": fine_tune if preset_config is None else None,
            "RBG SSV model type": requested_model,
            "RBG SSV fade curve": fade_curve,
            "RBG SSV direction shift": direction_shift,
            "RBG SSV shift strength": shift_strength,
            "RBG SSV noise injection": noise_injection,
            "RBG SSV schedule": variance_schedule,
            "RBG SSV cutoff step": cutoff_step if variance_schedule != "constant" else None,
            "RBG SSV cutoff strength": cutoff_strength if variance_schedule != "constant" else None,
            "RBG SSV protect mode": protect_mode,
            "RBG SSV protect regions": protect_regions if protect_mode == "⚙️ Custom Regions" else None,
            "RBG SSV seed": seed,
            "RBG SSV vibe prompt": cls.vibe_prompt or None,
            "RBG SSV vibe blend": vibe_blend if cls.vibe_prompt else None,
        })  # None values are omitted from the infotext by create_infotext

    def process_batch(self, p, *args, **kwargs):
        """Encode the optional vibe prompt here — the same stage where
        sd-forge-sve encodes its warmup prompt (model + LoRAs are ready)."""
        cls = RBGSmartSeedVarianceScript
        if not cls.enable or cls.config is None or not cls.vibe_prompt:
            return

        try:
            prompts = prompt_parser.SdConditioning(
                [cls.vibe_prompt],
                width=p.width,
                height=p.height,
                distilled_cfg_scale=getattr(p, "distilled_cfg_scale", None),
            )
            encoded = p.sd_model.get_learned_conditioning(prompts)
            if isinstance(encoded, dict):
                encoded = encoded.get("crossattn")
            elif isinstance(encoded, (list, tuple)) and encoded:
                encoded = encoded[0]
            if isinstance(encoded, torch.Tensor):
                cls.config["vibe_cond"] = encoded.detach().to("cpu", torch.float32)
            else:
                _log("[RBG SSV] vibe prompt produced no usable conditioning; continuing without it")
        except Exception:
            errors.report("RBG Smart Seed Variance: could not encode vibe prompt, continuing without it", exc_info=True)

    @classmethod
    def _describe_cond(cls, text_cond):
        if isinstance(text_cond, dict):
            parts = []
            for k, v in text_cond.items():
                if isinstance(v, torch.Tensor):
                    parts.append(f"{k}={tuple(v.shape)}/{v.dtype}")
                else:
                    parts.append(f"{k}=<{type(v).__name__}>")
            return "dict{" + ", ".join(parts) + "}"
        if isinstance(text_cond, torch.Tensor):
            return f"tensor{tuple(text_cond.shape)}/{text_cond.dtype}/{text_cond.device}"
        if isinstance(text_cond, (list, tuple)):
            return f"{type(text_cond).__name__}[len={len(text_cond)}]" + (f" first={cls._describe_cond(text_cond[0])}" if text_cond else "")
        return f"<{type(text_cond).__name__}>"

    @classmethod
    @torch.inference_mode()
    def on_cfg(cls, params: CFGDenoiserParams):
        if not cls.enable or cls.config is None:
            return
        cls.invoked = True

        # denoiser.step/total_steps reset per pass (hires fix gets its own timeline)
        step = getattr(params.denoiser, "step", params.sampling_step)
        total = getattr(params.denoiser, "total_steps", None) or params.total_sampling_steps

        if not cls.diagnosed:
            cls.diagnosed = True
            preview_mult, _, _ = _schedule_multipliers(cls.config, step, total)
            _log(f"[RBG SSV] diagnostic: step={step + 1}/{total}  text_cond={cls._describe_cond(params.text_cond)}  "
                 f"text_uncond={cls._describe_cond(params.text_uncond)}  schedule_strength_mult={preview_mult:.3f}")

        if params.text_cond is None:
            return

        try:
            strength_mult, randomize_mult, seed_offset = _schedule_multipliers(cls.config, step, total)
            if strength_mult <= 0.0:
                return

            text_cond = params.text_cond
            tensor = text_cond["crossattn"] if isinstance(text_cond, dict) else text_cond
            if not isinstance(tensor, torch.Tensor) or tensor.dim() < 2:
                _log(f"[RBG SSV] diagnostic: text_cond tensor unusable for noise injection: {cls._describe_cond(text_cond)}")
                return

            # Some text encoders (e.g. Krea2's per-chunk Qwen3VL output) keep
            # an extra leading batch dim per chunk, so the stacked cond ends
            # up 4D+: (num_chunks, chunk_batch, tokens, embed_dim). The noise
            # engine only expects (tokens, embed_dim) or (batch, tokens,
            # embed_dim), so collapse every leading dim into one batch dim
            # and restore the original shape afterwards.
            original_shape = tensor.shape
            work_tensor = tensor if tensor.dim() in (2, 3) else tensor.reshape(-1, tensor.shape[-2], tensor.shape[-1])

            new_tensor, before, after = _apply_to_tensor(cls.config, work_tensor, strength_mult, randomize_mult, seed_offset)
            if new_tensor.shape != original_shape:
                new_tensor = new_tensor.reshape(original_shape)
                before = before.reshape(original_shape)
                after = after.reshape(original_shape)

            if cls.config["show_heatmap"] and cls.heatmap is None:
                cls.heatmap = cls.build_heatmap(before, after)

            # text_cond is rebuilt from the cached schedules every step, so
            # replacing it here never contaminates later steps
            if isinstance(text_cond, dict):
                text_cond["crossattn"] = new_tensor
            else:
                params.text_cond = new_tensor

            if not cls.fired:
                cls.fired = True
                delta = (after - before).abs().mean().item()
                _log(f"[RBG SSV] injecting variance: step {step + 1}/{total}, cond {tuple(tensor.shape)}, mean |delta| = {delta:.4f}")
        except Exception:
            errors.report("RBG Smart Seed Variance: failed to apply variance, disabling for this run", exc_info=True)
            cls.enable = False
            cls.config = None

    @staticmethod
    def build_heatmap(before, after):
        """Token-wise |Δ embedding| strip, like the node's variance_heatmap output."""
        diff = (after - before).norm(dim=-1)
        diff = diff.reshape(-1, diff.shape[-1])  # collapse any leading dims down to (rows, tokens)
        row = diff[0]
        peak = row.max()
        if peak > 0:
            row = row / peak
        arr = (row.detach().float().cpu().numpy() * 255).astype(np.uint8)
        image = Image.fromarray(arr[None, :], mode="L").resize((512, 64), Image.NEAREST)
        return image.convert("RGB")

    def postprocess(self, p, processed, *args):
        cls = RBGSmartSeedVarianceScript
        if cls.config is not None and cls.enable and not cls.fired:
            if not cls.invoked:
                _log("[RBG SSV] WARNING: the cfg_denoiser callback was never invoked during sampling — "
                     "the dispatch list did not include this extension. Please report the sampler/model combo.")
            else:
                _log("[RBG SSV] WARNING: callback was invoked but no variance was applied — "
                     "see the [RBG SSV] diagnostic line above for the actual text_cond shape/schedule "
                     "at step 1, or check for an error reported above.")

        if cls.heatmap is not None:
            processed.images.append(cls.heatmap)
            processed.infotexts.append("RBG Smart Seed Variance heatmap")

        cls.enable = False
        cls.config = None
        cls.heatmap = None
        cls.invoked = False
        cls.fired = False
        cls.diagnosed = False


on_cfg_denoiser(RBGSmartSeedVarianceScript.on_cfg)
