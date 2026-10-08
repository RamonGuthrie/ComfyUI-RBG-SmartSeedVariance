# RBG Smart Seed Variance
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
RBG Smart Seed Variance 🌱 2.2.4
A user-friendly ComfyUI node for enhancing seed diversity across modern DiT and diffusion models.
Includes calibrated support for Qwen-Image-2.1, Structure Freeze, Normalize Strength, and NegPiP parsing.
"""

import torch
import random
import math
import re

try:
    import node_helpers
    _NODE_HELPERS_AVAILABLE = True
except ImportError:
    _NODE_HELPERS_AVAILABLE = False


class RBG_Smart_Seed_Variance:
    """
    Adds diversity to outputs by injecting noise into text embeddings during early generation steps.
    Supports low-frequency structural freezing, NegPiP parsing, and calibrated Qwen-Image-2.1 support.
    """
    
    PRESETS = {
        "❌ Disabled": (0.0, 0.0),
        "🌱 Subtle": (1.0, 10.0),
        "🌿 Balanced": (2.0, 20.0),
        "🪴 Creative": (3.0, 30.0),
        "🌳 Bold": (4.0, 40.0),
        "🌴 Wild": (5.0, 50.0),
        "⚙️ Custom": None,
    }
    
    MODEL_ADJUSTMENTS = {
        "⚡ Z-Image Turbo": (1.0, 1.0),
        "📸 Krea2 (SingleStream)": (1.2, 1.1),
        "🖼️ Qwen-Image-2.1 (7B SingleStream)": (0.45, 0.75),
        "🖼️ Qwen-Image (Legacy 20B MMDiT)": (0.8, 0.85),
        "🔮 Flux (Dev/Schnell)": (0.5, 0.8),
        "🎨 Chroma HD": (0.05, 0.5),
        "🧧 ERNIE-Image": (0.08, 0.45),
        "🔮 Ideogram 4.0": (0.4, 0.7),
        "🖌️ SDXL": (0.6, 0.7),
        "🎬 Wan2.2": (0.8, 0.8),
        "⚙️ Other": (0.8, 0.8),
    }
    
    PROTECT_OPTIONS = {
        "🚫 None": (0.0, "start"),
        "First Quarter": (0.25, "start"),
        "First Half": (0.50, "start"),
        "Last Quarter": (0.25, "end"),
        "Last Half": (0.50, "end"),
    }
    
    NOISE_INJECTION = [
        "🚫 None",
        "Beginning Steps",
        "Ending Steps",
        "All Steps",
    ]
    
    DIRECTION_SHIFTS = {
        "🚫 None": None,
        "🌀 Chaos": ("scatter", 1.2),
        "📐 Order": ("compress", 0.8),
        "🎨 Abstract": ("wave", 1.0),
        "📸 Realistic": ("sharpen", 0.9),
        "🌑 Moody": ("negative", 1.0),
        "💭 Dreamy": ("smooth", 1.1),
        "🎭 Dynamic Pose": ("spatial", 1.2),
        "🖼️ Composition": ("gradient", 1.0),
        "🛡️ Latent Noise Guard (Ideogram 4)": ("latent_noise_guard", 0.65),
        "🔤 Typography & Alpha Guard (Qwen 2.1)": ("qwen_typo_guard", 0.65),
        "🎞️ Filmic Contrast & De-Yellow (Qwen 2.1)": ("qwen_filmic_contrast", 0.85),
        "🧬 Structural Geometry & Hand Anchor (Qwen 2.1)": ("qwen_structural_anchor", 0.70),
        "🗿 Visceral Expression & Grit (Krea2)": ("visceral_grit", 1.2),
        "🤪 Facial Dynamics & Asymmetry (Krea2)": ("wild_facial_asymmetry", 1.50),
        "🎭 Expression Variance": ("expression", 1.2),
        "🔬 Microscopic": ("microscopic", 1.3),
        "👥 Group Diversity": ("group_diversity", 1.25),
        "🧬 Face-Variance Expansion": ("facevar", 1.25),
        "🌌 Cosmic": ("cosmic", 1.2),
        "🌎 Diversity": ("diversity", 1.15),
        "🌈 Vibrant": ("positive", 1.1),
        "🧭 Semantic Drift (Centroid-Safe)": ("semantic_drift", 1.0),
        "☠️ Bone Anatomical Coherence": ("anatomical_coherence", 0.6),
        "🧱 Structural Lock": ("structural_lock", 1.0),
        "🎞️ Cinematic Framing": ("cinematic_framing", 1.1),
        "🧠 Identity Stretch": ("identity_stretch", 1.25),
        "🪶 Texture Lift": ("texture_lift", 1.0),
        "🌸 Natural": ("pink", 1.0),
        "🏞️ Landscape Depth": ("landscape_depth", 1.1),
        "🌊 Motion Blur": ("motion_blur", 1.1),
    }
    
    FADE_CURVES = [
        "Instant",
        "Linear",
        "Ease-Out",
        "Ease-In",
        "Ease-In-Out",
        "Smooth Step",
        "Burst",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "variance_preset": (list(cls.PRESETS.keys()), {
                    "default": "🌿 Balanced",
                    "tooltip": "Select variance intensity level. Higher = more diverse outputs but less prompt adherence."
                }),
                "fine_tune_variance": ("INT", {
                    "default": 50,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Fine-tune variance (0-100%). Only used when preset is 'Custom'."
                }),
                "model_type": (list(cls.MODEL_ADJUSTMENTS.keys()), {
                    "default": "⚡ Z-Image Turbo",
                    "tooltip": "Select your model for optimized settings."
                }),
                "fade_curve": (cls.FADE_CURVES, {
                    "default": "Instant",
                    "tooltip": "How noise fades spatially across the embedding."
                }),
                "noise_injection": (cls.NOISE_INJECTION, {
                    "default": "Beginning Steps",
                    "tooltip": "When to apply noise during sampling. Beginning Steps = composition/pose diversity, Ending Steps = texture/detail diversity, All Steps = continuous variance. (Note: Qwen-Image-2.1 auto-locks to 'All Steps' to prevent KV-cache eviction crashes)."
                }),
                "protect_mode": (["🚫 None", "First Quarter", "First Half", "Last Quarter", "Last Half", "⚙️ Custom Regions", "🎲 Random Regions"], {
                    "default": "🚫 None",
                    "tooltip": "Protection mode: shield specific token regions from noise to preserve subject identity, core descriptors, or trigger words."
                }),
                "protect_regions": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Enter token indices or ranges to protect (comma-separated, e.g. '0-4, 8, 12-15'). Used when protect_mode is 'Custom Regions'."
                }),
                "direction_shift": (list(cls.DIRECTION_SHIFTS.keys()), {
                    "default": "🚫 None",
                    "tooltip": "Apply directional bias to embeddings instead of pure random noise."
                }),
                "shift_strength": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Strength multiplier for the direction shift pattern (0-200%)."
                }),
                "variance_schedule": (["constant", "decreasing", "step_cutoff", "tiered_release", "hard_lock"], {
                    "default": "constant",
                    "tooltip": "Composition Lock schedule."
                }),
                "cutoff_step": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 100,
                    "tooltip": "The step number where the cutoff ends."
                }),
                "total_steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Total sampling steps."
                }),
                "cutoff_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "Noise multiplier after cutoff."
                }),
                "structure_freeze": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Structure Freeze 🧊: Locks first 25% of low-frequency channels to preserve silhouette and pose."
                }),
                "normalize_strength": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Dynamically scales injected noise against the standard deviation spread of the input embedding."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for noise generation."
                }),
            },
            "optional": {
                "target_vibe": ("CONDITIONING", {
                    "tooltip": "Optional reference conditioning stream to steer variance direction."
                }),
                "vibe_blend": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": "Mix ratio of target vibe steering."
                }),
                "prompt_text": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "forceInput": True,
                    "tooltip": "Source prompt text string for NegPiP (word:-1.2) syntax parsing."
                }),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "IMAGE")
    RETURN_NAMES = ("conditioning", "variance_heatmap")
    FUNCTION = "apply_variance"
    CATEGORY = "RBG Suite/Advanced"

    def apply_variance(self, conditioning, variance_preset, fine_tune_variance, model_type, fade_curve,
                       noise_injection, protect_mode, protect_regions, direction_shift, shift_strength, seed,
                       variance_schedule="constant", cutoff_step=8, total_steps=20, cutoff_strength=0.0,
                       structure_freeze=False, normalize_strength=False, target_vibe=None, vibe_blend=0.5,
                       prompt_text=None):
        
        # 1. Qwen-Image-2.1 Prefix KV-Cache Guard
        if model_type == "🖼️ Qwen-Image-2.1 (7B SingleStream)":
            if noise_injection in ["Beginning Steps", "Ending Steps"]:
                noise_injection = "All Steps"
            if variance_schedule != "constant":
                variance_schedule = "constant"

        if not _NODE_HELPERS_AVAILABLE and (variance_schedule != "constant" or noise_injection in ["Beginning Steps", "Ending Steps"]):
            raise RuntimeError("RBG Smart Seed Variance: 'node_helpers' is required for scheduled conditioning operations.")

        if variance_preset == "❌ Disabled":
            return {"ui": {"protection_data": []}, "result": (conditioning, torch.zeros((1, 64, 64, 3), dtype=torch.float32))}
        
        preset_config = self.PRESETS.get(variance_preset)
        if preset_config is None:
            randomize_percent = (fine_tune_variance / 100.0) * 5.0  
            strength = (fine_tune_variance / 100.0) * 50.0  
        else:
            randomize_percent, strength = preset_config
        
        strength_mult, randomize_mult = self.MODEL_ADJUSTMENTS.get(model_type, (1.0, 1.0))
        randomize_percent *= randomize_mult
        strength *= strength_mult
        
        user_mult = shift_strength / 100.0
        direction_config = self.DIRECTION_SHIFTS.get(direction_shift, None)
        if direction_config is not None:
            pattern, preset_mult = direction_config
            direction_config = (pattern, preset_mult * user_mult)

        negpip_map = {}
        if prompt_text:
            weights = re.findall(r'\(([^:]+):(-?[\d.]+)\)', prompt_text)
            for word, val in weights:
                try:
                    fval = float(val)
                    if fval < 0:
                        negpip_map[word.strip().lower()] = fval
                except ValueError:
                    continue

        word_tokens_for_ui = re.findall(r'\w+|[^\w\s]', prompt_text) if prompt_text else []

        noisy_conditioning = []
        protection_masks_for_ui = []
        heatmap_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        
        for i, cond in enumerate(conditioning):
            if len(cond) < 2:
                noisy_conditioning.append(cond)
                continue
            
            cond_tensor = cond[0]
            cond_dict = cond[1].copy() if len(cond) > 1 else {}
            
            if cond_tensor.dim() == 3:
                num_tokens = cond_tensor.shape[1]
            elif cond_tensor.dim() == 2:
                num_tokens = cond_tensor.shape[0]
            else:
                noisy_conditioning.append(cond)
                continue
            
            if protect_mode == "⚙️ Custom Regions":
                protection_mask = self._parse_protection_regions(protect_regions, num_tokens)
            elif protect_mode == "🎲 Random Regions":
                protection_mask = self._generate_random_protection_mask(seed + i, num_tokens, cond_tensor.device)
            else:
                protection_mask = self._legacy_protection_to_mask(protect_mode, num_tokens)
            
            current_target = self._pick_target_chunk(target_vibe, i, cond_tensor)
            pattern = direction_config[0] if direction_config is not None else "random"

            rebalanced_cond_tensor = self._apply_base_rebalance(cond_tensor, pattern, model_type)

            token_negpip_modifiers = torch.ones(num_tokens, dtype=cond_tensor.dtype, device=cond_tensor.device)
            if negpip_map and word_tokens_for_ui:
                num_words = len(word_tokens_for_ui)
                word_values = [negpip_map.get(w.lower(), 1.0) for w in word_tokens_for_ui]
                word_value_tensor = torch.tensor(word_values, dtype=cond_tensor.dtype, device=cond_tensor.device)
                token_idx_range = torch.arange(num_tokens, device=cond_tensor.device)
                word_indices = torch.clamp((token_idx_range * num_words) // num_tokens, max=num_words - 1)
                token_negpip_modifiers = word_value_tensor[word_indices]

            if normalize_strength:
                emb_std = float(cond_tensor.to(torch.float32).std().clamp(min=1e-5).item())
                strength_scaler = emb_std / 0.4
                current_strength = strength * strength_scaler
            else:
                current_strength = strength

            modified_tensor = self._apply_noise(
                rebalanced_cond_tensor,
                randomize_percent,
                current_strength,
                protection_mask,
                direction_config,
                current_target,
                fade_curve,
                seed + i,
                vibe_blend,
                model_type=model_type,
                structure_freeze=structure_freeze,
                negpip_modifiers=token_negpip_modifiers
            )
            
            if model_type == "🖌️ SDXL" and "pooled_output" in cond_dict and isinstance(cond_dict["pooled_output"], torch.Tensor):
                cond_dict["pooled_output"] = self._apply_noise_to_pooled(
                    cond_dict["pooled_output"], randomize_percent, current_strength, direction_config,
                    seed + i + 777, model_type=model_type
                )

            if i == 0:
                diff = (modified_tensor.to(torch.float32) - cond_tensor.to(torch.float32)).norm(dim=-1)
                if diff.dim() == 1:
                    diff = diff.unsqueeze(0)
                diff_slice = diff[0]
                max_diff = diff_slice.max()
                if max_diff > 0:
                    diff_slice = diff_slice / max_diff
                
                viz_width = max(1, diff_slice.shape[0])
                heatmap_row = diff_slice.view(1, 1, viz_width, 1).expand(1, 64, viz_width, 3)
                heatmap_tensor = torch.nn.functional.interpolate(
                    heatmap_row.permute(0, 3, 1, 2), size=(64, 512), mode='nearest'
                ).permute(0, 2, 3, 1).contiguous()
            
            # Extract flat array of booleans for the Sync & Bloom UI widget
            if hasattr(protection_mask, 'tolist'):
                mask_list = (protection_mask.cpu().numpy().tolist()
                             if protection_mask.device.type != 'cpu'
                             else protection_mask.tolist())
                if mask_list and isinstance(mask_list[0], list):
                    mask_list = [item for sublist in mask_list for item in sublist]
            else:
                mask_list = []

            cond_dict["rbg_variance_fade_curve"] = fade_curve
            cond_dict["rbg_variance_applied"] = True
            cond_dict["rbg_direction_shift"] = direction_shift
            cond_dict["rbg_noise_injection"] = noise_injection
            cond_dict["rbg_protect_mode"] = protect_mode
            if protect_mode == "⚙️ Custom Regions":
                cond_dict["rbg_protect_regions"] = protect_regions

            protection_masks_for_ui.append(mask_list)
            noisy_conditioning.append((modified_tensor, cond_dict))
        
        # Schedule branching
        if variance_schedule != "constant":
            cutoff_percent = min(1.0, max(0.0, cutoff_step / total_steps))
            
            if variance_schedule == "step_cutoff":
                new_conditioning = node_helpers.conditioning_set_values(
                    noisy_conditioning, {"start_percent": 0.0, "end_percent": cutoff_percent}
                )
                if cutoff_strength > 0:
                    scaled_noisy = self._create_scaled_conditioning(
                        conditioning, randomize_percent, strength * cutoff_strength, target_vibe,
                        protect_mode, protect_regions, direction_config, fade_curve, seed, vibe_blend,
                        model_type=model_type, structure_freeze=structure_freeze,
                        negpip_modifiers=token_negpip_modifiers, normalize_strength=normalize_strength
                    )
                    new_conditioning += node_helpers.conditioning_set_values(
                        scaled_noisy, {"start_percent": cutoff_percent, "end_percent": 1.0}
                    )
                else:
                    new_conditioning += node_helpers.conditioning_set_values(
                        conditioning, {"start_percent": cutoff_percent, "end_percent": 1.0}
                    )
                return {"ui": {"protection_data": protection_masks_for_ui}, "result": (new_conditioning, heatmap_tensor)}
                
            elif variance_schedule == "decreasing":
                num_segments = 5
                new_conditioning = []
                for s in range(num_segments):
                    seg_start = (s / num_segments) * cutoff_percent
                    seg_end = ((s + 1) / num_segments) * cutoff_percent
                    seg_multiplier = 1.0 - (s / num_segments) * (1.0 - cutoff_strength)
                    
                    seg_noisy = self._create_scaled_conditioning(
                        conditioning, randomize_percent, strength * seg_multiplier, target_vibe,
                        protect_mode, protect_regions, direction_config, fade_curve, seed, vibe_blend,
                        model_type=model_type, structure_freeze=structure_freeze,
                        negpip_modifiers=token_negpip_modifiers, normalize_strength=normalize_strength
                    )
                    new_conditioning += node_helpers.conditioning_set_values(
                        seg_noisy, {"start_percent": seg_start, "end_percent": seg_end}
                    )
                
                if cutoff_percent < 1.0:
                    final_noisy = self._create_scaled_conditioning(
                        conditioning, randomize_percent, strength * cutoff_strength, target_vibe,
                        protect_mode, protect_regions, direction_config, fade_curve, seed, vibe_blend,
                        model_type=model_type, structure_freeze=structure_freeze,
                        negpip_modifiers=token_negpip_modifiers, normalize_strength=normalize_strength
                    )
                    new_conditioning += node_helpers.conditioning_set_values(
                        final_noisy, {"start_percent": cutoff_percent, "end_percent": 1.0}
                    )
                return {"ui": {"protection_data": protection_masks_for_ui}, "result": (new_conditioning, heatmap_tensor)}
            
            elif variance_schedule == "hard_lock":
                new_conditioning = node_helpers.conditioning_set_values(
                    conditioning, {"start_percent": 0.0, "end_percent": cutoff_percent}
                )
                if cutoff_strength > 0:
                    scaled_noisy = self._create_scaled_conditioning(
                        conditioning, randomize_percent, strength * cutoff_strength, target_vibe,
                        protect_mode, protect_regions, direction_config, fade_curve, seed, vibe_blend,
                        model_type=model_type, structure_freeze=structure_freeze,
                        negpip_modifiers=token_negpip_modifiers, normalize_strength=normalize_strength
                    )
                    new_conditioning += node_helpers.conditioning_set_values(
                        scaled_noisy, {"start_percent": cutoff_percent, "end_percent": 1.0}
                    )
                else:
                    new_conditioning += node_helpers.conditioning_set_values(
                        conditioning, {"start_percent": cutoff_percent, "end_percent": 1.0}
                    )
                return {"ui": {"protection_data": protection_masks_for_ui}, "result": (new_conditioning, heatmap_tensor)}

            elif variance_schedule == "tiered_release":
                remaining  = 1.0 - cutoff_percent
                phase2_end = cutoff_percent + remaining * 0.25

                phase1_noisy = self._create_scaled_conditioning(
                    conditioning, randomize_percent * max(cutoff_strength, 0.1),
                    strength * cutoff_strength, target_vibe,
                    protect_mode, protect_regions, direction_config, fade_curve, seed, vibe_blend,
                    model_type=model_type, structure_freeze=structure_freeze,
                    negpip_modifiers=token_negpip_modifiers, normalize_strength=normalize_strength
                )
                new_conditioning = node_helpers.conditioning_set_values(
                    phase1_noisy, {"start_percent": 0.0, "end_percent": cutoff_percent}
                )

                phase2_noisy = self._create_scaled_conditioning(
                    conditioning, randomize_percent * 0.7, strength * 0.6, target_vibe,
                    protect_mode, protect_regions, direction_config, fade_curve, seed + 1, vibe_blend,
                    model_type=model_type, structure_freeze=structure_freeze,
                    negpip_modifiers=token_negpip_modifiers, normalize_strength=normalize_strength
                )
                new_conditioning += node_helpers.conditioning_set_values(
                    phase2_noisy, {"start_percent": cutoff_percent, "end_percent": phase2_end}
                )

                if phase2_end < 1.0:
                    new_conditioning += node_helpers.conditioning_set_values(
                        noisy_conditioning, {"start_percent": phase2_end, "end_percent": 1.0}
                    )
                return {"ui": {"protection_data": protection_masks_for_ui}, "result": (new_conditioning, heatmap_tensor)}

        if noise_injection == "🚫 None" or noise_injection == "All Steps":
            return {"ui": {"protection_data": protection_masks_for_ui}, "result": (noisy_conditioning, heatmap_tensor)}
        
        switchover = 0.20
        if noise_injection == "Beginning Steps":
            new_conditioning = node_helpers.conditioning_set_values(
                noisy_conditioning, {"start_percent": 0.0, "end_percent": switchover}
            )
            new_conditioning += node_helpers.conditioning_set_values(
                conditioning, {"start_percent": switchover, "end_percent": 1.0}
            )
        elif noise_injection == "Ending Steps":
            new_conditioning = node_helpers.conditioning_set_values(
                conditioning, {"start_percent": 0.0, "end_percent": switchover}
            )
            new_conditioning += node_helpers.conditioning_set_values(
                noisy_conditioning, {"start_percent": switchover, "end_percent": 1.0}
            )
        else:
            new_conditioning = noisy_conditioning
        
        return {"ui": {"protection_data": protection_masks_for_ui}, "result": (new_conditioning, heatmap_tensor)}

    def _create_scaled_conditioning(self, conditioning, randomize_percent, strength,
                                    target_vibe, protect_mode, protect_regions,
                                    direction_config, fade_curve, seed, vibe_blend=0.5, model_type="⚙️ Other",
                                    structure_freeze=False, negpip_modifiers=None, normalize_strength=False):
        scaled_conditioning = []
        for i, cond in enumerate(conditioning):
            if len(cond) < 2:
                scaled_conditioning.append(cond)
                continue
            cond_tensor = cond[0]
            cond_dict   = cond[1].copy()

            if cond_tensor.dim() == 3:
                num_tokens = cond_tensor.shape[1]
            elif cond_tensor.dim() == 2:
                num_tokens = cond_tensor.shape[0]
            else:
                scaled_conditioning.append(cond)
                continue

            if protect_mode == "⚙️ Custom Regions":
                protection_mask = self._parse_protection_regions(protect_regions, num_tokens)
            elif protect_mode == "🎲 Random Regions":
                protection_mask = self._generate_random_protection_mask(seed + i, num_tokens, cond_tensor.device)
            else:
                protection_mask = self._legacy_protection_to_mask(protect_mode, num_tokens)

            current_target = self._pick_target_chunk(target_vibe, i, cond_tensor)
            pattern = direction_config[0] if direction_config is not None else "random"

            rebalanced_cond_tensor = self._apply_base_rebalance(cond_tensor, pattern, model_type)

            if normalize_strength:
                emb_std = float(cond_tensor.to(torch.float32).std().clamp(min=1e-5).item())
                strength_scaler = emb_std / 0.4
                current_strength = strength * strength_scaler
            else:
                current_strength = strength

            modified_tensor = self._apply_noise(
                rebalanced_cond_tensor, randomize_percent, current_strength, protection_mask,
                direction_config, current_target, fade_curve, seed + i, vibe_blend, model_type=model_type,
                structure_freeze=structure_freeze, negpip_modifiers=negpip_modifiers
            )

            if model_type == "🖌️ SDXL" and "pooled_output" in cond_dict and isinstance(cond_dict["pooled_output"], torch.Tensor):
                cond_dict["pooled_output"] = self._apply_noise_to_pooled(
                    cond_dict["pooled_output"], randomize_percent, current_strength, direction_config,
                    seed + i + 777, model_type=model_type
                )

            scaled_conditioning.append((modified_tensor, cond_dict))
        return scaled_conditioning
    
    def _parse_protection_regions(self, region_string, num_tokens):
        if not region_string or region_string.lower() == "none":
            return torch.zeros(num_tokens, dtype=torch.bool)
        protected_mask = torch.zeros(num_tokens, dtype=torch.bool)
        try:
            regions = region_string.replace(" ", "").split(",")
            for region in regions:
                if not region:
                    continue
                if "-" in region:
                    parts = region.split("-")
                    if len(parts) != 2:
                        continue
                    start_idx = int(parts[0])
                    end_idx = int(parts[1])
                    if start_idx < 0 or end_idx >= num_tokens or start_idx > end_idx:
                        continue
                    protected_mask[start_idx:end_idx+1] = True
                else:
                    idx = int(region)
                    if 0 <= idx < num_tokens:
                        protected_mask[idx] = True
        except Exception:
            return torch.zeros(num_tokens, dtype=torch.bool)
        return protected_mask
    
    def _generate_random_protection_mask(self, seed, num_tokens, device):
        generator = torch.Generator(device=device)
        generator.manual_seed((seed ^ 0x5EED) & 0xFFFFFFFFFFFFFFFF)
        num_to_protect = int(num_tokens * 0.3)
        if num_to_protect <= 0 and num_tokens > 0:
            num_to_protect = 1
        protected_indices = torch.randperm(num_tokens, generator=generator, device=device)[:num_to_protect]
        mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        mask[protected_indices] = True
        return mask
    
    def _legacy_protection_to_mask(self, protect_mode, num_tokens):
        protect_config = self.PROTECT_OPTIONS.get(protect_mode, (0.0, "start"))
        protect_fraction, protect_position = protect_config
        protected_count = int(num_tokens * protect_fraction)
        protected_mask = torch.zeros(num_tokens, dtype=torch.bool)
        if protected_count > 0:
            if protect_position == "end":
                protected_mask[num_tokens - protected_count:] = True
            else:
                protected_mask[:protected_count] = True
        return protected_mask

    def _pick_target_chunk(self, target_vibe, source_chunk_index, source_tensor):
        if not target_vibe or len(target_vibe) == 0:
            return None
        chunk_index = min(source_chunk_index, len(target_vibe) - 1)
        target_tensor = target_vibe[chunk_index][0]
        src_embed = source_tensor.shape[-1]
        tgt_embed = target_tensor.shape[-1]
        if src_embed != tgt_embed:
            return None
        return target_tensor

    def _prepare_vibe_tensor(self, target_tensor, source_tensor):
        device = source_tensor.device
        dtype  = source_tensor.dtype
        tgt = target_tensor.to(device=device, dtype=torch.float32)
        src = source_tensor.to(dtype=torch.float32)
        squeezed = False
        if tgt.dim() == 2:
            tgt = tgt.unsqueeze(0)
        if src.dim() == 2:
            src      = src.unsqueeze(0)
            squeezed = True
        batch_size, src_tokens, embed_dim = src.shape
        if tgt.shape[0] != batch_size:
            tgt = tgt.expand(batch_size, -1, -1)
        tgt_tokens = tgt.shape[1]
        if tgt_tokens != src_tokens:
            tgt = tgt.permute(0, 2, 1)
            tgt = torch.nn.functional.interpolate(tgt, size=src_tokens, mode="linear", align_corners=False)
            tgt = tgt.permute(0, 2, 1)
        if squeezed:
            tgt = tgt.squeeze(0)
        return tgt.to(dtype=dtype)

    def _apply_base_rebalance(self, cond_tensor, pattern, model_type):
        if model_type != "📸 Krea2 (SingleStream)":
            return cond_tensor
            
        embed_dim = cond_tensor.shape[-1]
        if embed_dim % 12 != 0:
            return cond_tensor
            
        if pattern == "visceral_grit":
            base_multipliers = [1.0, 1.0, 1.0, 1.8, 2.2, 2.2, 1.8, 1.4, 2.5, 2.2, 2.8, 1.4]
        elif pattern == "wild_facial_asymmetry":
            base_multipliers = [1.05, 1.05, 1.10, 1.60, 2.00, 1.80, 2.20, 2.20, 3.20, 2.40, 1.80, 1.10]
        elif pattern == "expression":
            base_multipliers = [1.0, 1.0, 1.0, 2.0, 2.4, 2.4, 3.2, 1.0, 1.0, 1.0, 1.0, 1.0]
        else:
            return cond_tensor
            
        band_width = embed_dim // 12
        band_tensor = torch.tensor(base_multipliers, dtype=cond_tensor.dtype, device=cond_tensor.device)
        band_tensor = band_tensor.repeat_interleave(band_width)
        return cond_tensor * band_tensor

    def _apply_noise_to_pooled(self, pooled_tensor, randomize_percent, strength, direction_config, seed, model_type="⚙️ Other"):
        """
        Apply proportional variance noise to SDXL global pooled_output vector.
        """
        if pooled_tensor is None or not isinstance(pooled_tensor, torch.Tensor):
            return pooled_tensor
            
        modified = pooled_tensor.clone()
        ref_std = torch.std(modified).detach().clamp(min=1e-5)
        generator = torch.Generator(device=pooled_tensor.device)
        generator.manual_seed(seed)
        
        pattern = direction_config[0] if direction_config is not None else "random"
        dir_mult = direction_config[1] if direction_config is not None else 1.0
        
        num_values = modified.numel()
        noise = self._generate_directional_noise(num_values, pattern, strength * dir_mult, generator, pooled_tensor.device, pooled_tensor.dtype)
        noise = noise.view_as(modified)
        
        noise_std = torch.std(noise).detach().clamp(min=1e-6)
        scale_factor = (strength / 35.0) * ref_std
        scaled_noise = (noise / noise_std) * scale_factor
        scaled_noise = torch.clamp(scaled_noise, -ref_std * 1.5, ref_std * 1.5)
        
        return modified + scaled_noise.to(dtype=modified.dtype)

    def _apply_noise(self, tensor, randomize_percent, strength, protection_mask,
                     direction_config, target_tensor, fade_curve, seed, vibe_blend=0.5, model_type="⚙️ Other",
                     structure_freeze=False, negpip_modifiers=None):
        """
        Calibrated noise injection matching DiT architecture without secondary standard-deviation attenuation.
        """
        modified = tensor.clone()
        generator = torch.Generator(device=tensor.device)
        generator.manual_seed(seed)

        if direction_config is not None:
            pattern, dir_multiplier = direction_config
            strength *= dir_multiplier
        else:
            pattern = "random"

        is_2d = (modified.dim() == 2)
        if is_2d:
            modified = modified.unsqueeze(0)

        batch_size, num_tokens, embed_dim = modified.shape
        if protection_mask.shape[0] != num_tokens:
            protection_mask = torch.zeros(num_tokens, dtype=torch.bool, device=tensor.device)

        protection_mask = protection_mask.to(tensor.device)
        modifiable_mask = ~protection_mask

        if modifiable_mask.any():
            batch_mask = modifiable_mask.unsqueeze(0).unsqueeze(-1).expand(batch_size, num_tokens, embed_dim)
            modifiable_values = modified[batch_mask]
            total_values = modifiable_values.numel()
            num_to_modify = int(total_values * (randomize_percent / 100.0))

            if num_to_modify > 0:
                num_mod_tokens = modifiable_mask.sum().item()
                indices = torch.randperm(total_values, generator=generator, device=tensor.device)[:num_to_modify]

                if target_tensor is not None:
                    tgt = self._prepare_vibe_tensor(target_tensor, modified)
                    target_values = tgt[batch_mask]
                    raw_dir = (target_values - modifiable_values) * vibe_blend

                    pattern_weight = 1.0 - vibe_blend
                    if pattern_weight > 0 and (pattern != "random" or direction_config is not None):
                        pattern_noise = self._generate_directional_noise(
                            num_to_modify, pattern, strength * pattern_weight, generator, tensor.device, tensor.dtype
                        )
                        sampled_vibe = raw_dir[indices] + pattern_noise
                    else:
                        sampled_vibe = raw_dir[indices]
                    noise = sampled_vibe
                else:
                    noise = self._generate_directional_noise(
                        num_to_modify, pattern, strength, generator, tensor.device, tensor.dtype
                    )

                if negpip_modifiers is not None:
                    token_indices = (indices // embed_dim) % num_mod_tokens
                    modifiable_mod_tokens = negpip_modifiers[modifiable_mask]
                    noise = noise * modifiable_mod_tokens[token_indices]

                if structure_freeze:
                    freeze_dim = max(1, int(embed_dim * 0.25))
                    freeze_mask = torch.ones(embed_dim, dtype=tensor.dtype, device=tensor.device)
                    freeze_mask[:freeze_dim] = 0.0
                    feature_indices = indices % embed_dim
                    noise = noise * freeze_mask[feature_indices]

                token_envelope = self._generate_fade_envelope(num_mod_tokens, fade_curve, tensor.device)
                full_envelope = token_envelope.repeat_interleave(embed_dim).repeat(batch_size)
                noise = noise * full_envelope[indices]

                modifiable_values[indices] += noise.to(dtype=modifiable_values.dtype)
                modified[batch_mask] = modifiable_values

        if is_2d:
            modified = modified.squeeze(0)
        return modified

    def _generate_fade_envelope(self, num_values, fade_curve, device):
        t = torch.linspace(0, 1, num_values, device=device, dtype=torch.float32)
        if fade_curve == "Instant":
            return torch.ones(num_values, device=device)
        elif fade_curve == "Linear":
            return 1.0 - t
        elif fade_curve == "Ease-Out":
            return 1.0 - t * t
        elif fade_curve == "Ease-In":
            return (1.0 - t) ** 2
        elif fade_curve == "Ease-In-Out":
            return torch.where(t < 0.5, 1.0 - 2 * t * t, 2 * (1.0 - t) ** 2)
        elif fade_curve == "Smooth Step":
            smooth = 3 * t * t - 2 * t * t * t
            return 1.0 - smooth
        elif fade_curve == "Burst":
            return torch.exp(-4 * t)
        else:
            return torch.ones(num_values, device=device)

    def _generate_directional_noise(self, num_values, pattern, strength, generator, device, dtype=torch.float32):
        if pattern == "random":
            res = torch.randn(num_values, device=device, generator=generator) * strength
        elif pattern == "scatter":
            base_noise = torch.randn(num_values, device=device, generator=generator)
            res = (base_noise * torch.abs(base_noise)) * strength
        elif pattern == "compress":
            base_noise = torch.randn(num_values, device=device, generator=generator)
            res = torch.tanh(base_noise) * strength * 0.5
        elif pattern == "wave":
            indices = torch.arange(num_values, device=device, dtype=torch.float32)
            wave = torch.sin(indices * 0.1) * strength
            wave += torch.randn(num_values, device=device, generator=generator) * strength * 0.3
            res = wave
        elif pattern == "sharpen":
            base = torch.randn(num_values, device=device, generator=generator)
            detail = torch.randn(num_values, device=device, generator=generator) * 0.35
            contrast = base * torch.pow(torch.abs(base) + 1e-6, 0.5)
            combined = (base * 0.55) + (detail * 0.30) + (contrast * 0.85)
            res = (combined / (combined.std() + 1e-6)) * strength
        elif pattern == "positive":
            base_noise = torch.randn(num_values, device=device, generator=generator)
            res = (torch.abs(base_noise) * 0.7 + base_noise * 0.3) * strength
        elif pattern == "negative":
            base_noise = torch.randn(num_values, device=device, generator=generator)
            res = (-torch.abs(base_noise) * 0.7 + base_noise * 0.3) * strength
        elif pattern == "smooth":
            base_noise = torch.randn(num_values, device=device, generator=generator)
            smoothed = base_noise.clone()
            if num_values > 2:
                smoothed[1:-1] = (base_noise[:-2] + base_noise[1:-1] + base_noise[2:]) / 3
            res = smoothed * strength
        elif pattern == "spatial":
            base_noise = torch.randn(num_values, device=device, generator=generator)
            chunk_size = max(1, num_values // 16)
            spatial_noise = base_noise.clone()
            for i in range(0, num_values, chunk_size):
                end = min(i + chunk_size, num_values)
                chunk_offset = torch.randn(1, device=device, generator=generator).item()
                spatial_noise[i:end] += chunk_offset * 0.5
            res = spatial_noise * strength
        elif pattern == "gradient":
            indices = torch.arange(num_values, device=device, dtype=torch.float32)
            normalized = indices / max(1, num_values - 1)
            direction = torch.randn(1, device=device, generator=generator).item()
            gradient = (normalized - 0.5) * direction * 2
            gradient += torch.randn(num_values, device=device, generator=generator) * 0.3
            res = gradient * strength
        elif pattern == "diversity":
            base_noise = (torch.rand(num_values, device=device, generator=generator) * 2.0) - 1.0
            res = base_noise * strength
        elif pattern == "facevar":
            base = torch.randn(num_values, device=device, generator=generator)
            jitter = torch.randn(num_values, device=device, generator=generator) * 0.35
            curved = torch.sign(base) * torch.pow(torch.abs(base), 1.4)
            combined = (base * 0.55) + (jitter * 0.25) + (curved * 0.85)
            res = (combined / (combined.std() + 1e-6)) * strength
        elif pattern == "visceral_grit":
            base = torch.randn(num_values, device=device, generator=generator)
            spikes = torch.pow(base, 3.0) * 0.7
            raw = torch.randn(num_values + 1, device=device, generator=generator)
            high_freq = (raw[1:] - raw[:-1]) / 1.414 * 0.5
            res = (spikes + high_freq) * strength
        elif pattern == "wild_facial_asymmetry":
            base = torch.randn(num_values, device=device, generator=generator)
            asymmetric_spikes = torch.tanh(torch.pow(base, 3.0) * 0.8) * 1.50 + base * 0.30
            raw = torch.randn(num_values + 1, device=device, generator=generator)
            diff_jitter = (raw[1:] - raw[:-1]) / 1.414 * 0.50
            combined = asymmetric_spikes + diff_jitter
            res = (combined / (combined.std() + 1e-6)) * strength
        elif pattern == "semantic_drift":
            shift = torch.randn(1, device=device, generator=generator).item() * 0.15
            jitter = torch.randn(num_values, device=device, generator=generator) * 0.05
            res = (torch.full((num_values,), shift, device=device) + jitter) * strength
        elif pattern == "structural_lock":
            t = torch.linspace(0, 1, num_values, device=device)
            decay = torch.where(t < 0.2, torch.ones_like(t), torch.exp(-5.0 * (t - 0.2)))
            base = torch.randn(num_values, device=device, generator=generator)
            res = base * decay * strength
        elif pattern == "cinematic_framing":
            t = torch.linspace(-1, 1, num_values, device=device)
            gradient = t
            center_bias = torch.exp(-2.0 * t**2)
            combined = (gradient * 0.6) + (center_bias * 0.4)
            combined += torch.randn(num_values, device=device, generator=generator) * 0.2
            res = combined * strength
        elif pattern == "identity_stretch":
            base = torch.randn(num_values, device=device, generator=generator)
            abs_base = torch.abs(base)
            mid_mask = (abs_base > 0.5) & (abs_base < 1.5)
            curvature = torch.sign(base) * torch.pow(abs_base - 0.5, 2) * 0.5
            result = base.clone()
            result[mid_mask] += curvature[mid_mask]
            res = result * strength
        elif pattern == "texture_lift":
            raw = torch.randn(num_values + 1, device=device, generator=generator)
            high_freq = (raw[1:] - raw[:-1]) / 1.414
            res = high_freq * strength
        elif pattern == "pink":
            white = torch.randn(num_values, device=device, generator=generator)
            fft = torch.fft.rfft(white)
            freqs = torch.arange(len(fft), device=device, dtype=torch.float32)
            freqs[0] = 1.0
            scale = 1.0 / torch.sqrt(freqs)
            scale[0] = 0.0
            pink = torch.fft.irfft(fft * scale, n=num_values)
            res = (pink / (pink.std() + 1e-6)) * strength
        elif pattern == "landscape_depth":
            indices = torch.arange(num_values, device=device, dtype=torch.float32)
            normalized = indices / max(1, num_values - 1)
            depth_curve = torch.log(normalized + 0.1) 
            res = ((depth_curve - depth_curve.mean()) / (depth_curve.std() + 1e-6)) * strength
        elif pattern == "group_diversity":
            base = torch.randn(num_values, device=device, generator=generator)
            selector = torch.randint(0, 3, (num_values,), device=device, generator=generator)
            shifts = torch.tensor([0.0, -1.5, 1.5], device=device)[selector]
            res = (base + shifts) * strength * 0.7
        elif pattern == "expression":
            base = torch.randn(num_values, device=device, generator=generator)
            spikes = torch.pow(base, 3.0)
            res = (spikes / (spikes.std() + 1e-6)) * strength
        elif pattern == "motion_blur":
            white_noise = torch.randn(num_values, device=device, generator=generator)
            if num_values > 4:
                blurred = torch.zeros_like(white_noise)
                for k in range(-2, 3):
                    blurred += torch.roll(white_noise, k, dims=0)
                blurred /= 5.0
                res = (blurred / (blurred.std() + 1e-6)) * strength
            else:
                res = white_noise * strength
        elif pattern == "microscopic":
            uniform = (torch.rand(num_values, device=device, generator=generator) * 2.0) - 1.0
            jitter = torch.randn(num_values, device=device, generator=generator) * 0.5
            res = (uniform + jitter) * strength
        elif pattern == "cosmic":
            base = torch.zeros(num_values, device=device)
            w = 1.0
            total_w = 0.0
            for _ in range(3):
                octave_noise = torch.randn(num_values, device=device, generator=generator)
                base += octave_noise * w
                total_w += w
                w *= 0.6
            base /= total_w
            res = torch.sinh(base) * strength
        elif pattern == "latent_noise_guard":
            white_noise = torch.randn(num_values, device=device, generator=generator)
            if num_values > 4:
                smoothed = torch.zeros_like(white_noise)
                for k in range(-1, 2):
                    smoothed += torch.roll(white_noise, k, dims=0)
                smoothed /= 3.0
            else:
                smoothed = white_noise
            compressed = torch.tanh(smoothed * 0.7)
            res = (compressed + 0.05) * strength
        elif pattern == "qwen_typo_guard":
            base_noise = torch.randn(num_values, device=device, generator=generator)
            if num_values > 6:
                smoothed = base_noise.clone()
                smoothed[1:-1] = 0.25 * base_noise[:-2] + 0.5 * base_noise[1:-1] + 0.25 * base_noise[2:]
            else:
                smoothed = base_noise
            clamped = torch.tanh(smoothed * 0.75) * 0.95
            res = (clamped / (clamped.std() + 1e-6)) * strength
        elif pattern == "qwen_filmic_contrast":
            base = torch.randn(num_values, device=device, generator=generator)
            s_curve = base * (1.0 + 0.35 * torch.pow(base, 2))
            s_curve = s_curve / (s_curve.std() + 1e-6)
            raw = torch.randn(num_values + 1, device=device, generator=generator)
            micro_grain = (raw[1:] - raw[:-1]) / 1.414 * 0.30
            cool_shift = -0.04
            combined = (s_curve * 0.75) + micro_grain + cool_shift
            clamped = torch.tanh(combined * 0.80) * 1.10
            res = (clamped / (clamped.std() + 1e-6)) * strength
        elif pattern == "qwen_structural_anchor":
            base_noise = torch.randn(num_values, device=device, generator=generator)
            if num_values > 8:
                smoothed = base_noise.clone()
                smoothed[2:-2] = (
                    0.0625 * base_noise[:-4] +
                    0.2500 * base_noise[1:-3] +
                    0.3750 * base_noise[2:-2] +
                    0.2500 * base_noise[3:-1] +
                    0.0625 * base_noise[4:]
                )
                combined = (smoothed * 0.70) + (base_noise * 0.30)
            else:
                combined = base_noise
            compressed = torch.tanh(combined * 0.75)
            res = (compressed / (compressed.std() + 1e-6)) * strength
        elif pattern == "anatomical_coherence":
            base_noise = torch.randn(num_values, device=device, generator=generator)
            if num_values > 8:
                smoothed = base_noise.clone()
                for _ in range(2):
                    if num_values > 2:
                        smoothed_pass = torch.zeros_like(smoothed)
                        for i in range(num_values):
                            start = max(0, i - 1)
                            end = min(num_values, i + 2)
                            smoothed_pass[i] = smoothed[start:end].mean()
                        smoothed = smoothed_pass
            else:
                smoothed = base_noise
            compressed = torch.tanh(smoothed * 0.8)
            structural_bias = torch.randn(1, device=device, generator=generator).item() * 0.05
            combined = compressed + structural_bias
            res = (combined / (combined.std() + 1e-6)) * strength
        else:
            res = torch.randn(num_values, device=device, generator=generator) * strength

        return res.to(dtype=dtype)


NODE_CLASS_MAPPINGS = {
    "RBG_Smart_Seed_Variance": RBG_Smart_Seed_Variance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RBG_Smart_Seed_Variance": "RBG Smart Seed Variance 🌱",
}