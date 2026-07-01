"""
RBG Smart Seed Variance 🌱
A user-friendly ComfyUI node for enhancing seed diversity in Z-Image Turbo and Qwen-Image.
"""

import torch
import random
import math

try:
    import node_helpers
    _NODE_HELPERS_AVAILABLE = True
except ImportError:
    _NODE_HELPERS_AVAILABLE = False


class RBG_Smart_Seed_Variance:
    """
    Adds diversity to outputs by injecting noise into text embeddings during early generation steps.
    Designed for Z-Image Turbo and Qwen-Image which have low seed variance.
    """
    
    # Preset configurations: (randomize_percent, strength)
    PRESETS = {
        "❌ Disabled": (0.0, 0.0),
        "🌱 Subtle": (1.0, 10.0),
        "🌿 Balanced": (2.0, 20.0),
        "🪴 Creative": (3.0, 30.0),
        "🌳 Bold": (4.0, 40.0),
        "🌴 Wild": (5.0, 50.0),
        "⚙️ Custom": None,  # Use fine_tune slider
    }
    
    # Model-specific adjustments: (strength_multiplier, randomize_multiplier)
    MODEL_ADJUSTMENTS = {
        "⚡ Z-Image Turbo": (1.0, 1.0),      # Baseline - well tested with strength 15-30
        "📸 Krea2 (SingleStream)": (1.0, 0.95), # Custom preset added for Krea2 models
        "🖼️ Qwen-Image": (1.0, 0.9),         # Similar architecture to Z-Image
        "🔮 Flux (Dev/Schnell)": (0.5, 0.8), # Dual encoder (CLIP + T5) - more sensitive
        "🎨 Chroma HD": (0.05, 0.5),         # Very sensitive - needs strength < 1
        "🧧 ERNIE-Image": (0.08, 0.45),      # Very sensitive - collapses past Creative. Trained on Baidu corpus.
        "🖌️ SDXL": (0.6, 0.8),               # Dual CLIP encoder - moderate sensitivity
        "🎬 Wan2.2": (0.8, 0.8),             # Video model - conservative
        "⚙️ Other": (0.8, 0.8),              # Conservative default
    }
    
    # Protect prompt options: (fraction to mask, position)
    # Position: "start" protects from beginning, "end" protects from end
    PROTECT_OPTIONS = {
        "🚫 None": (0.0, "start"),
        "First Quarter": (0.25, "start"),
        "First Half": (0.50, "start"),
        "Last Quarter": (0.25, "end"),
        "Last Half": (0.50, "end"),
    }
    
    # Step-based noise injection options
    NOISE_INJECTION = [
        "🚫 None",             # Apply to all (simple embedding modification)
        "Beginning Steps",  # Noise on early steps only (better for composition)
        "Ending Steps",     # Noise on late steps only (affects details)
        "All Steps",        # Noise throughout entire generation
    ]
    
    # Embedding Direction Shift options
    DIRECTION_SHIFTS = {
        "🚫 None": None,                    # Pure random noise (default behavior)
        "🌀 Chaos": ("scatter", 1.2),       # Increase entropy/randomness
        "📐 Order": ("compress", 0.8),      # Reduce variance, more consistent
        "🎨 Abstract": ("wave", 1.0),       # Artistic, painterly direction
        "📸 Realistic": ("sharpen", 0.9),   # Push toward photorealism
        "🌈 Vibrant": ("positive", 1.1),    # More colorful/saturated direction
        "🌑 Moody": ("negative", 1.0),      # Darker, moodier direction
        "💭 Dreamy": ("smooth", 1.1),       # Soft, ethereal direction
        "🎭 Dynamic Pose": ("spatial", 1.2),  # Varied poses/actions. Block-based noise.
        "🖼️ Composition": ("gradient", 1.0),  # Different layouts/framing
        "🌎 Diversity": ("diversity", 1.15),  # Simple uniform noise
        "🧬 Face-Variance Expansion": ("facevar", 1.25),  # Advanced curvature noise
        "🗿 Visceral Expression & Grit (Krea2)": ("visceral_grit", 1.2), # Custom engineered Krea2 emotional & texture lift
        "🧭 Semantic Drift (Centroid-Safe)": ("semantic_drift", 1.0), # Small constant shift
        "🧱 Structural Lock": ("structural_lock", 1.0), # Decaying noise structure
        "🎞️ Cinematic Framing": ("cinematic_framing", 1.1), # Gradient + center bias
        "🪶 Texture Lift": ("texture_lift", 1.0), # Mid-range curvature
        "💡 Studio Portrait": ("spatial", 1.0), # Early-mid structural balance
        "🌸 Natural": ("pink", 1.0), # 1/f noise (Pink Noise)
        "🏞️ Landscape Depth": ("landscape_depth", 1.1), # Depth-based gradient
        "👥 Group Diversity": ("group_diversity", 1.25), # Multi-modal noise
        "🎭 Expression Variance": ("expression", 1.2), # Facial expression focus
        "🌊 Motion Blur": ("motion_blur", 1.1), # Directional streak patterns
        "🔬 Microscopic": ("microscopic", 1.3), # Ultra-high-freq detail
        "🌌 Cosmic": ("cosmic", 1.2), # Fractal-based noise
        "☠️ Bone Anatomical Coherence": ("anatomical_coherence", 0.6), # Low-frequency smoothing
    }
    
    # Fade curves for noise application
    FADE_CURVES = [
        "Instant",      # Sharp cutoff, no fade
        "Linear",       # Gradual linear fade
        "Ease-Out",     # Fast start, slow fade (common)
        "Ease-In",      # Slow start, fast fade
        "Ease-In-Out",  # Smooth both ends
        "Smooth Step",  # Cubic smooth (Ken Perlin style)
        "Burst",        # Aggressive initial noise, quick drop
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
                    "tooltip": "When to apply noise during generation. Beginning Steps = more composition variety, Ending Steps = more detail variety."
                }),
                "protect_mode": (["🚫 None", "First Quarter", "First Half", "Last Quarter", "Last Half", "⚙️ Custom Regions", "🎲 Random Regions"], {
                    "default": "🚫 None",
                    "tooltip": "Protection mode: use preset regions, define custom token ranges, or protect random tokens."
                }),
                "protect_regions": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Custom protection regions (e.g., '0-5,15-20'). Only used when mode is 'Custom Regions'. Format: single tokens (5) or ranges (0-5), comma-separated."
                }),
                "direction_shift": (list(cls.DIRECTION_SHIFTS.keys()), {
                    "default": "🚫 None",
                    "tooltip": "Apply directional bias to embeddings instead of pure random noise. Creates predictable artistic shifts."
                }),
                "shift_strength": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "display": "slider",
                    "tooltip": "Strength of the direction shift effect (0-200%). 100% = default, 0% = disabled, 200% = double strength."
                }),
                "variance_schedule": (["constant", "decreasing", "step_cutoff", "hard_lock", "tiered_release"], {
                    "default": "constant",
                    "tooltip": "Composition Lock 🔒: Control how variance changes over time. 'constant'=standard, 'decreasing'=fade out, 'step_cutoff'=block switch, 'hard_lock'=zero variance until step, 'tiered_release'=multi-phase unlock."
                }),
                "cutoff_step": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 100,
                    "tooltip": "The step number where the cutoff or fade ends. (e.g. if you like composition at step 8, set this to 8)."
                }),
                "total_steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Estimate of total sampling steps. Required to map 'cutoff_step' to a timeline percentage."
                }),
                "cutoff_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider",
                    "tooltip": "The noise intensity multiplier after the cutoff step. 0.0 = no noise (lock), 1.0 = full noise."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for noise generation. Different seeds = different variance patterns."
                }),
            },
            "optional": {
                "target_vibe": ("CONDITIONING", {
                    "tooltip": (
                        "Optional: Connect a conditioning to steer variance direction.\n"
                        "The node computes a normalised per-token direction vector from source → target, "
                        "then blends it with your chosen direction_shift pattern.\n"
                        "Both work together — the vibe sets the direction, the pattern adds texture.\n"
                        "Multi-chunk targets are matched chunk-for-chunk with the source.\n"
                        "Use vibe_blend to control how strongly the target steers the output."
                    )
                }),
                "vibe_blend": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider",
                    "tooltip": (
                        "Controls the mix between target_vibe direction and your direction_shift pattern.\n"
                        "0.0 — direction_shift pattern only, target_vibe has no influence\n"
                        "0.5 — equal blend of vibe direction and pattern (default)\n"
                        "1.0 — pure vibe steering, direction_shift pattern silent\n"
                        "Has no effect when target_vibe is not connected."
                    )
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
                       target_vibe=None, vibe_blend=0.5):
        """
        Apply variance noise to conditioning embeddings with step-based control.
        """
        if not _NODE_HELPERS_AVAILABLE:
            raise RuntimeError(
                "RBG Smart Seed Variance: 'node_helpers' could not be imported. "
                "This module ships with ComfyUI — check your installation."
            )

        # Handle disabled mode
        if variance_preset == "❌ Disabled":
            return {"ui": {"protection_data": []}, "result": (conditioning, torch.zeros((1, 64, 64, 3)))}
        
        # Get preset values or calculate from fine_tune_variance
        preset_config = self.PRESETS.get(variance_preset)
        if preset_config is None:
            randomize_percent = (fine_tune_variance / 100.0) * 5.0  # 0-5%
            strength = (fine_tune_variance / 100.0) * 50.0  # 0-50
        else:
            randomize_percent, strength = preset_config
        
        # Apply model-specific adjustments
        strength_mult, randomize_mult = self.MODEL_ADJUSTMENTS.get(model_type, (1.0, 1.0))
        randomize_percent *= randomize_mult
        strength *= strength_mult
        
        # Get direction shift config and apply user strength
        direction_config = self.DIRECTION_SHIFTS.get(direction_shift, None)
        if direction_config is not None:
            pattern, preset_mult = direction_config
            user_mult = shift_strength / 100.0
            direction_config = (pattern, preset_mult * user_mult)

        # Create noisy conditioning
        noisy_conditioning = []
        protection_masks_for_ui = []
        heatmap_tensor = torch.zeros((1, 64, 64, 3))
        
        for i, cond in enumerate(conditioning):
            if len(cond) < 2:
                noisy_conditioning.append(cond)
                continue
            
            cond_tensor = cond[0]
            cond_dict = cond[1].copy() if len(cond) > 1 else {}
            
            # Determine number of tokens from tensor shape
            if len(cond_tensor.shape) == 3:
                num_tokens = cond_tensor.shape[1]
            elif len(cond_tensor.shape) == 2:
                num_tokens = cond_tensor.shape[0]
            else:
                noisy_conditioning.append(cond)
                continue
            
            # Generate protection mask based on mode
            if protect_mode == "⚙️ Custom Regions":
                protection_mask = self._parse_protection_regions(protect_regions, num_tokens)
            elif protect_mode == "🎲 Random Regions":
                protection_mask = self._generate_random_protection_mask(seed + i, num_tokens, cond_tensor.device)
            else:
                protection_mask = self._legacy_protection_to_mask(protect_mode, num_tokens)
            
            # Match target chunk
            current_target = self._pick_target_chunk(target_vibe, i, cond_tensor)

            pattern = direction_config[0] if direction_config is not None else "random"

            # Tweak 4: Apply base conditioning layer rebalancing to boost expression signals in base prompt
            rebalanced_cond_tensor = self._apply_base_rebalance(cond_tensor, pattern, model_type)

            # Apply noise with rebalanced tensor
            modified_tensor = self._apply_noise(
                rebalanced_cond_tensor,
                randomize_percent,
                strength,
                protection_mask,
                direction_config,
                current_target,
                fade_curve,
                seed + i,
                vibe_blend,
                model_type=model_type,
            )
            
            # Generate Heatmap for first conditioning chunk
            if i == 0:
                diff = (modified_tensor - cond_tensor).norm(dim=-1)
                max_diff = diff.max()
                if max_diff > 0: diff = diff / max_diff
                
                viz_width = diff.shape[1]
                heatmap_row = diff.view(1, 1, viz_width, 1).expand(1, 64, viz_width, 3)
                heatmap_tensor = torch.nn.functional.interpolate(heatmap_row.permute(0, 3, 1, 2), size=(64, 512), mode='nearest').permute(0, 2, 3, 1)
            
            # Store protection mask for UI
            if hasattr(protection_mask, 'tolist'):
                mask_list = (protection_mask.cpu().numpy().tolist()
                             if protection_mask.device.type != 'cpu'
                             else protection_mask.tolist())
                if mask_list and isinstance(mask_list[0], list):
                    mask_list = [item for sublist in mask_list for item in sublist]
            else:
                mask_list = []

            # Serialisable metadata
            cond_dict["rbg_variance_fade_curve"] = fade_curve
            cond_dict["rbg_variance_applied"] = True
            cond_dict["rbg_direction_shift"] = direction_shift
            cond_dict["rbg_noise_injection"] = noise_injection
            cond_dict["rbg_protect_mode"] = protect_mode
            if protect_mode == "⚙️ Custom Regions":
                cond_dict["rbg_protect_regions"] = protect_regions

            protection_masks_for_ui.append(mask_list)
            noisy_conditioning.append((modified_tensor, cond_dict))
        
        # --- Composition Lock Logic ---
        if variance_schedule != "constant":
            cutoff_percent = min(1.0, max(0.0, cutoff_step / total_steps))
            
            if variance_schedule == "step_cutoff":
                new_conditioning = node_helpers.conditioning_set_values(
                    noisy_conditioning, {"start_percent": 0.0, "end_percent": cutoff_percent}
                )
                if cutoff_strength > 0:
                    scaled_noisy = self._create_scaled_conditioning(
                        conditioning, randomize_percent, strength * cutoff_strength, target_vibe,
                        protect_mode, protect_regions, direction_config, fade_curve, seed, vibe_blend, model_type=model_type
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
                
                for i in range(num_segments):
                    seg_start = (i / num_segments) * cutoff_percent
                    seg_end = ((i + 1) / num_segments) * cutoff_percent
                    seg_multiplier = 1.0 - (i / num_segments) * (1.0 - cutoff_strength)
                    
                    seg_noisy = self._create_scaled_conditioning(
                        conditioning, randomize_percent, strength * seg_multiplier, target_vibe,
                        protect_mode, protect_regions, direction_config, fade_curve, seed, vibe_blend, model_type=model_type
                    )
                    new_conditioning += node_helpers.conditioning_set_values(
                        seg_noisy, {"start_percent": seg_start, "end_percent": seg_end}
                    )
                
                if cutoff_percent < 1.0:
                    final_noisy = self._create_scaled_conditioning(
                        conditioning, randomize_percent, strength * cutoff_strength, target_vibe,
                        protect_mode, protect_regions, direction_config, fade_curve, seed, vibe_blend, model_type=model_type
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
                        protect_mode, protect_regions, direction_config, fade_curve, seed, vibe_blend, model_type=model_type
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
                    protect_mode, protect_regions, direction_config, fade_curve, seed, vibe_blend, model_type=model_type
                )
                new_conditioning = node_helpers.conditioning_set_values(
                    phase1_noisy, {"start_percent": 0.0, "end_percent": cutoff_percent}
                )

                phase2_noisy = self._create_scaled_conditioning(
                    conditioning, randomize_percent * 0.7, strength * 0.6, target_vibe,
                    protect_mode, protect_regions, direction_config, fade_curve, seed + 1, vibe_blend, model_type=model_type
                )
                new_conditioning += node_helpers.conditioning_set_values(
                    phase2_noisy, {"start_percent": cutoff_percent, "end_percent": phase2_end}
                )

                if phase2_end < 1.0:
                    new_conditioning += node_helpers.conditioning_set_values(
                        noisy_conditioning, {"start_percent": phase2_end, "end_percent": 1.0}
                    )
                return {"ui": {"protection_data": protection_masks_for_ui}, "result": (new_conditioning, heatmap_tensor)}

        # --- Standard Noise Injection Logic ---
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
                                    direction_config, fade_curve, seed, vibe_blend=0.5, model_type="⚙️ Other"):
        scaled_conditioning = []
        for i, cond in enumerate(conditioning):
            if len(cond) < 2:
                scaled_conditioning.append(cond)
                continue
            cond_tensor = cond[0]
            cond_dict   = cond[1].copy()

            if len(cond_tensor.shape) == 3:
                num_tokens = cond_tensor.shape[1]
            elif len(cond_tensor.shape) == 2:
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

            # Tweak 4: Apply base conditioning layer rebalancing
            rebalanced_cond_tensor = self._apply_base_rebalance(cond_tensor, pattern, model_type)

            modified_tensor = self._apply_noise(
                rebalanced_cond_tensor, randomize_percent, strength, protection_mask,
                direction_config, current_target, fade_curve, seed + i, vibe_blend, model_type=model_type
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
            
        # Determine base multipliers for Krea2 to boost expressions/textures
        if pattern == "visceral_grit":
            # Composition (1-3) neutral; Emotion (4-7) heavily boosted; Detail/Grit (8-12) heavily boosted
            base_multipliers = [1.0, 1.0, 1.0, 2.2, 2.5, 2.5, 2.2, 1.8, 2.8, 2.5, 3.2, 1.8]
        elif pattern == "expression":
            # Boost facial expression / emotional layers
            base_multipliers = [1.0, 1.0, 1.0, 2.5, 2.8, 2.8, 2.2, 1.0, 1.0, 1.0, 1.0, 1.0]
        else:
            return cond_tensor
            
        band_width = embed_dim // 12
        band_tensor = torch.tensor(base_multipliers, dtype=cond_tensor.dtype, device=cond_tensor.device)
        band_tensor = band_tensor.repeat_interleave(band_width) # [embed_dim]
        
        return cond_tensor * band_tensor

    def _apply_noise(self, tensor, randomize_percent, strength, protection_mask,
                     direction_config, target_tensor, fade_curve, seed, vibe_blend=0.5, model_type="⚙️ Other"):
        """
        Apply noise to a fraction of the tensor values, optionally with directional bias and fade curve.
        """
        modified = tensor.clone()
        generator = torch.Generator(device=tensor.device)
        generator.manual_seed(seed)
        
        if direction_config is not None:
            pattern, dir_multiplier = direction_config
            strength *= dir_multiplier
        else:
            pattern = "random"
        
        if len(modified.shape) == 3:
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
                    if target_tensor is not None:
                        tgt = self._prepare_vibe_tensor(target_tensor, modified)
                        target_values = tgt[batch_mask]

                        raw_dir = target_values - modifiable_values
                        num_mod_tokens = modifiable_mask.sum().item()
                        raw_dir_2d = raw_dir.view(num_mod_tokens * batch_size, embed_dim)
                        norms = raw_dir_2d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        unit_dir = (raw_dir_2d / norms).view(-1)
                        vibe_noise = unit_dir * strength * vibe_blend

                        pattern_weight = 1.0 - vibe_blend
                        if pattern_weight > 0 and (pattern != "random" or direction_config is not None):
                            pattern_noise = self._generate_directional_noise(
                                vibe_noise.shape[0], pattern, strength * pattern_weight,
                                generator, tensor.device
                            )
                            noise = vibe_noise + pattern_noise
                        else:
                            noise = vibe_noise

                        # Tweak 2: Krea 2 Band-Aware Noise scaling (3D target vibe path)
                        if model_type == "📸 Krea2 (SingleStream)" and embed_dim % 12 == 0:
                            band_width = embed_dim // 12
                            if pattern == "spatial":
                                band_multipliers = [1.5, 1.5, 1.3, 1.2, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1]
                            elif pattern == "texture_lift":
                                band_multipliers = [0.1, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.5]
                            elif pattern == "visceral_grit":
                                band_multipliers = [0.0, 0.0, 0.1, 1.3, 1.5, 1.5, 1.2, 1.1, 1.5, 1.5, 1.5, 1.5]
                            else:
                                band_multipliers = [1.0] * 12
                            band_tensor = torch.tensor(band_multipliers, dtype=tensor.dtype, device=tensor.device)
                            band_tensor = band_tensor.repeat_interleave(band_width)
                            band_scale = band_tensor.repeat(num_mod_tokens * batch_size)
                            noise = noise * band_scale

                        token_envelope = self._generate_fade_envelope(num_mod_tokens, fade_curve, tensor.device)
                        full_envelope = token_envelope.repeat_interleave(embed_dim).repeat(batch_size)
                        noise = noise * full_envelope

                        modifiable_values += noise
                        modified[batch_mask] = modifiable_values

                    else:
                        noise = self._generate_directional_noise(
                            num_to_modify, pattern, strength, generator, tensor.device
                        )
                        num_modifiable_tokens = modifiable_mask.sum().item()
                        token_envelope  = self._generate_fade_envelope(num_modifiable_tokens, fade_curve, tensor.device)
                        full_envelope   = token_envelope.repeat_interleave(embed_dim).repeat(batch_size)
                        indices         = torch.randperm(total_values, generator=generator, device=tensor.device)[:num_to_modify]
                        
                        # Tweak 2: Krea 2 Band-Aware Noise scaling (3D random path)
                        if model_type == "📸 Krea2 (SingleStream)" and embed_dim % 12 == 0:
                            band_width = embed_dim // 12
                            if pattern == "spatial":
                                band_multipliers = [1.5, 1.5, 1.3, 1.2, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1]
                            elif pattern == "texture_lift":
                                band_multipliers = [0.1, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.5]
                            elif pattern == "visceral_grit":
                                band_multipliers = [0.0, 0.0, 0.1, 1.3, 1.5, 1.5, 1.2, 1.1, 1.5, 1.5, 1.5, 1.5]
                            else:
                                band_multipliers = [1.0] * 12
                            band_tensor = torch.tensor(band_multipliers, dtype=tensor.dtype, device=tensor.device)
                            band_tensor = band_tensor.repeat_interleave(band_width)
                            
                            feature_indices = indices % embed_dim
                            band_scale = band_tensor[feature_indices]
                            noise = noise * band_scale

                        noise           = noise * full_envelope[indices]
                        modifiable_values[indices] += noise
                        modified[batch_mask] = modifiable_values

        elif len(modified.shape) == 2:
            num_tokens, embed_dim = modified.shape
            if protection_mask.shape[0] != num_tokens:
                protection_mask = torch.zeros(num_tokens, dtype=torch.bool, device=tensor.device)
            
            protection_mask = protection_mask.to(tensor.device)
            modifiable_mask = ~protection_mask
            
            if modifiable_mask.any():
                token_mask = modifiable_mask.unsqueeze(-1).expand(num_tokens, embed_dim)
                modifiable_values = modified[token_mask]
                total_values = modifiable_values.numel()
                num_to_modify = int(total_values * (randomize_percent / 100.0))
                
                if num_to_modify > 0:
                    if target_tensor is not None:
                        tgt = self._prepare_vibe_tensor(target_tensor, modified)
                        target_values = tgt[token_mask]

                        raw_dir = target_values - modifiable_values
                        num_mod_tokens = modifiable_mask.sum().item()
                        raw_dir_2d = raw_dir.view(num_mod_tokens, embed_dim)
                        norms = raw_dir_2d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        unit_dir = (raw_dir_2d / norms).view(-1)
                        vibe_noise = unit_dir * strength * vibe_blend

                        pattern_weight = 1.0 - vibe_blend
                        if pattern_weight > 0 and (pattern != "random" or direction_config is not None):
                            pattern_noise = self._generate_directional_noise(
                                vibe_noise.shape[0], pattern, strength * pattern_weight,
                                generator, tensor.device
                            )
                            noise = vibe_noise + pattern_noise
                        else:
                            noise = vibe_noise

                        # Tweak 2: Krea 2 Band-Aware Noise scaling (2D target vibe path)
                        if model_type == "📸 Krea2 (SingleStream)" and embed_dim % 12 == 0:
                            band_width = embed_dim // 12
                            if pattern == "spatial":
                                band_multipliers = [1.5, 1.5, 1.3, 1.2, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1]
                            elif pattern == "texture_lift":
                                band_multipliers = [0.1, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.5]
                            elif pattern == "visceral_grit":
                                band_multipliers = [0.0, 0.0, 0.1, 1.3, 1.5, 1.5, 1.2, 1.1, 1.5, 1.5, 1.5, 1.5]
                            else:
                                band_multipliers = [1.0] * 12
                            band_tensor = torch.tensor(band_multipliers, dtype=tensor.dtype, device=tensor.device)
                            band_tensor = band_tensor.repeat_interleave(band_width)
                            band_scale = band_tensor.repeat(num_mod_tokens)
                            noise = noise * band_scale

                        token_envelope = self._generate_fade_envelope(num_mod_tokens, fade_curve, tensor.device)
                        full_envelope  = token_envelope.repeat_interleave(embed_dim)
                        noise = noise * full_envelope

                        modifiable_values += noise
                        modified[token_mask] = modifiable_values

                    else:
                        noise = self._generate_directional_noise(
                            num_to_modify, pattern, strength, generator, tensor.device
                        )
                        num_modifiable_tokens = modifiable_mask.sum().item()
                        token_envelope  = self._generate_fade_envelope(num_modifiable_tokens, fade_curve, tensor.device)
                        full_envelope   = token_envelope.repeat_interleave(embed_dim)
                        indices         = torch.randperm(total_values, generator=generator, device=tensor.device)[:num_to_modify]
                        
                        # Tweak 2: Krea 2 Band-Aware Noise scaling (2D random path)
                        if model_type == "📸 Krea2 (SingleStream)" and embed_dim % 12 == 0:
                            band_width = embed_dim // 12
                            if pattern == "spatial":
                                band_multipliers = [1.5, 1.5, 1.3, 1.2, 1.0, 0.8, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1]
                            elif pattern == "texture_lift":
                                band_multipliers = [0.1, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 1.5]
                            elif pattern == "visceral_grit":
                                band_multipliers = [0.0, 0.0, 0.1, 1.3, 1.5, 1.5, 1.2, 1.1, 1.5, 1.5, 1.5, 1.5]
                            else:
                                band_multipliers = [1.0] * 12
                            band_tensor = torch.tensor(band_multipliers, dtype=tensor.dtype, device=tensor.device)
                            band_tensor = band_tensor.repeat_interleave(band_width)
                            
                            feature_indices = indices % embed_dim
                            band_scale = band_tensor[feature_indices]
                            noise = noise * band_scale

                        noise           = noise * full_envelope[indices]
                        modifiable_values[indices] += noise
                        modified[token_mask] = modifiable_values
        
        return modified
    
    def _generate_fade_envelope(self, num_values, fade_curve, device):
        t = torch.linspace(0, 1, num_values, device=device)
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
    
    def _generate_directional_noise(self, num_values, pattern, strength, generator, device):
        if pattern == "random":
            return torch.randn(num_values, device=device, generator=generator) * strength
        elif pattern == "scatter":
            base_noise = torch.randn(num_values, device=device, generator=generator)
            return (base_noise * torch.abs(base_noise)) * strength
        elif pattern == "compress":
            base_noise = torch.randn(num_values, device=device, generator=generator)
            return torch.tanh(base_noise) * strength * 0.5
        elif pattern == "wave":
            indices = torch.arange(num_values, device=device, dtype=torch.float32)
            wave = torch.sin(indices * 0.1) * strength
            wave += torch.randn(num_values, device=device, generator=generator) * strength * 0.3
            return wave
        elif pattern == "sharpen":
            base = torch.randn(num_values, device=device, generator=generator)
            detail = torch.randn(num_values, device=device, generator=generator) * 0.35
            contrast = base * torch.pow(torch.abs(base) + 1e-6, 0.5)
            combined = (base * 0.55) + (detail * 0.30) + (contrast * 0.85)
            combined = combined / (combined.std() + 1e-6)
            return combined * strength
        elif pattern == "positive":
            base_noise = torch.randn(num_values, device=device, generator=generator)
            return (torch.abs(base_noise) * 0.7 + base_noise * 0.3) * strength
        elif pattern == "negative":
            base_noise = torch.randn(num_values, device=device, generator=generator)
            return (-torch.abs(base_noise) * 0.7 + base_noise * 0.3) * strength
        elif pattern == "smooth":
            base_noise = torch.randn(num_values, device=device, generator=generator)
            smoothed = base_noise.clone()
            if num_values > 2:
                smoothed[1:-1] = (base_noise[:-2] + base_noise[1:-1] + base_noise[2:]) / 3
            return smoothed * strength
        elif pattern == "spatial":
            base_noise = torch.randn(num_values, device=device, generator=generator)
            chunk_size = max(1, num_values // 16)
            spatial_noise = base_noise.clone()
            for i in range(0, num_values, chunk_size):
                end = min(i + chunk_size, num_values)
                chunk_offset = torch.randn(1, device=device, generator=generator).item()
                spatial_noise[i:end] += chunk_offset * 0.5
            return spatial_noise * strength
        elif pattern == "gradient":
            indices = torch.arange(num_values, device=device, dtype=torch.float32)
            normalized = indices / max(1, num_values - 1)
            direction = torch.randn(1, device=device, generator=generator).item()
            gradient = (normalized - 0.5) * direction * 2
            gradient += torch.randn(num_values, device=device, generator=generator) * 0.3
            return gradient * strength
        elif pattern == "diversity":
            base_noise = (torch.rand(num_values, device=device, generator=generator) * 2.0) - 1.0
            return base_noise * strength
        elif pattern == "facevar":
            base = torch.randn(num_values, device=device, generator=generator)
            jitter = torch.randn(num_values, device=device, generator=generator) * 0.35
            curved = torch.sign(base) * torch.pow(torch.abs(base), 1.4)
            combined = (base * 0.55) + (jitter * 0.25) + (curved * 0.85)
            combined = combined / (combined.std() + 1e-6)
            return combined * strength
        elif pattern == "visceral_grit":
            base = torch.randn(num_values, device=device, generator=generator)
            spikes = torch.pow(base, 3.0) * 0.7
            raw = torch.randn(num_values + 1, device=device, generator=generator)
            high_freq = (raw[1:] - raw[:-1]) / 1.414 * 0.5
            combined = spikes + high_freq
            return combined * strength
        elif pattern == "semantic_drift":
            shift = torch.randn(1, device=device, generator=generator).item() * 0.15
            jitter = torch.randn(num_values, device=device, generator=generator) * 0.05
            return (torch.full((num_values,), shift, device=device) + jitter) * strength
        elif pattern == "structural_lock":
            t = torch.linspace(0, 1, num_values, device=device)
            decay = torch.where(t < 0.2, torch.ones_like(t), torch.exp(-5.0 * (t - 0.2)))
            base = torch.randn(num_values, device=device, generator=generator)
            return base * decay * strength
        elif pattern == "cinematic_framing":
            t = torch.linspace(-1, 1, num_values, device=device)
            gradient = t
            center_bias = torch.exp(-2.0 * t**2)
            combined = (gradient * 0.6) + (center_bias * 0.4)
            combined += torch.randn(num_values, device=device, generator=generator) * 0.2
            return combined * strength
        elif pattern == "identity_stretch":
            base = torch.randn(num_values, device=device, generator=generator)
            abs_base = torch.abs(base)
            mid_mask = (abs_base > 0.5) & (abs_base < 1.5)
            curvature = torch.sign(base) * torch.pow(abs_base - 0.5, 2) * 0.5
            result = base.clone()
            result[mid_mask] += curvature[mid_mask]
            return result * strength
        elif pattern == "texture_lift":
            raw = torch.randn(num_values + 1, device=device, generator=generator)
            high_freq = raw[1:] - raw[:-1]
            high_freq = high_freq / 1.414
            return high_freq * strength
        elif pattern == "pink":
            white = torch.randn(num_values, device=device, generator=generator)
            fft = torch.fft.rfft(white)
            freqs = torch.arange(len(fft), device=device, dtype=torch.float32)
            freqs[0] = 1.0
            scale = 1.0 / torch.sqrt(freqs)
            scale[0] = 0.0
            pink = torch.fft.irfft(fft * scale, n=num_values)
            pink = pink / (pink.std() + 1e-6)
            return pink * strength
        elif pattern == "landscape_depth":
            indices = torch.arange(num_values, device=device, dtype=torch.float32)
            normalized = indices / max(1, num_values - 1)
            depth_curve = torch.log(normalized + 0.1) 
            depth_curve = (depth_curve - depth_curve.mean()) / (depth_curve.std() + 1e-6)
            return depth_curve * strength
        elif pattern == "group_diversity":
            base = torch.randn(num_values, device=device, generator=generator)
            selector = torch.randint(0, 3, (num_values,), device=device, generator=generator)
            shifts = torch.tensor([0.0, -1.5, 1.5], device=device)[selector]
            return (base + shifts) * strength * 0.7
        elif pattern == "expression":
            base = torch.randn(num_values, device=device, generator=generator)
            spikes = torch.pow(base, 3.0)
            spikes = spikes / (spikes.std() + 1e-6)
            return spikes * strength
        elif pattern == "motion_blur":
            white_noise = torch.randn(num_values, device=device, generator=generator)
            if num_values > 4:
                blurred = torch.zeros_like(white_noise)
                for k in range(-2, 3):
                    blurred += torch.roll(white_noise, k, dims=0)
                blurred /= 5.0
                blurred = blurred / (blurred.std() + 1e-6)
                return blurred * strength
            else:
                 return white_noise * strength
        elif pattern == "microscopic":
            uniform = (torch.rand(num_values, device=device, generator=generator) * 2.0) - 1.0
            jitter = torch.randn(num_values, device=device, generator=generator) * 0.5
            combined = uniform + jitter
            return combined * strength
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
            base = torch.sinh(base) 
            return base * strength
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
            result = compressed + structural_bias
            result = result / (result.std() + 1e-6)
            return result * strength
        else:
            return torch.randn(num_values, device=device, generator=generator) * strength


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "RBG_Smart_Seed_Variance": RBG_Smart_Seed_Variance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RBG_Smart_Seed_Variance": "RBG Smart Seed Variance 🌱",
}