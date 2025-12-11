"""
RBG Smart Seed Variance 🌱
A user-friendly ComfyUI node for enhancing seed diversity in Z-Image Turbo and Qwen-Image.
"""

import torch
import random


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
        "🖼️ Qwen-Image": (1.0, 0.9),         # Similar architecture to Z-Image
        "🔮 Flux (Dev/Schnell)": (0.5, 0.8), # Dual encoder (CLIP + T5) - more sensitive
        "🎨 Chroma HD": (0.05, 0.5),         # Very sensitive - needs strength < 1
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
    # Each direction applies a structured bias pattern to the embeddings
    # Format: (dimension_bias_pattern, strength_multiplier)
    # dimension_bias_pattern: "positive", "negative", "alternating", "wave", etc.
    DIRECTION_SHIFTS = {
        "🚫 None": None,                    # Pure random noise (default behavior)
        "🌀 Chaos": ("scatter", 1.2),       # Increase entropy/randomness
        "📐 Order": ("compress", 0.8),      # Reduce variance, more consistent
        "🎨 Abstract": ("wave", 1.0),       # Artistic, painterly direction
        "📸 Realistic": ("sharpen", 0.9),   # Push toward photorealism
        "🌈 Vibrant": ("positive", 1.1),    # More colorful/saturated direction
        "🌑 Moody": ("negative", 1.0),      # Darker, moodier direction
        "💭 Dreamy": ("smooth", 1.1),       # Soft, ethereal direction
        "🎭 Dynamic Pose": ("spatial", 1.2),  # Varied poses/actions
        "🖼️ Composition": ("gradient", 1.0),  # Different layouts/framing
        "🌍 Diversity": ("diversity", 1.15),  # Uses uniform noise to break mode collapse/bias
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
                "protect_prompt": (list(cls.PROTECT_OPTIONS.keys()), {
                    "default": "🚫 None",
                    "tooltip": "Protect portions of your prompt from noise modification."
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
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Seed for noise generation. Different seeds = different variance patterns."
                }),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply_variance"
    CATEGORY = "RBG-Suite-Pack"
    
    def apply_variance(self, conditioning, variance_preset, fine_tune_variance, model_type, fade_curve,
                       noise_injection, protect_prompt, direction_shift, shift_strength, seed):
        """
        Apply variance noise to conditioning embeddings with step-based control.
        """
        import node_helpers
        
        # Handle disabled mode
        if variance_preset == "❌ Disabled":
            return (conditioning,)
        
        # Get preset values or calculate from fine_tune_variance
        preset_config = self.PRESETS.get(variance_preset)
        if preset_config is None:
            # Custom mode: map fine_tune_variance (0-100) to reasonable ranges
            randomize_percent = (fine_tune_variance / 100.0) * 5.0  # 0-5%
            strength = (fine_tune_variance / 100.0) * 50.0  # 0-50
        else:
            randomize_percent, strength = preset_config
        
        # Apply model-specific adjustments
        strength_mult, randomize_mult = self.MODEL_ADJUSTMENTS.get(model_type, (1.0, 1.0))
        randomize_percent *= randomize_mult
        strength *= strength_mult
        
        # Get protection mask config (fraction, position)
        protect_config = self.PROTECT_OPTIONS.get(protect_prompt, (0.0, "start"))
        protect_fraction, protect_position = protect_config
        
        # Get direction shift config and apply user strength
        direction_config = self.DIRECTION_SHIFTS.get(direction_shift, None)
        if direction_config is not None:
            pattern, preset_mult = direction_config
            # Apply user's direction strength (0-200 maps to 0.0-2.0 multiplier)
            user_mult = shift_strength / 100.0
            direction_config = (pattern, preset_mult * user_mult)
        
        # Create noisy conditioning
        noisy_conditioning = []
        
        for i, cond in enumerate(conditioning):
            # Each conditioning is a tuple: (tensor, dict)
            if len(cond) < 2:
                noisy_conditioning.append(cond)
                continue
            
            cond_tensor = cond[0]
            cond_dict = cond[1].copy() if len(cond) > 1 else {}
            
            # Apply noise to the embedding tensor with fade curve
            modified_tensor = self._apply_noise(
                cond_tensor, 
                randomize_percent, 
                strength, 
                protect_fraction,
                protect_position,
                direction_config,
                fade_curve,
                seed + i  # Offset seed for each conditioning
            )
            
            # Store metadata
            cond_dict["rbg_variance_fade_curve"] = fade_curve
            cond_dict["rbg_variance_applied"] = True
            cond_dict["rbg_direction_shift"] = direction_shift
            cond_dict["rbg_noise_injection"] = noise_injection
            
            noisy_conditioning.append((modified_tensor, cond_dict))
        
        # Apply step-based noise injection
        if noise_injection == "🚫 None" or noise_injection == "All Steps":
            # Simple: just return the noisy conditioning
            return (noisy_conditioning,)
        
        switchover = 0.20
        
        if noise_injection == "Beginning Steps":
            # Noisy embedding for first 20%, original for rest
            new_conditioning = node_helpers.conditioning_set_values(
                noisy_conditioning, {"start_percent": 0.0, "end_percent": switchover}
            )
            new_conditioning += node_helpers.conditioning_set_values(
                conditioning, {"start_percent": switchover, "end_percent": 1.0}
            )
        elif noise_injection == "Ending Steps":
            # Original for first 80%, noisy for last 20%
            new_conditioning = node_helpers.conditioning_set_values(
                conditioning, {"start_percent": 0.0, "end_percent": switchover}
            )
            new_conditioning += node_helpers.conditioning_set_values(
                noisy_conditioning, {"start_percent": switchover, "end_percent": 1.0}
            )
        else:
            # Fallback
            new_conditioning = noisy_conditioning
        
        return (new_conditioning,)
    
    def _apply_noise(self, tensor, randomize_percent, strength, protect_fraction, protect_position, direction_config, fade_curve, seed):
        """
        Apply noise to a fraction of the tensor values, optionally with directional bias and fade curve.
        
        Args:
            tensor: The embedding tensor to modify
            randomize_percent: Percentage of values to modify (0-100)
            strength: Scale of the noise to add
            protect_fraction: Fraction of tokens to protect (0-1)
            protect_position: "start" or "end" - which part of prompt to protect
            direction_config: Tuple of (pattern, multiplier) or None for random noise
            fade_curve: Type of fade curve to apply spatially across embedding
            seed: Random seed for reproducibility
        
        Returns:
            Modified tensor with noise applied
        """
        # Clone tensor to avoid modifying original
        modified = tensor.clone()
        
        # Set random seed for reproducibility
        generator = torch.Generator(device=tensor.device)
        generator.manual_seed(seed)
        
        # Extract direction pattern and multiplier
        if direction_config is not None:
            pattern, dir_multiplier = direction_config
            strength *= dir_multiplier
        else:
            pattern = "random"
        
        # Calculate dimensions
        if len(modified.shape) == 3:
            # Shape: (batch, tokens, embedding_dim)
            batch_size, num_tokens, embed_dim = modified.shape
            
            # Calculate protected token count based on position
            protected_count = int(num_tokens * protect_fraction)
            
            if protect_position == "end":
                # Protect from end: modify tokens from 0 to (num_tokens - protected_count)
                start_idx = 0
                end_idx = num_tokens - protected_count
            else:
                # Protect from start: modify tokens from protected_count to end
                start_idx = protected_count
                end_idx = num_tokens
            
            # Only modify if there are unprotected tokens
            if start_idx < end_idx:
                # Clone the slice to avoid memory aliasing issues
                unprotected = modified[:, start_idx:end_idx, :].clone()
                
                # Calculate number of values to modify
                total_values = unprotected.numel()
                num_to_modify = int(total_values * (randomize_percent / 100.0))
                
                if num_to_modify > 0:
                    # Generate noise based on pattern
                    noise = self._generate_directional_noise(
                        num_to_modify, pattern, strength, generator, tensor.device
                    )
                    
                    # Generate spatial fade envelope (across tokens)
                    token_count = unprotected.shape[1]
                    token_envelope = self._generate_fade_envelope(token_count, fade_curve, tensor.device)
                    
                    # Broadcast: (1, Tokens, 1) -> (Batch, Tokens, Dim)
                    # This ensures the fade is applied structurally across the sequence
                    token_envelope = token_envelope.view(1, token_count, 1)
                    full_envelope = token_envelope.expand(unprotected.shape).contiguous().flatten()
                    
                    # Generate random indices to modify
                    flat_unprotected = unprotected.flatten()
                    indices = torch.randperm(total_values, generator=generator)[:num_to_modify]
                    
                    # Apply spatial fade modifiers to the noise based on where it lands
                    spatial_modifiers = full_envelope[indices]
                    noise = noise * spatial_modifiers
                    
                    # Apply noise to selected indices
                    flat_unprotected[indices] += noise
                    
                    # Reshape and assign back
                    modified[:, start_idx:end_idx, :] = flat_unprotected.reshape(unprotected.shape)
        
        elif len(modified.shape) == 2:
            # Shape: (tokens, embedding_dim)
            num_tokens, embed_dim = modified.shape
            
            # Calculate protected token count based on position
            protected_count = int(num_tokens * protect_fraction)
            
            if protect_position == "end":
                start_idx = 0
                end_idx = num_tokens - protected_count
            else:
                start_idx = protected_count
                end_idx = num_tokens
            
            if start_idx < end_idx:
                # Clone the slice to avoid memory aliasing issues
                unprotected = modified[start_idx:end_idx, :].clone()
                
                total_values = unprotected.numel()
                num_to_modify = int(total_values * (randomize_percent / 100.0))
                
                if num_to_modify > 0:
                    # Generate noise based on pattern
                    noise = self._generate_directional_noise(
                        num_to_modify, pattern, strength, generator, tensor.device
                    )
                    
                    # Generate spatial fade envelope (across tokens)
                    token_count = unprotected.shape[0]
                    token_envelope = self._generate_fade_envelope(token_count, fade_curve, tensor.device)
                    
                    # Broadcast: (Tokens, 1) -> (Tokens, Dim)
                    token_envelope = token_envelope.view(token_count, 1)
                    full_envelope = token_envelope.expand(unprotected.shape).contiguous().flatten()
                    
                    # Generate random indices to modify
                    flat_unprotected = unprotected.flatten()
                    indices = torch.randperm(total_values, generator=generator)[:num_to_modify]
                    
                    # Apply spatial fade modifiers to the noise based on where it lands
                    spatial_modifiers = full_envelope[indices]
                    noise = noise * spatial_modifiers
                    
                    # Apply noise to selected indices
                    flat_unprotected[indices] += noise
                    modified[start_idx:end_idx, :] = flat_unprotected.reshape(unprotected.shape)
        
        return modified
    
    def _generate_fade_envelope(self, num_values, fade_curve, device):
        """
        Generate a fade envelope to spatially modulate noise intensity.
        
        Args:
            num_values: Number of envelope values to generate
            fade_curve: Type of fade curve to use
            device: Target device for tensor
        
        Returns:
            Tensor of multipliers (0-1) that modulate noise intensity
        """
        # Create normalized position (0 to 1)
        t = torch.linspace(0, 1, num_values, device=device)
        
        if fade_curve == "Instant":
            # No fade - full strength everywhere
            return torch.ones(num_values, device=device)
        
        elif fade_curve == "Linear":
            # Linear fade from 1 to 0
            return 1.0 - t
        
        elif fade_curve == "Ease-Out":
            # Fast start, slow fade (quadratic)
            return 1.0 - t * t
        
        elif fade_curve == "Ease-In":
            # Slow start, fast fade
            return (1.0 - t) ** 2
        
        elif fade_curve == "Ease-In-Out":
            # Smooth both ends (cubic)
            return torch.where(
                t < 0.5,
                1.0 - 2 * t * t,
                2 * (1.0 - t) ** 2
            )
        
        elif fade_curve == "Smooth Step":
            # Ken Perlin's smoothstep
            # smoothstep(t) = 3t² - 2t³
            smooth = 3 * t * t - 2 * t * t * t
            return 1.0 - smooth
        
        elif fade_curve == "Burst":
            # Aggressive initial, quick drop
            # exp(-4t) gives sharp decay
            return torch.exp(-4 * t)
        
        else:
            # Default to instant (no fade)
            return torch.ones(num_values, device=device)
    
    def _generate_directional_noise(self, num_values, pattern, strength, generator, device):
        """
        Generate noise with a specific directional pattern.
        
        Args:
            num_values: Number of noise values to generate
            pattern: The noise pattern type
            strength: Base strength multiplier
            generator: Torch random generator
            device: Target device for tensor
        
        Returns:
            Tensor of noise values with directional bias
        """
        if pattern == "random":
            # Pure random Gaussian noise (default behavior)
            return torch.randn(num_values, device=device, generator=generator) * strength
        
        elif pattern == "scatter":
            # Chaos: High variance, scattered noise
            base_noise = torch.randn(num_values, device=device, generator=generator)
            # Amplify outliers
            return (base_noise * torch.abs(base_noise)) * strength
        
        elif pattern == "compress":
            # Order: Low variance, compressed noise
            base_noise = torch.randn(num_values, device=device, generator=generator)
            # Compress to smaller range
            return torch.tanh(base_noise) * strength * 0.5
        
        elif pattern == "wave":
            # Abstract: Sinusoidal wave pattern
            indices = torch.arange(num_values, device=device, dtype=torch.float32)
            wave = torch.sin(indices * 0.1) * strength
            # Add small random variation
            wave += torch.randn(num_values, device=device, generator=generator) * strength * 0.3
            return wave
        
        elif pattern == "sharpen":
            # Realistic: Emphasize edges/details
            base_noise = torch.randn(num_values, device=device, generator=generator)
            # Sharpen by emphasizing extreme values
            return torch.sign(base_noise) * torch.pow(torch.abs(base_noise), 0.7) * strength
        
        elif pattern == "positive":
            # Vibrant: Bias toward positive values
            base_noise = torch.randn(num_values, device=device, generator=generator)
            return (torch.abs(base_noise) * 0.7 + base_noise * 0.3) * strength
        
        elif pattern == "negative":
            # Moody: Bias toward negative values
            base_noise = torch.randn(num_values, device=device, generator=generator)
            return (-torch.abs(base_noise) * 0.7 + base_noise * 0.3) * strength
        
        elif pattern == "smooth":
            # Dreamy: Smoothed, soft noise
            base_noise = torch.randn(num_values, device=device, generator=generator)
            # Apply smoothing by averaging with neighbors (simulated)
            smoothed = base_noise.clone()
            if num_values > 2:
                smoothed[1:-1] = (base_noise[:-2] + base_noise[1:-1] + base_noise[2:]) / 3
            return smoothed * strength
        
        elif pattern == "spatial":
            # Dynamic Pose: Block-based noise to encourage different poses/actions
            # Apply noise in chunks to affect structural elements
            base_noise = torch.randn(num_values, device=device, generator=generator)
            chunk_size = max(1, num_values // 16)  # 16 structural blocks
            spatial_noise = base_noise.clone()
            for i in range(0, num_values, chunk_size):
                end = min(i + chunk_size, num_values)
                # Apply random offset to each chunk
                chunk_offset = torch.randn(1, device=device, generator=generator).item()
                spatial_noise[i:end] += chunk_offset * 0.5
            return spatial_noise * strength
        
        elif pattern == "gradient":
            # Composition: Linear gradient noise to affect layout/framing
            # Creates directional bias that can shift composition
            indices = torch.arange(num_values, device=device, dtype=torch.float32)
            # Normalize to 0-1 range
            normalized = indices / max(1, num_values - 1)
            # Create gradient with random direction
            direction = torch.randn(1, device=device, generator=generator).item()
            gradient = (normalized - 0.5) * direction * 2
            # Add small random variation
            gradient += torch.randn(num_values, device=device, generator=generator) * 0.3
            return gradient * strength
        
        elif pattern == "diversity":
            # Diversity: Uniform distribution (vs Gaussian normal)
            # Gaussian clusters near 0 (average), Uniform treats extremes equally.
            # Helps break "mode collapse" (bias toward average faces/ethnicity).
            base_noise = torch.rand(num_values, device=device, generator=generator)
            # Shift 0..1 to -1..1
            return (base_noise - 0.5) * 2.0 * strength
        
        else:
            # Fallback to random
            return torch.randn(num_values, device=device, generator=generator) * strength


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "RBG_Smart_Seed_Variance": RBG_Smart_Seed_Variance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RBG_Smart_Seed_Variance": "RBG Smart Seed Variance 🌱",
}
