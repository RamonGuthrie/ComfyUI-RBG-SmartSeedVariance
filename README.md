# ComfyUI-RBG-SmartSeedVariance 🌱

<img src="https://img.shields.io/badge/ComfyUI-Compatible-blue?style=for-the-badge" alt="ComfyUI Compatible"><img src="https://img.shields.io/badge/Nodes-1-green?style=for-the-badge" alt="1 Nodes"><img src="https://img.shields.io/badge/Categories-1-orange?style=for-the-badge" alt="1 Category">

## Advanced Seed Diversity Enhancement for ComfyUI ✨

Some generative models (like Krea 2, Z-Image Turbo and Qwen-Image) suffer from **limited seed variance & diversity bias**, changing the seed (especially in single-step or Turbo workflows) often produces nearly identical compositions or subtle variations. Conversely, highly sensitive multi-encoder models (like Flux or Wan 2.2) can easily collapse or lose prompt alignment when raw noise is injected.
The **RBG Smart Seed Variance** node solves this by injecting targeted, mathematically structured noise directly into text embeddings during early generation steps. It creates meaningful diversity, styling variation, and composition shifting while preserving your core prompt intention.

---
<img width="3614" height="1600" alt="Screenshot 2025-12-09 140138" src="https://github.com/user-attachments/assets/272a0bb6-d94e-400b-9be0-40df36dc7941" />

## Feature List 🚀

- **7 Intelligent Presets:** Pre-configured variance levels from Subtle to Wild:

  - **🌱 Subtle** - Gentle diversity for fine-tuning
  - **🌿 Balanced** - Sweet spot for most use cases
  - **🪴 Creative** - Unlock more artistic variations
  - **🌳 Bold** - Significant structural changes
  - **🌴 Wild** - Maximum diversity for exploration (Note this might break your prompt, use with caution!)
  - **⚙️ Custom** - Fine-tune with percentage slider (0-100%)

- **Model-Specific Optimization:** Automatic adjustment for your architecture:

  - **Krea 2**, **Z-Image Turbo**, **Ernie-Image**, **Qwen-Image**, **Flux (Dev/Schnell)**, **Wan**, **Chroma HD**, **SDXL** and more!

- **25+ Direction Shift Patterns:** Apply structured artistic biases instead of pure random noise.
- **7 Spatial Fade Curves:** Control how noise fades across the embedding space.
- **Flexible Noise Injection Timing:** Control when variance is applied.
- **Prompt Token Protection:** Preserve specific parts of your prompt from noise.
---
<img width="3626" height="1511" alt="Screenshot 2025-12-09 143429" src="https://github.com/user-attachments/assets/9d330fa1-b7b5-48c4-91ef-853dcf9a2f06" />

<img width="3612" height="1508" alt="Screenshot 2025-12-09 151729" src="https://github.com/user-attachments/assets/7c2d5b50-2d75-4d79-ab0d-802a62cb4e26" />

---
### 🖼️ Output Comparison Grid

> _Same Prompt & Seed — Comparing Standard KSampler vs. Smart Seed Variance across presets._

|                                               **Standard Seed (No Variance)**                                                |                                                   **Balanced Preset**                                                   |                                                   **Creative Preset**                                                   |                                                   **Bold Preset**                                                   |
| :--------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/user-attachments/assets/00fe795f-f8ec-496e-9a38-51ce132faf1d" width="220" alt="Original Seed"/> | <img src="https://github.com/user-attachments/assets/295c7e3e-e768-48d2-9893-b5e384490dd0" width="220" alt="Balanced"/> | <img src="https://github.com/user-attachments/assets/1226b481-ab7c-4fe2-8829-e0595b1ab71f" width="220" alt="Creative"/> | <img src="https://github.com/user-attachments/assets/a2f3fbe1-91c8-4b8d-ae05-2336b26b598f" width="220" alt="Bold"/> |
|                                                   _Identical pose & frame_                                                   |                                            _Subtle lighting & detail shifts_                                            |                                             _New angle & background shift_                                              |                                           _Complete structural overhaul_                                            |


# 🆕 What's New

- **Ideogram 4.0 support** — a dedicated model preset tuned for Ideogram 4's Qwen3-VL text encoder, paired with a **NEW 🛡️ Latent Noise Guard** direction shift (outlier smoothing + centroid stabilization) to keep noise from tripping the model's built-in safety checks.
- **NEW 🤪 Facial Dynamics & Asymmetry (Krea2)** — a new direction shift that boosts micro facial expressions on facial-descriptive prompts.
- **NEW🗿 Visceral Expression & Grit (Krea2)** rebalanced to behave better when stacked with LoRAs at high weight.
- **⏱️ Variance Schedule (Composition Lock 🔒):** Beyond a flat strength, control how variance evolves across the sampling timeline with 5 modes — `constant`, `decreasing`, `step_cutoff`, `hard_lock`, and `tiered_release` — using `cutoff_step`, `total_steps`, and `cutoff_strength` to map a sampling step to a point on that timeline. Lock composition early while letting fine-detail variance keep going later.
- **Prompt Token Protection:** Preserve specific parts of your prompt from noise, with five modes:
  - `First Quarter` / `First Half` / `Last Quarter` / `Last Half` — quick fractional protection
  - **⚙️ Custom Regions** — type explicit token/range syntax (e.g. `0-5,15-20`)
  - **🎲 Random Regions** — seeds a procedural protection mask for you
- **💾 Export / Import Settings:** Save your full widget configuration to a JSON file and reload it later, or share it with the community.
- **🖱️ Interactive Token Inspector:** A live visualization of the protection mask sits right on the node — click any token to toggle it between protected 🔒 and varied 🎲, hover for a tooltip showing its state, and after a run the bar updates to reflect the actual tokens used (not just a static preview).

https://github.com/user-attachments/assets/84d09d53-187f-4410-a86f-7b1f53da9c89

---

## 📥 Installation

1.  Clone this repository into your `ComfyUI/custom_nodes` directory:
    ```bash
    git clone https://github.com/RamonGuthrie/ComfyUI-RBG-SmartSeedVariance.git
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Restart ComfyUI.

The node will appear under the **RBG Suite/Advanced** category.

---

## 💡 Pro Tips

- **Start conservative:** Begin with "Balanced" preset and adjust up/down based on results
- **Experiment with directions:** Each artistic direction creates unique aesthetic shifts - try them all!
- **Protect your keywords:** Use "First Half" protection if important prompt concepts are being overridden
- **Track your seed:** Use consistent seeds to compare variance effects side-by-side
- **Model matters:** Always select the correct model type for optimal results and to avoid over-correction
- **Combine strategies:** Mix direction shifts with fade curves and noise injection timing for sophisticated effects
- **Use custom mode:** Fine-tune with the slider when presets don't match your needs exactly
- **Chain more than one:** For advanced effects, you can chain multiple Smart Seed Variance nodes.
- **Share settings with the Community:** If you find a killer setting combination Export them and share with the open-source community.

## Watch the Demo video 📺

https://github.com/user-attachments/assets/e33b7139-8979-44f1-ae70-dd46d9b1a91e

---

## 🐛 Troubleshooting

**Output looks exactly the same?**

- Increase preset to "Bold" or "Creative"
- Check that the node is connected to your conditioning
- Verify model type is correct for your actual model
- Try a different seed value

**Quality degraded or image broken?**

- Reduce preset to "Subtle"
- Enable prompt protection ("First Half" or "First Quarter")
- Switch direction shift to "🚫 None" to use pure random
- Try "Ending Steps" to limit variance timing to fine details only

**Getting strange/unexpected outputs?**

- Reduce shift_strength to 50-70%
- Try a different direction shift pattern
- Verify ComfyUI version compatibility

---

## Usage 🚀

To use the `RBG Smart Seed Variance` node, connect it between your KSampler and the Conditioning input. This allows the node to modify the conditioning based on your chosen variance settings.

## Contributing ❤️

Contributions are always welcome! If you have any suggestions, improvements, or new ideas, please feel free to submit a pull request or open an issue.

---

## License 📜

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)** — see the [LICENSE](LICENSE) file for the full text.
 
**What this means in practice:**
- ✅ You can use, study, modify, and redistribute this code — including for commercial ComfyUI workflows.
- ✅ You can fork it and build on it.
- ⚠️ If you distribute a **modified version**, you must release your changes under GPL-3.0 too, and make the source available to whoever you distribute it to.
- ⚠️ You must keep the copyright notice and this license intact, and clearly mark what you changed.
- ❌ You cannot fold this code into a closed-source or proprietary node pack.
This matches the license of ComfyUI itself, so it stays fully compatible with the ecosystem this suite is built for.
 
> Prior to [12/11/2025], this repository was released under the MIT License. Code distributed under the old MIT license terms before that date remains available under MIT for anyone who already forked/cloned it then; all new releases and changes are GPL-3.0.

This project is licensed under the MIT License.
