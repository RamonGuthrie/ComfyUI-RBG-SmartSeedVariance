# ComfyUI-RBG-SmartSeedVariance 🌱

<img src="https://img.shields.io/badge/ComfyUI-Compatible-blue?style=for-the-badge" alt="ComfyUI Compatible"><img src="https://img.shields.io/badge/Nodes-1-green?style=for-the-badge" alt="1 Nodes"><img src="https://img.shields.io/badge/Categories-1-orange?style=for-the-badge" alt="1 Category">

## Advanced Seed Diversity Enhancement for ComfyUI ✨

Some generative models (like Z-Image Turbo and Qwen-Image) suffer from **limited seed variance** — changing the seed produces only subtle variations or nearly identical outputs. The **RBG Smart Seed Variance** node solves this by intelligently injecting controlled noise into text embeddings during generation, creating meaningful diversity while preserving your prompt intention.

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

  - **Z-Image Turbo**, **Qwen-Image**, **Flux (Dev/Schnell)**, **Chroma HD**, **SDXL** and more!

- **11 Direction Shift Patterns:** Apply structured artistic biases instead of pure random noise.
- **7 Spatial Fade Curves:** Control how noise fades across the embedding space.
- **Flexible Noise Injection Timing:** Control when variance is applied.
- **Prompt Token Protection:** Preserve specific parts of your prompt from noise.

---
<img width="3626" height="1511" alt="Screenshot 2025-12-09 143429" src="https://github.com/user-attachments/assets/9d330fa1-b7b5-48c4-91ef-853dcf9a2f06" />

<img width="3612" height="1508" alt="Screenshot 2025-12-09 151729" src="https://github.com/user-attachments/assets/7c2d5b50-2d75-4d79-ab0d-802a62cb4e26" />

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
- **Model matters:** Always select the correct model type for optimal results and avoiding over-correction
- **Combine strategies:** Mix direction shifts with fade curves and noise injection timing for sophisticated effects
- **Use custom mode:** Fine-tune with the slider when presets don't match your needs exactly
- **Chain more then one:** For advanced effects, you can chain multiple Smart Seed Variance nodes.

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

This project is licensed under the MIT License.
