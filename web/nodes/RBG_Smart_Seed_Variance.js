import { app } from "/scripts/app.js";

app.registerExtension({
  name: "RBG.SmartSeedVariance.Presets",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "RBG_Smart_Seed_Variance") {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply(this, arguments);
        const node = this;

        // --- CONTAINER FOR BUTTONS ---
        const container = document.createElement("div");
        container.style.display = "flex";
        container.style.flexDirection = "column";
        container.style.gap = "5px";
        container.style.width = "100%";
        container.style.padding = "5px 0";

        // --- ROW 1: IMPORT / EXPORT ---
        const row1 = document.createElement("div");
        row1.style.display = "flex";
        row1.style.gap = "5px";
        row1.style.width = "100%";

                const btnStyle = `
                    padding: 8px;
                    cursor: pointer;
                    background: rgba(255, 255, 255, 0.05);
                    color: white;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 4px;
                    font-size: 14px;
                    text-align: center;
                    transition: all 0.2s ease;
                    user-select: none;
                    box-sizing: border-box;
                `;

                const addEffects = (btn) => {
                    btn.onmouseover = () => btn.style.backgroundColor = "rgba(255, 255, 255, 0.15)";
                    btn.onmouseout = () => btn.style.backgroundColor = "rgba(255, 255, 255, 0.05)";
                    btn.onmousedown = () => btn.style.opacity = "0.5";
                    btn.onmouseup = () => btn.style.opacity = "1";
                    btn.onmouseleave = () => {
                        btn.style.backgroundColor = "rgba(255, 255, 255, 0.05)";
                        btn.style.opacity = "1";
                    };
                };

        const importBtn = document.createElement("button");
        importBtn.innerHTML = "Import 🫘";
        importBtn.style.cssText = btnStyle + "flex: 1;";
        addEffects(importBtn);

        const exportBtn = document.createElement("button");
        exportBtn.innerHTML = "Export 🌼";
        exportBtn.style.cssText = btnStyle + "flex: 1;";
        addEffects(exportBtn);

        row1.appendChild(importBtn);
        row1.appendChild(exportBtn);

        // --- ROW 2: SYNC & BLOOM ---
        const syncBtn = document.createElement("button");
        syncBtn.innerHTML = "Sync & Bloom 🔄";
        syncBtn.style.cssText = btnStyle + "width: 100%; flex: 0 0 auto;";
        addEffects(syncBtn);

        container.appendChild(row1);
        container.appendChild(syncBtn);

        // --- LOGIC: EXPORT ---
        exportBtn.onclick = () => {
          const settings = {};
          for (const widget of node.widgets) {
            if (widget.name && widget.serialize !== false) {
              settings[widget.name] = widget.value;
            }
          }

          const data = JSON.stringify(settings, null, 2);
          const blob = new Blob([data], { type: "application/json" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = `seed_variance_preset_${new Date().getTime()}.json`;
          a.click();
          URL.revokeObjectURL(url);
        };

        // --- LOGIC: IMPORT ---
        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = ".json";
        fileInput.style.display = "none";
        document.body.appendChild(fileInput);

        importBtn.onclick = () => fileInput.click();

        fileInput.onchange = (e) => {
          const file = e.target.files[0];
          if (!file) return;

          const reader = new FileReader();
          reader.onload = (event) => {
            try {
              const settings = JSON.parse(event.target.result);
              for (const widgetName in settings) {
                const widget = node.widgets.find((w) => w.name === widgetName);
                if (widget) {
                  widget.value = settings[widgetName];
                  if (widget.callback) {
                    widget.callback(widget.value);
                  }
                }
              }
              node.setDirtyCanvas(true, true);
            } catch (err) {
              console.error("[RBG Smart Seed Variance] Import failed:", err);
            }
          };
          reader.readAsText(file);
          // Reset input so same file can be imported again if needed
          fileInput.value = "";
        };

        // --- LOGIC: SYNC & BLOOM (Local Curated Presets) ---
        const curatedPresets = [
          {
            variance_preset: "🌿 Balanced",
            fine_tune_variance: 50,
            fade_curve: "Smooth Step",
            noise_injection: "Beginning Steps",
            direction_shift: "🌎 Diversity",
            shift_strength: 120,
            variance_schedule: "decreasing",
            cutoff_step: 4,
            cutoff_strength: 0.15,
          },
          {
            variance_preset: "🌳 Bold",
            fine_tune_variance: 85,
            fade_curve: "Instant",
            noise_injection: "All Steps",
            direction_shift: "🧬 Face-Variance Expansion",
            shift_strength: 150,
            variance_schedule: "constant",
            cutoff_step: 8,
            cutoff_strength: 0.2,
          },
          {
            variance_preset: "🌱 Subtle",
            fine_tune_variance: 25,
            fade_curve: "Ease-Out",
            noise_injection: "Ending Steps",
            direction_shift: "🧭 Semantic Drift (Centroid-Safe)",
            shift_strength: 80,
            variance_schedule: "decreasing",
            cutoff_step: 2,
            cutoff_strength: 0.05,
          },
        ];

        syncBtn.onclick = () => {
          const preset =
            curatedPresets[Math.floor(Math.random() * curatedPresets.length)];
          for (const key in preset) {
            const widget = node.widgets.find((w) => w.name === key);
            if (widget) {
              widget.value = preset[key];
              if (widget.callback) {
                widget.callback(widget.value);
              }
            }
          }
          node.setDirtyCanvas(true, true);

          // Visual feedback
          const originalText = syncBtn.innerHTML;
          syncBtn.innerHTML = "Bloomed! ✨";
          setTimeout(() => {
            syncBtn.innerHTML = originalText;
          }, 1000);
        };

        // Add to node
        const widget = this.addDOMWidget("presets_buttons", "div", container);
        widget.serialize = false;

        // Ensure the node has enough height to contain the buttons
        const originalComputeSize = this.computeSize;
        this.computeSize = function (width) {
          const size = originalComputeSize ? originalComputeSize.apply(this, arguments) : [width || 300, 200];
          // Use a comfortable default width of 300px if it's smaller
          const finalWidth = Math.max(size[0], 300);
          // Tightened height: Using the user's preferred 50px offset
          size[1] += 45; 
          return [finalWidth, size[1]];
        };

        // Trigger a resize to apply the new computeSize logic
        setTimeout(() => {
          this.setSize(this.computeSize());
        }, 100);
      };
    }
  },
});
