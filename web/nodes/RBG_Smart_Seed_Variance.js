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

        // --- HELPER: widget visibility ---
        const widgetVisibilityMap = new Map();
        const setWidgetVisible = (widget, visible) => {
          if (!widget) return;
          widget.hidden = !visible;
          widget.computeSize = visible ? undefined : (w) => [w, -4];
          widgetVisibilityMap.set(widget.name, visible);
        };

        const getWidget = (name) => node.widgets.find((w) => w.name === name);

        const updateVisibility = () => {
          const preset = getWidget("variance_preset")?.value;
          const directionShift = getWidget("direction_shift")?.value;
          const schedule = getWidget("variance_schedule")?.value;
          const protectMode = getWidget("protect_mode")?.value;

          setWidgetVisible(getWidget("fine_tune_variance"), preset === "⚙️ Custom");
          setWidgetVisible(getWidget("shift_strength"), directionShift && directionShift !== "🚫 None");
          setWidgetVisible(getWidget("cutoff_step"), schedule && schedule !== "constant");
          setWidgetVisible(getWidget("total_steps"), schedule && schedule !== "constant");
          setWidgetVisible(getWidget("cutoff_strength"), schedule && schedule !== "constant");
          setWidgetVisible(getWidget("protect_regions"), protectMode === "⚙️ Custom Regions");

          const [w] = node.size;
          node.setSize([w, node.computeSize()[1]]);
          node.setDirtyCanvas(true, true);
        };

        // Wrap callbacks to react to visibility changes.
        const wrapCallback = (widgetName) => {
          const widget = getWidget(widgetName);
          if (!widget) return;
          const original = widget.callback;
          widget.callback = (value, ...args) => {
            original?.call(widget, value, ...args);
            updateVisibility();
          };
        };

        wrapCallback("variance_preset");
        wrapCallback("direction_shift");
        wrapCallback("variance_schedule");
        wrapCallback("protect_mode");

        updateVisibility();

        // --- LOGIC: EXPORT ---
        exportBtn.onclick = () => {
          const settings = {};
          for (const widget of node.widgets) {
            if (widget.name && widget.serialize !== false) {
              settings[widget.name] = widget.value;
            }
          }

          const presetName = getWidget("variance_preset")?.value || "custom";
          const saneName = presetName
            .toString()
            .replace(/[^a-zA-Z0-9]+/g, "_")
            .replace(/^_+|_+$/g, "")
            .toLowerCase() || "preset";
          const dateStr = new Date().toISOString().slice(0, 16).replace("T", "_").replace(/:/g, "");

          const data = JSON.stringify(settings, null, 2);
          const blob = new Blob([data], { type: "application/json" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = `rbg_variance_${saneName}_${dateStr}.json`;
          a.click();
          URL.revokeObjectURL(url);
        };

        // --- LOGIC: IMPORT ---
        const fileInput = document.createElement("input");
        fileInput.type = "file";
        fileInput.accept = ".json";
        fileInput.style.display = "none";
        container.appendChild(fileInput);

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
              node.refreshFromWidgets();
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

        // --- TOKEN INSPECTOR VISUALIZATION ---
        const inspectorContainer = document.createElement("div");
        inspectorContainer.style.width = "100%";
        inspectorContainer.style.height = "25px";
        inspectorContainer.style.marginTop = "5px";
        inspectorContainer.style.marginBottom = "5px";
        inspectorContainer.style.background = "#222";
        inspectorContainer.style.borderRadius = "4px";
        inspectorContainer.style.position = "relative";
        inspectorContainer.style.overflow = "hidden";
        inspectorContainer.title = "Token Inspector: Green = Varied, Red = Protected";
        
        const canvas = document.createElement("canvas");
        canvas.width = 300; // Will be resized
        canvas.height = 25;
        canvas.style.width = "100%";
        canvas.style.height = "100%";
        inspectorContainer.appendChild(canvas);
        
        // Tooltip element
        const tooltip = document.createElement("div");
        tooltip.style.position = "absolute";
        tooltip.style.top = "0";
        tooltip.style.left = "0";
        tooltip.style.background = "rgba(0,0,0,0.8)";
        tooltip.style.color = "white";
        tooltip.style.padding = "2px 5px";
        tooltip.style.fontSize = "10px";
        tooltip.style.pointerEvents = "none";
        tooltip.style.display = "none";
        tooltip.style.whiteSpace = "nowrap";
        inspectorContainer.appendChild(tooltip);
        
        container.appendChild(inspectorContainer);
        
        // Data storage
        let protectionMask = [];
        
        const drawInspector = () => {
            const ctx = canvas.getContext("2d");
            const w = canvas.width;
            const h = canvas.height;
            
            ctx.clearRect(0, 0, w, h);
            
            if (!protectionMask || protectionMask.length === 0) {
                // Draw "No Data" placeholder
                ctx.fillStyle = "#333";
                ctx.fillRect(0, 0, w, h);
                ctx.fillStyle = "#666";
                ctx.font = "10px monospace";
                ctx.textAlign = "center";
                ctx.fillText("Run to Inspect Tokens", w/2, h/2 + 3);
                return;
            }
            
            const numTokens = protectionMask.length;
            const tokenWidth = w / numTokens;
            
            for (let i = 0; i < numTokens; i++) {
                const isProtected = protectionMask[i];
                ctx.fillStyle = isProtected ? "#ff4444" : "#44ff44"; // Red for locked, Green for free
                
                // Draw segment with slight gap
                const x = i * tokenWidth;
                const gap = numTokens > 50 ? 0 : 1; // Only show gaps if few tokens
                ctx.fillRect(x, 0, tokenWidth - gap, h);
            }
        };
        
        // Initial draw
        drawInspector();

        // Sync with Widgets (for persistence/imports)
        node.refreshFromWidgets = () => {
            const modeWidget = node.widgets.find(w => w.name === "protect_mode");
            const regionsWidget = node.widgets.find(w => w.name === "protect_regions");
            
            // We need a known token count to draw the bar
            const numTokens = node.properties.lastTokenCount || 0;
            if (numTokens <= 0) return;

            const mode = modeWidget ? modeWidget.value : "🚫 None";
            const regionsStr = regionsWidget ? regionsWidget.value : "";
            
            // Create a temporary mask based on current UI settings
            const newMask = new Array(numTokens).fill(0);
            
            if (mode === "⚙️ Custom Regions") {
                const set = parseRegions(regionsStr);
                set.forEach(idx => {
                    if (idx < numTokens) newMask[idx] = 1;
                });
            } else if (mode === "🎲 Random Regions") {
                const seedWidget = node.widgets.find(w => w.name === "seed");
                const seed = seedWidget ? seedWidget.value : 0;
                // Simple LCG for preview consistency (approximate backend logic)
                let s = seed ^ 0x5EED;
                for (let i = 0; i < numTokens; i++) {
                    // Basic linear congruential generator
                    s = (Math.imul(s, 1664525) + 1013904223) | 0;
                    if (Math.abs(s % 100) < 30) newMask[i] = 1;
                }
            } else if (mode !== "🚫 None") {
                // Approximate legacy modes for UI preview
                // In Python: protect_config = self.PROTECT_OPTIONS.get(mode, (0.0, "start"))
                const opts = {
                    "First Quarter": [0.25, "start"],
                    "First Half": [0.5, "start"],
                    "Last Quarter": [0.25, "end"],
                    "Last Half": [0.5, "end"]
                };
                if (opts[mode]) {
                    const [frac, pos] = opts[mode];
                    const count = Math.floor(numTokens * frac);
                    if (pos === "start") {
                        for (let i = 0; i < count; i++) newMask[i] = 1;
                    } else {
                        for (let i = numTokens - count; i < numTokens; i++) newMask[i] = 1;
                    }
                }
            }
            
            protectionMask = newMask;
            drawInspector();
        };
        
        // Helper: Convert "0-5, 10" string to Set of indices
        const parseRegions = (str) => {
            const set = new Set();
            if (!str) return set;
            str.split(",").forEach(part => {
                part = part.trim();
                if (part.includes("-")) {
                    const [start, end] = part.split("-").map(n => parseInt(n));
                    if (!isNaN(start) && !isNaN(end)) {
                        for (let i = start; i <= end; i++) set.add(i);
                    }
                } else {
                    const n = parseInt(part);
                    if (!isNaN(n)) set.add(n);
                }
            });
            return set;
        };

        // Helper: Convert Set of indices to optimized "0-5, 10" string
        const regionsToString = (set) => {
            const sorted = Array.from(set).sort((a, b) => a - b);
            const ranges = [];
            let start = null, prev = null;
            
            for (const idx of sorted) {
                if (start === null) { start = idx; prev = idx; continue; }
                if (idx === prev + 1) { prev = idx; continue; }
                ranges.push(start === prev ? `${start}` : `${start}-${prev}`);
                start = idx; prev = idx;
            }
            if (start !== null) ranges.push(start === prev ? `${start}` : `${start}-${prev}`);
            return ranges.join(",");
        };

        // Handle Click to Toggle Protection
        canvas.onclick = (e) => {
            if (!protectionMask.length) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const tokenIndex = Math.floor((x / rect.width) * protectionMask.length);
            
            if (tokenIndex >= 0 && tokenIndex < protectionMask.length) {
                // 1. Get Widgets
                const modeWidget = node.widgets.find(w => w.name === "protect_mode");
                const regionsWidget = node.widgets.find(w => w.name === "protect_regions");
                
                if (!regionsWidget) return;

                // 2. Parse Current State
                // If we are NOT in Custom Mode, we should probably start from scratch or the visualized state?
                // Better to start from the *visualized* state (protectionMask) because that matches what the user sees.
                // But wait, protectionMask comes from the last run.
                // If user changes mode to "First Half" but hasn't run, bar is old.
                // Let's assume bar is current.
                // Actually, safer to read the protectionMask itself as the source of truth for the *current* set of protected tokens,
                // then toggle the clicked one.
                
                const currentSet = new Set();
                protectionMask.forEach((isProtected, idx) => {
                    if (isProtected) currentSet.add(idx);
                });
                
                // 3. Toggle
                if (currentSet.has(tokenIndex)) {
                    currentSet.delete(tokenIndex);
                    protectionMask[tokenIndex] = 0; // Optimistic update
                } else {
                    currentSet.add(tokenIndex);
                    protectionMask[tokenIndex] = 1; // Optimistic update
                }
                
                // 4. Update UI
                const newString = regionsToString(currentSet);
                regionsWidget.value = newString;
                
                // Force Mode to Custom
                if (modeWidget && modeWidget.value !== "⚙️ Custom Regions") {
                    modeWidget.value = "⚙️ Custom Regions";
                }
                
                // Trigger updates
                drawInspector(); // Redraw bar immediately
                node.setDirtyCanvas(true, true); // Mark node as needing execution (optional, mostly for style)
            }
        };

        // Handle Mouse Hover for Tooltip
        canvas.onmousemove = (e) => {
            if (!protectionMask.length) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const tokenIndex = Math.floor((x / rect.width) * protectionMask.length);
            
            if (tokenIndex >= 0 && tokenIndex < protectionMask.length) {
                const isProtected = protectionMask[tokenIndex];
                tooltip.style.display = "block";
                tooltip.style.left = `${Math.min(x + 10, rect.width - 80)}px`;
                // Tooltip text
                tooltip.textContent = `Token ${tokenIndex}: ${isProtected ? "Protected 🔒" : "Varied 🎲"} (Click to Toggle)`;
                tooltip.style.color = isProtected ? "#ffaaaa" : "#aaffaa";
                canvas.style.cursor = "pointer";
            }
        };
        
        canvas.onmouseleave = () => {
            tooltip.style.display = "none";
            canvas.style.cursor = "default";
        };

        // --- VIBE_BLEND VISIBILITY ---
        // vibe_blend is only meaningful when target_vibe is connected.
        // We watch the input slot connection state and show/hide accordingly.
        function updateVibeBlendVisibility() {
            const vibeBlendWidget = node.widgets?.find(w => w.name === "vibe_blend");
            if (!vibeBlendWidget) return;

            // Find the target_vibe input slot by name
            const targetVibeInput = node.inputs?.find(inp => inp.name === "target_vibe");
            const isConnected = targetVibeInput?.link != null;

            vibeBlendWidget.hidden = !isConnected;
            if (vibeBlendWidget.element) {
                vibeBlendWidget.element.style.display = isConnected ? "" : "none";
            }
            node.setDirtyCanvas(true, true);
        }

        // Hook into connection changes so visibility updates live
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
            onConnectionsChange?.apply(this, arguments);
            updateVibeBlendVisibility();
        };

        // Handle Execution Data
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            
            if (message && message.protection_data) {
                // protection_data is an array of masks (one per batch item). Take the first one.
                const mask = message.protection_data[0];
                if (mask) {
                   protectionMask = mask;
                   node.properties.lastTokenCount = mask.length;
                   drawInspector();
                }
            }
        };

        // Workflow load persistence
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function() {
            onConfigure?.apply(this, arguments);
            setTimeout(() => {
                this.refreshFromWidgets();
                updateVibeBlendVisibility();
            }, 100);
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function() {
            onRemoved?.apply(this, arguments);
            importBtn.onclick = null;
            fileInput.onchange = null;
            if (fileInput.parentElement) {
                fileInput.parentElement.removeChild(fileInput);
            }
        };

        // Add to node
        const widget = this.addDOMWidget("presets_buttons", "div", container);
        widget.serialize = false;

        // Ensure the node has enough height to contain the buttons + inspector
        const originalComputeSize = this.computeSize;
        this.computeSize = function (width) {
          const size = originalComputeSize ? originalComputeSize.apply(this, arguments) : [width || 300, 200];
          const finalWidth = Math.max(size[0], 300);
          
          // Update canvas resolution on resize for crisp rendering
          if (canvas.width !== finalWidth) {
              canvas.width = finalWidth;
              drawInspector();
          }
          
          size[1] += 80; // height for buttons + inspector (45 + 35)
          return [finalWidth, size[1]];
        };

        // Trigger a resize to apply the new computeSize logic
        // Also run initial vibe_blend visibility check
        setTimeout(() => {
          this.setSize(this.computeSize());
          updateVibeBlendVisibility();
        }, 100);
      };
    }
  },
});