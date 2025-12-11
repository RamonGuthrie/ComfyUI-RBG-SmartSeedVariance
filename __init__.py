from .nodes.RBG_Smart_Seed_Variance import RBG_Smart_Seed_Variance

NODE_CLASS_MAPPINGS = {
    "RBG_Smart_Seed_Variance": RBG_Smart_Seed_Variance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RBG_Smart_Seed_Variance": "RBG Smart Seed Variance 🌱",
}

WEB_DIRECTORY = "./web"

print("### Loading: ComfyUI-RBG-SmartSeedVariance ###")

# Add Google Fonts Icons stylesheet
import os
import server

# Get the directory of the current script
dir_path = os.path.dirname(os.path.realpath(__file__))
js_path = os.path.join(dir_path, "web", "js")

# Create the directory if it doesn't exist
os.makedirs(js_path, exist_ok=True)

# Create a new JS file to load the stylesheet
js_file_path = os.path.join(js_path, "RBGSuitePack.js")
with open(js_file_path, "w") as f:
    f.write("""
(function() {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://fonts.googleapis.com/icon?family=Material+Icons';
    document.head.appendChild(link);
})();
""")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]