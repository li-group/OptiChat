import json

cfg_template = {
    "model_name": "your_model_name_here",
    "models": {
        "local_resources": [
            "models/*",
            "models/**/*.pkl"
        ]
    },
    "models_code": {
        "local_resources": [
            "models_code/*",
            "models_code/**/*.py"
        ]
    },
    "models_paper": {
        "local_resources": [
            "models_paper/*",
            "models_paper/**/*.txt",
            "models_paper/**/*.pdf"
        ]
    }
}

# json dump example
with open("cfg_template.json", "w") as f:
    json.dump(cfg_template, f, indent=4)
