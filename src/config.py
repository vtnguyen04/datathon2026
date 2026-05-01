from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SUBMISSION_DIR = PROJECT_ROOT / "data" / "submissions"
LOGS_DIR = PROJECT_ROOT / "logs"
SEED = 42
OPENROUTER_API_KEY = (
    "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"
)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openai/gpt-4o"
PALETTE = {
    "primary": "#2EC4B6",
    "secondary": "#E71D36",
    "accent": "#FF9F1C",
    "dark": "#011627",
    "light": "#FDFFFC",
    "cyan": "#41EAD4",
    "purple": "#7B2FBE",
    "grey": "#8D99AE",
}
PALETTE_LIST = list(PALETTE.values())
