# configuration file for the TIKI project
import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve the root directory (where server_config.py is located)
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# Fallback values are set to the default cluster paths
base_path = Path(os.getenv("TIKI_BASE_PATH", "/sc-projects/sc-proj-cc15-preact/SP6"))

# Construct subpaths relative to the base directory
# (Keep as string with trailing slash for compatibility with legacy notebooks)
datapath = str(base_path) + "/"
raw_path = str(base_path / "raw")
preprocessed_path = str(base_path / "preprocessed")
preprocessed_path_jitai = str(base_path / "preprocessed" / "jitai_leona")
backup_path = str(base_path / "backup")
redcap_path = str(base_path / "redcap")

proj_sheet = os.getenv("TIKI_PROJ_SHEET")
credential_path = os.getenv("TIKI_CREDENTIAL_PATH", "/home/leha18/tiki_code/notebooks/tiki-399609-b68b5ade4dfb.json")

drivesheet_url = f"https://docs.google.com/spreadsheets/d/{proj_sheet}"