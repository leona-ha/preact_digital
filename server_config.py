# configuration file for the TIKI project
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    PROJECT_ROOT = Path(__file__).resolve().parent
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    print("Couldn't load .env file, continuing with defaults")

# Fallback values are set to the default cluster paths
base_path = Path(os.getenv("TIKI_BASE_PATH", "/sc-projects/sc-proj-cc15-preact/SP6"))

# Construct subpaths relative to the base directory
# (Keep as string for compatibility with legacy notebooks)
datapath = str(base_path) + "/"
raw_path = str(base_path / "raw")
preprocessed_path = str(base_path / "preprocessed")
preprocessed_path_jitai = str(base_path / "preprocessed" / "jitai_leona")
backup_path = str(base_path / "backup")
redcap_path = str(base_path / "redcap")
sp1_path = os.getenv("TIKI_SP1_PATH", str(base_path.parent / "SP1"))

proj_sheet = os.getenv("TIKI_PROJ_SHEET", "1z8LZJBBMzzAmiXIS47X8SLk-zSMwDIXSKPit4IlmfuE")
credential_path = os.getenv("TIKI_CREDENTIAL_PATH", "/home/leha18/tiki_code/notebooks/tiki-399609-b68b5ade4dfb.json")

drivesheet_url = f"https://docs.google.com/spreadsheets/d/{proj_sheet}"
