# PREACT-digital

This repository contains the codebase for the **PREACT-digital** project.

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Environment Setup & System Requirements](#environment-setup--system-requirements)
- [Dataset](#dataset)

## Repository Structure

The project codebase is organized as follows:

```
├── assets/                  # Static assets and images
├── notebooks/               # Jupyter notebooks for interactive analysis and exploration
│   ├── 03_Meta_Data.ipynb
│   ├── 04_Features_Crosssectional.ipynb
│   ├── 07_EMA_Network.ipynb
│   ├── 08_Mplus_timegrid.ipynb
│   ├── X2_Sample_Overview.ipynb
│   └── old_notebooks/       # Deprecated notebooks
├── quarto/                  # Quarto documentation
├── src/                     # Core source code of the project
│   ├── model_pipelines/     # Machine learning training and evaluation pipelines
│   │   ├── ML_config.py
│   │   ├── ML_pipeline.py
│   │   ├── custom_models.py
│   │   └── run_ML_pipeline.py
│   ├── preprocessing/       # Feature extraction and cleaning functions for sensors/EMA
│   │   ├── aggregation.py   # Daily passive feature aggregations (HR, sleep, steps, activity)
│   │   ├── ema_features.py
│   │   ├── ema_mappings.py
│   │   ├── gps_features.py
│   │   ├── infer_timeoffset.py
│   │   ├── missing_data.py
│   │   └── redcap_features.py
│   └── scripts/             # Python executable scripts
│       ├── 00_temp_gps_minivalid.py
│       ├── 01_Data_Preprocess.py
│       ├── 02_Aggregate_All_Passive.py
│       ├── 02gps_Aggregate_GPS.py
│       ├── 06_ECG_Preprocess.py
│       └── create_backup.py
└── server_config.py         # Root server configuration (paths, partitions, environment variables)
```

## Environment Setup & System Requirements

### Environment Variables (.env)

The project loads custom path configurations and API keys from a `.env` file located at the repository root. You can set this up by copying the template file `.env.example`:

```bash
cp .env.example .env
```

And then adjusting the variables inside `.env`:

```env
# The base directory path containing the datasets (raw/, preprocessed/, backup/, redcap/)
# Fallback default: /sc-projects/sc-proj-cc15-preact/SP6
TIKI_BASE_PATH="/sc-projects/sc-proj-cc15-preact/SP6"

# Google Sheets ID for the project metadata sheet
TIKI_PROJ_SHEET="your_google_sheets_id_here"

# TODO credentials
# Fallback default: /home/leha18/tiki_code/notebooks/tiki-399609-b68b5ade4dfb.json
TIKI_CREDENTIAL_PATH="/path/to/your/google-credentials.json"
```

## Dataset


