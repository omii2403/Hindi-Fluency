# Hindi Fluency Project - Run Guide

This README gives only important steps to run the project end-to-end.

## 1. What this project does

This project analyzes Hindi verbal fluency (VFT) and SpAM semantic data.

Main scripts:
- `gen_demographics.py` -> demographic figures from `responses.json`
- `analysis_part1.py` -> core analysis (RQ1, RQ2), embeddings, saves state
- `analysis_part2.py` -> EH1-EH4, SpAM analysis, RQ3/RQ4/RQ5, summary tables

Report files:
- `Report_Final.pdf` -> final PDF

## 2. Required input files

Keep these files in project root:
- `merged_vft_spam_responses.csv`
- `responses.json`

## 3. Environment setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- First run downloads transformer model (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`). Internet is needed.
- Dependencies are listed in `requirements.txt`.

## 4. Run order (important)

Run scripts in this exact order:

```powershell
python .\gen_demographics.py
python .\analysis_part1.py
python .\analysis_part2.py
```

Why this order:
- `analysis_part2.py` needs `analysis_state.pkl` created by `analysis_part1.py`.

## 5. Main outputs you should see

Generated artifacts:
- Figures in `images/` (RQ/EH/SpAM/embedding/demographics)
- `analysis_state.pkl`
- CSV tables like:
  - `table_irt_by_domain.csv`
  - `table_cluster_metrics.csv`
  - `table_hypothesis_summary.csv`
  - `table_spam_rq3_results.csv` (when SpAM data available)
  - `table_spam_rq45_results.csv`
- Final report:
  - `Report_Final.pdf`

## 6. Quick rerun commands

If environment already exists:

```powershell
.\.venv\Scripts\Activate.ps1
python .\analysis_part1.py
python .\analysis_part2.py
```
