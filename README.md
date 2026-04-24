# Hindi Fluency Analysis (VFT + SpAM)

This repository contains the full project material for the Hindi fluency study. The main analysis uses the Verbal Fluency Task (VFT), the Spatial Arrangement Method (SpAM), and semantic clustering to study how Hindi/Hinglish responses are organized.

## Main files

- `Hindi_fluency_final.ipynb`: the executed notebook with the full analysis
- `Report_Final.tex`: the IEEE-style report draft
- `BRSM-Syllabus.md`: course summary used to shape the report structure
- `merged_vft_spam_responses_enriched.csv`: processed dataset used by the notebook
- `responses.json`: raw response log
- `images/img/`: generated figures used in the report

## What the notebook does

- Descriptive statistics for IRT and domain-level patterns
- Cluster-based within-vs-between retrieval analysis
- Mixed-effects models for serial position effects
- Kruskal-Wallis and Mann-Whitney tests for domain comparisons
- Semantic, phonological, and SpAM alignment checks
- Composite fluency score comparison

## Setup

### Prerequisites

- Python 3.10 or newer
- `pip`
- MiKTeX or TeX Live if you want to compile the LaTeX report

### Create and activate the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install Python dependencies

```powershell
pip install -r requirements.txt
```

## Run the notebook

Open `Hindi_fluency_final.ipynb` and run the cells from top to bottom. The notebook reads `merged_vft_spam_responses_enriched.csv` and regenerates the figures in `images/img/`.

## Compile the report

Run the LaTeX file twice so the references settle:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error Report_Final.tex
pdflatex -interaction=nonstopmode -halt-on-error Report_Final.tex
```

This creates `Report_Final.pdf`.

## Notes

- The report uses placeholder citations in `Report_Final.tex`. Replace them with the final bibliography when you are ready.
- The notebook analysis is already executed, so the figures in `images/img/` can be reused directly.
- If you add a poster later, the same figures and result summary can be reused in the report.
