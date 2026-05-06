# Hindi Fluency Analysis (VFT + SpAM)

This repository contains the full project material for the Hindi fluency study. The main analysis uses the Verbal Fluency Task (VFT), the Spatial Arrangement Method (SpAM), and semantic clustering to study how Hindi/Hinglish responses are organized.

## Main files

- `Hindi_fluency_final.ipynb`: the executed notebook with the full analysis
- `merged_vft_spam_responses_enriched.csv`: processed dataset used by the notebook
- `responses.json`: raw response log

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

Open `Hindi_fluency_final.ipynb` and run the cells from top to bottom. The notebook reads `merged_vft_spam_responses_enriched.csv` and regenerates the figures in `images/`.
