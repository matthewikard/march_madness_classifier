# March Madness Qualifier Prediction

A Random Forest classifier that predicts which teams will make the NCAA March Madness tournament based on resume metrics used by the selection committee.

## Overview

The selection committee relies on quantifiable metrics — quad records, strength of schedule, and win/loss data — to choose the 68-team field. This project scrapes historical data from multiple sources, engineers features that mirror the committee's evaluation criteria, and trains a model to predict tournament qualifiers.

Classifier correctly projected 67/68 tournament teams in 2025. More in my substack [here](https://ikard.substack.com/p/reflection-march-madness-prediction)

## Data Sources

- **[KenPom](https://kenpom.com/)** — Strength of schedule rankings (requires paid subscription, accessed via [kenpompy](https://kenpompy.readthedocs.io/en/latest/))
- **[Sports Reference](https://www.sports-reference.com/)** — Conference tournament champions (automatic qualifiers)
- **[Bracketologists.com](https://bracketologists.com/)** — Quad 1–4 records
- **[Wikipedia](https://en.wikipedia.org/)** — Historical tournament fields (ground truth labels)

Training data spans 2019–2024 (excluding 2020), aligning with the committee's adoption of NET rankings and the Quad system.

## Features

| Feature | Description |
|---|---|
| `sos_adj_em_rank` | KenPom strength of schedule ranking |
| `win_percentage` | Overall win percentage |
| `quad_1–4_win_percentage` | Win percentage against each quad |
| `is_aq` | Whether the team won its conference tournament |
| `is_power_conference` | ACC, B10, B12, SEC, P12, or BE membership |
| `quad_1_wins` | Total Quad 1 wins |
| `bad_losses` | Combined Quad 3 + Quad 4 losses |

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your KenPom credentials (see `.env.example`):
```
KENPOM_USER=your_email@example.com
KENPOM_PASSWORD=your_password
```

A [KenPom](https://kenpom.com/) premium subscription is required.

## Usage

### 1. Scrape training data
Scrapes historical data (2019–2024) from all sources and saves it to `data/training_data.csv`. Requires KenPom login:
```bash
python cli.py scrape
```

You only need to re-run this when new seasons finish or you want to refresh the training data.

### 2. Train the model
Trains the classifier from saved data and saves it to `models/model.joblib`. No KenPom login needed:
```bash
python cli.py train
```

Use `--no-plots` to skip the confusion matrix and feature importance charts:
```bash
python cli.py train --no-plots
```

To scrape fresh data and train in one step:
```bash
python cli.py train --scrape
```

### 3. Predict a tournament field
Scrapes current-season data and predicts the 68-team field:
```bash
python cli.py predict 2026
```

Optionally specify a date for quad records (defaults to today):
```bash
python cli.py predict 2026 --date 2026-03-15
```

**Note:** If conference tournaments haven't been played yet, the model uses regular season conference leaders as the automatic qualifier proxy.

## Project Structure

```
config.py       — Constants, feature list, team/conference name mappings
scrapers.py     — Data scraping (KenPom, Sports Reference, Bracketologists, Wikipedia)
features.py     — Data merging and feature engineering
model.py        — Model training, persistence (joblib), and prediction
cli.py          — Command-line interface
data/           — Saved training data
models/         — Saved model files
```

## Dependencies

- pandas / numpy
- scikit-learn
- matplotlib
- beautifulsoup4
- requests
- kenpompy
- python-dotenv
- joblib
