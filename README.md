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

## Usage

Run the Jupyter notebook:

```bash
jupyter notebook march_madness_prediction.ipynb
```

**Requirements:** A KenPom premium subscription is needed to pull efficiency and strength of schedule data.

## Dependencies

- pandas / numpy
- scikit-learn
- matplotlib
- beautifulsoup4
- requests
- kenpompy
