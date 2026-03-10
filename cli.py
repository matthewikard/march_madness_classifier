"""
CLI for March Madness tournament field prediction.

Usage:
    python cli.py scrape                  # scrape & save training data
    python cli.py train                   # train model from saved data
    python cli.py train --scrape          # scrape fresh data then train
    python cli.py predict 2026            # predict using today's quad data
    python cli.py predict 2026 --date 2026-02-22
"""
import argparse
import os
from datetime import date

import pandas as pd

from config import (
    TRAINING_YEARS, TOURNAMENT_SEED_YEARS, HISTORICAL_QUAD_DATES,
    DEFAULT_MODEL_PATH, DEFAULT_DATASET_PATH, MODEL_PARAMS,
    model_path_for,
)


def _scrape_training_data():
    """Scrape all historical data sources, build and save the training dataset."""
    from scrapers import scrape_torvik, scrape_conference_champions, scrape_quad_records, scrape_tournament_seeds
    from features import build_dataset

    print('Scraping Torvik ratings...')
    torvik_df = scrape_torvik(TRAINING_YEARS)

    print('Scraping conference champions...')
    champs_df = scrape_conference_champions(TRAINING_YEARS)

    quad_dates = [HISTORICAL_QUAD_DATES[y] for y in TRAINING_YEARS]
    print('Scraping quad records...')
    quad_df = scrape_quad_records(quad_dates)

    print('Scraping tournament seeds from Wikipedia...')
    seeds_df = scrape_tournament_seeds(TOURNAMENT_SEED_YEARS)

    print('Building dataset...')
    dataset = build_dataset(torvik_df, quad_df, champs_df, seeds_df)

    os.makedirs(os.path.dirname(DEFAULT_DATASET_PATH), exist_ok=True)
    dataset.to_csv(DEFAULT_DATASET_PATH, index=False)
    print(f'Training data saved to {DEFAULT_DATASET_PATH} ({len(dataset)} teams)')

    return dataset


def cmd_scrape(args):
    """Scrape historical data and save to disk."""
    _scrape_training_data()


def cmd_train(args):
    """Train model from saved (or freshly scraped) training data."""
    from model import train

    if args.scrape:
        dataset = _scrape_training_data()
    else:
        if not os.path.exists(DEFAULT_DATASET_PATH):
            print(f'No saved training data found at {DEFAULT_DATASET_PATH}.')
            print('Run "python cli.py scrape" first, or use "python cli.py train --scrape".')
            return
        print(f'Loading saved training data from {DEFAULT_DATASET_PATH}...')
        dataset = pd.read_csv(DEFAULT_DATASET_PATH)

    model_path = args.model_path if args.model_path != DEFAULT_MODEL_PATH else model_path_for(args.model_type)
    print(f'Training model ({len(dataset)} teams across {len(TRAINING_YEARS)} seasons)...')
    train(dataset, model_path=model_path, show_plots=not args.no_plots, model_type=args.model_type)


def cmd_predict(args):
    """Load saved model, scrape current year data, predict tournament field."""
    from scrapers import scrape_torvik, scrape_conference_champions, scrape_quad_records
    from features import build_dataset
    from model import load_model, predict

    year = args.year
    quad_date = args.date or date.today().isoformat()

    print(f'Predicting {year} tournament field using quad data from {quad_date}...')

    print('Scraping Torvik ratings...')
    torvik_df = scrape_torvik([year])

    print('Scraping conference champions...')
    champs_df = scrape_conference_champions([year])

    print('Scraping quad records...')
    quad_df = scrape_quad_records([quad_date])

    print('Building dataset...')
    dataset = build_dataset(torvik_df, quad_df, champs_df, seeds_df=None)

    print(f'Loading model ({args.model_type})...')
    model_path = args.model_path if args.model_path != DEFAULT_MODEL_PATH else model_path_for(args.model_type)
    clf = load_model(model_path=model_path)

    print('Predicting...\n')
    results = predict(clf, dataset)

    # display predicted field
    field = results[results['predicted_tournament']].sort_values('tournament_prob', ascending=False)
    print(f'Predicted {year} Tournament Field ({len(field)} teams):')
    print(field[['team', 'conference', 'record', 'tournament_prob']].to_string(index=False))

    # display bubble teams (next 10 teams outside the field)
    bubble = results[~results['predicted_tournament']].sort_values('tournament_prob', ascending=False).head(10)
    print(f'\nFirst 10 Out:')
    print(bubble[['team', 'conference', 'record', 'tournament_prob']].to_string(index=False))

    # save predictions to CSV
    from config import DATA_DIR
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f'predictions_{year}_{quad_date}.csv')
    results.sort_values('tournament_prob', ascending=False).to_csv(out_path, index=False)
    print(f'\nPredictions saved to {out_path}')


def main():
    parser = argparse.ArgumentParser(description='March Madness Tournament Field Predictor')
    parser.add_argument(
        '--model-path', default=DEFAULT_MODEL_PATH,
        help=f'Path to model file (default: {DEFAULT_MODEL_PATH})'
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # scrape command
    subparsers.add_parser('scrape', help='Scrape historical data and save to disk')

    # train command
    train_parser = subparsers.add_parser('train', help='Train model from saved training data')
    train_parser.add_argument(
        '--scrape', action='store_true',
        help='Scrape fresh data before training'
    )
    train_parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip displaying confusion matrix and feature importance plots'
    )
    train_parser.add_argument(
        '--model-type', default='random_forest',
        choices=list(MODEL_PARAMS.keys()),
        help='Model type to train (default: random_forest)'
    )

    # predict command
    predict_parser = subparsers.add_parser('predict', help='Predict tournament field for a year')
    predict_parser.add_argument('year', type=int, help='Year to predict (e.g., 2026)')
    predict_parser.add_argument(
        '--date', type=str, default=None,
        help='Quad records date (YYYY-MM-DD). Defaults to today.'
    )
    predict_parser.add_argument(
        '--model-type', default='random_forest',
        choices=list(MODEL_PARAMS.keys()),
        help='Model type to use for prediction (default: random_forest)'
    )

    args = parser.parse_args()

    if args.command == 'scrape':
        cmd_scrape(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'predict':
        cmd_predict(args)


if __name__ == '__main__':
    main()
