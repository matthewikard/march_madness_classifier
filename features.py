import pandas as pd

from config import POWER_CONFERENCES, PERCENTILE_FEATURES


def build_dataset(torvik_df, quad_df, champs_df, seeds_df=None):
    """
    Merge all data sources and engineer features for the model.

    Args:
        torvik_df: from scrape_torvik()
        quad_df: from scrape_quad_records()
        champs_df: from scrape_conference_champions()
        seeds_df: from scrape_tournament_seeds() — None for prediction mode

    Returns:
        DataFrame with all features and (if seeds_df provided) the
        made_tournament target column.
    """
    # merge torvik with quad records
    merged = pd.merge(torvik_df, quad_df, on=['team', 'year'], how='left')

    # merge with conference champions
    merged = pd.merge(
        merged, champs_df,
        left_on=['team', 'year'],
        right_on=['postseason_champion', 'year'],
        how='left',
    )

    # merge with tournament seeds if available (training mode)
    if seeds_df is not None:
        merged = pd.merge(merged, seeds_df, on=['team', 'year'], how='left')

    # select and rename columns
    # after merges, conference appears as conference_x (from torvik) and
    # conference_y (from champs). record appears as record_x and record_y.
    cols = [
        'team', 'year', 'conference_x', 'record_y',
        'sos',
        'wins', 'losses', 'win_percentage',
        'quad_1_record', 'quad_2_record', 'quad_3_record', 'quad_4_record',
        'non_d1_record',
        'quad_1_win_percentage', 'quad_2_win_percentage',
        'quad_3_win_percentage', 'quad_4_win_percentage',
        'postseason_champion',
    ]

    if seeds_df is not None:
        cols.append('seed')

    merged = merged[cols]

    rename = {
        'conference_x': 'conference',
        'record_y': 'record',
    }

    merged = merged.rename(columns=rename)

    # --- Check for unmatched conference leaders ---
    if not champs_df.empty:
        all_teams = set(merged['team'].unique())
        unmatched = champs_df[~champs_df['postseason_champion'].isin(all_teams)]
        if not unmatched.empty:
            print('  WARNING: conference leaders not matching any Torvik team (update TEAM_NAME_MAPPING):')
            for _, row in unmatched.iterrows():
                print(f'    {row["conference"]}: {repr(row["postseason_champion"])}')

    # --- Feature engineering ---

    # is automatic qualifier
    merged['is_aq'] = merged['postseason_champion'].notna()

    # made tournament (target variable, training only)
    if seeds_df is not None:
        merged['made_tournament'] = merged['seed'].notna()

    # is power conference
    merged['is_power_conference'] = merged['conference'].isin(POWER_CONFERENCES)

    # filter out teams with no quad records (didn't play)
    merged = merged[merged['quad_1_record'].notnull()]

    # total quad 1 wins
    merged['quad_1_wins'] = merged['quad_1_record'].str.split('-', expand=True).astype(int)[0]

    # bad losses (quad 3 + quad 4 losses)
    merged['bad_losses'] = (
        merged['quad_3_record'].str.split('-', expand=True).astype(int)[1]
        + merged['quad_4_record'].str.split('-', expand=True).astype(int)[1]
    )

    # convert count features to per-year percentile ranks so mid-season
    # values are comparable to end-of-season training data
    merged = _apply_percentile_ranking(merged, PERCENTILE_FEATURES)

    return merged


def _apply_percentile_ranking(df, features):
    """
    Convert specified features to per-year percentile ranks (0–1).

    Ties receive the average of their rank positions.
    """
    df = df.copy()
    for col in features:
        df[col] = df.groupby('year')[col].rank(pct=True, method='average')
    return df
