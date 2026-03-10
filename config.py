import os

# --- Model features ---
FEATURES = [
    'sos',
    'win_percentage',
    'quad_1_win_percentage',
    'quad_2_win_percentage',
    'quad_3_win_percentage',
    'quad_4_win_percentage',
    'is_aq',
    'is_power_conference',
    'quad_1_wins',
    'bad_losses',
]

# Features to convert to per-year percentile ranks.
# Only absolute counts that grow over the season — ratios and ranks
# are already comparable at any point.
PERCENTILE_FEATURES = ['quad_1_wins', 'bad_losses']

# --- Training config ---
TRAINING_YEARS = [2019, 2021, 2022, 2023, 2024, 2025]

# Wikipedia ground truth includes 2018 for extra training data
TOURNAMENT_SEED_YEARS = [2018, 2019, 2021, 2022, 2023, 2024, 2025]

HISTORICAL_QUAD_DATES = {
    2019: '2019-03-17',
    2021: '2021-03-14',
    2022: '2022-03-13',
    2023: '2023-03-11',
    2024: '2024-03-17',
    2025: '2025-03-16',
}

# --- Model parameters ---
MODEL_TYPE = 'random_forest'  # default model type

MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'random_state': 68,
    },
    'logistic_regression': {
        'max_iter': 1000,
        'random_state': 68,
    },
    'gradient_boosting': {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 4,
        'random_state': 68,
    },
}

TEST_SIZE = 0.2
RANDOM_STATE = 68
FIELD_SIZE = 68

# --- Data & model persistence ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DEFAULT_DATASET_PATH = os.path.join(DATA_DIR, 'training_data.csv')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'model.joblib')

def model_path_for(model_type):
    """Return model path with model type in filename, e.g. models/model_logistic_regression.joblib"""
    return os.path.join(MODEL_DIR, f'model_{model_type}.joblib')

# --- Power conferences ---
POWER_CONFERENCES = ['ACC', 'B10', 'B12', 'SEC', 'P12', 'BE']

# --- Conference name mapping (sports-reference full name -> abbreviation) ---
CONFERENCE_NAME_MAPPING = {
    'Big 12 Conference': 'B12',
    'American Athletic Conference': 'Amer',
    'Southeastern Conference': 'SEC',
    'Big East Conference': 'BE',
    'Big Ten Conference': 'B10',
    'Pac-12 Conference': 'P12',
    'Atlantic 10 Conference': 'A10',
    'West Coast Conference': 'WCC',
    'Atlantic Coast Conference': 'ACC',
    'Mid-American Conference': 'MAC',
    'Colonial Athletic Association': 'CAA',
    'Ohio Valley Conference': 'OVC',
    'Conference USA': 'CUSA',
    'Missouri Valley Conference': 'MVC',
    'Ivy League': 'Ivy',
    'Southland Conference': 'Slnd',
    'Big South Conference': 'BSth',
    'Sun Belt Conference': 'SB',
    'Patriot League': 'PL',
    'Mountain West Conference': 'MWC',
    'Atlantic Sun Conference': 'ASun',
    'Western Athletic Conference': 'WAC',
    'Big Sky Conference': 'BSky',
    'Southern Conference': 'SC',
    'Horizon League': 'Horz',
    'Metro Atlantic Athletic Conference': 'MAAC',
    'America East Conference': 'AE',
    'Summit League': 'Sum',
    'Mid-Eastern Athletic Conference': 'MEAC',
    'Big West Conference': 'BW',
    'Northeast Conference': 'NEC',
    'Southwest Athletic Conference': 'SWAC',
    'Coastal Athletic Association': 'CAA',
}

# --- Conference slug mapping (abbreviation -> sports-reference URL slug) ---
CONFERENCE_SLUG_MAPPING = {
    'B12': 'big-12',
    'Amer': 'american',
    'SEC': 'sec',
    'BE': 'big-east',
    'B10': 'big-ten',
    'P12': 'pac-12',
    'A10': 'atlantic-10',
    'WCC': 'wcc',
    'ACC': 'acc',
    'MAC': 'mac',
    'CAA': 'coastal',
    'OVC': 'ovc',
    'CUSA': 'cusa',
    'MVC': 'mvc',
    'Ivy': 'ivy',
    'Slnd': 'southland',
    'BSth': 'big-south',
    'SB': 'sun-belt',
    'PL': 'patriot',
    'MWC': 'mwc',
    'ASun': 'atlantic-sun',
    'WAC': 'wac',
    'BSky': 'big-sky',
    'SC': 'southern',
    'Horz': 'horizon',
    'MAAC': 'maac',
    'AE': 'america-east',
    'Sum': 'summit',
    'MEAC': 'meac',
    'BW': 'big-west',
    'NEC': 'nec',
    'SWAC': 'swac',
}

# --- Unified team name mapping (any variant -> canonical name) ---
# Consolidates all source-specific mappings from the notebook.
# Applied AFTER the 'State' -> 'St.' replacement.
TEAM_NAME_MAPPING = {
    # bracketologists.com variants
    'CSUN': 'Cal St. Northridge',
    'California Baptist': 'Cal Baptist',
    'Central Connecticut St.': 'Central Connecticut',
    'East Texas A&M': 'Texas A&M Commerce',
    'Florida International': 'FIU',
    "Hawai'i": 'Hawaii',
    'IU Indy': 'IUPUI',
    'Kansas City': 'UMKC',
    'Loyola (MD)': 'Loyola MD',
    'MD Eastern Shore': 'Maryland Eastern Shore',
    'Miami (FL)': 'Miami FL',
    'Miami (OH)': 'Miami OH',
    'NC St.': 'N.C. State',
    'Ole Miss': 'Mississippi',
    'Omaha': 'Nebraska Omaha',
    'Pennsylvania': 'Penn',
    'SE Missouri St.': 'Southeast Missouri St.',
    'Saint Francis (BKN)': 'Saint Francis',
    'Sam Houston': 'Sam Houston St.',
    'South Carolina Upstate': 'USC Upstate',
    "St. Peter's": "Saint Peter's",
    'Texas A&M CC': 'Texas A&M Corpus Chris',
    'UAlbany': 'Albany',
    'UConn': 'Connecticut',
    'UIC': 'Illinois Chicago',
    'UT Martin': 'Tennessee Martin',
    'McNeese St.': 'McNeese',
    'SIU Edwardsville': 'SIUE',

    # Torvik variants (State -> St. replacement over-normalizes these)
    'N.C. St.': 'N.C. State',

    # sports-reference.com variants (conference champions & standings)
    'Albany (NY)': 'Albany',
    "St. John's (NY)": "St. John's",
    'Southern Illinois-Edwardsville': 'SIUE',
    'College of Charleston': 'Charleston',
    'FDU': 'Fairleigh Dickinson',
    'Long Island University': 'LIU',
    'Maryland Baltimore County': 'UMBC',
    'Massachusetts Lowell': 'UMass Lowell',
    'Saint Francis (PA)': 'Saint Francis',
    'Gardner-Webb': 'Gardner Webb',
    'Grambling': 'Grambling St.',
    'Loyola (IL)': 'Loyola Chicago',
    'Prairie View': 'Prairie View A&M',
    "Saint Mary's (CA)": "Saint Mary's",
    'Texas A&M-Corpus Christi': 'Texas A&M Corpus Chris',
    'Virginia Commonwealth': 'VCU',

    # Wikipedia variants (en-dash characters)
    'Gardner\u2013Webb': 'Gardner Webb',
    'Loyola\u2013Chicago': 'Loyola Chicago',
    'North Carolina St.': 'N.C. State',
    'Texas A&M\u2013Corpus Christi': 'Texas A&M Corpus Chris',
}

# --- Manual conference champion overrides (ineligible teams) ---
CONFERENCE_CHAMP_OVERRIDES = {
    (2022, 'ASun'): 'Jacksonville St.',
    (2023, 'NEC'): 'Fairleigh Dickinson',
}
