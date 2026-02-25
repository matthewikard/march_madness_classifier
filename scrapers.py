import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from kenpompy.utils import login
from kenpompy.misc import get_pomeroy_ratings

from config import CONFERENCE_NAME_MAPPING, TEAM_NAME_MAPPING, CONFERENCE_CHAMP_OVERRIDES

_kenpom_browser = None


def _get_kenpom_browser():
    """Return a cached authenticated KenPom session, logging in once."""
    global _kenpom_browser
    if _kenpom_browser is None:
        load_dotenv()
        username = os.environ['KENPOM_USER']
        password = os.environ['KENPOM_PASSWORD']
        _kenpom_browser = login(username, password)
    return _kenpom_browser


def _normalize_team_names(series):
    """Apply standard team name normalization: State -> St., then unified mapping."""
    series = series.str.replace('State', 'St.', regex=False)
    series = series.map(TEAM_NAME_MAPPING).fillna(series)
    return series


def scrape_kenpom(years):
    """
    Scrape KenPom ratings for the given years.
    Credentials loaded from KENPOM_USER and KENPOM_PASSWORD environment variables.

    Returns DataFrame with columns:
        team, year, seed, conference, record, sos_adj_em_rank, adj_em,
        o_adj_rank, d_adj_rank, adj_em_rank
    """
    browser = _get_kenpom_browser()

    frames = []
    for year in years:
        ratings = get_pomeroy_ratings(browser, year)
        ratings['year'] = year
        frames.append(ratings)

    df = pd.concat(frames, ignore_index=True)

    # select columns of interest
    df = df[['Team', 'year', 'Seed', 'Conf', 'W-L', 'SOS-AdjEM.Rank', 'AdjEM', 'AdjO.Rank', 'AdjD.Rank']]

    # rename columns
    df.columns = ['team', 'year', 'seed', 'conference', 'record',
                  'sos_adj_em_rank', 'adj_em', 'o_adj_rank', 'd_adj_rank']

    # filter out teams who didn't play (2021 issue)
    df[['wins', 'losses']] = df['record'].str.split('-', expand=True).astype(int)
    df = df[df['wins'] + df['losses'] > 0]
    df = df.drop(['wins', 'losses'], axis=1)

    # remove '+' from adjusted efficiency, convert to float
    df['adj_em'] = df['adj_em'].str.replace('+', '', regex=False).astype('double')

    # create rank column for efficiency
    df['adj_em_rank'] = df.groupby('year')['adj_em'].rank(ascending=False, method='min').astype(int)

    return df


def scrape_conference_champions(years):
    """
    Scrape conference tournament champions from sports-reference.com.

    Uses tournament champions (conf_champ_post). If no tournament champion
    data exists at all for a year (e.g. mid-season), falls back to regular
    season champions (conf_champ_reg) for that entire year.

    Returns DataFrame with columns: year, conference, postseason_champion
    """
    all_data = []

    for year in years:
        url = f"https://www.sports-reference.com/cbb/seasons/men/{year}.html"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        conf_names = [tag.text.strip() for tag in soup.find_all("td", {"data-stat": "conf_name"})]
        tourney_champs = [tag.text.strip() for tag in soup.find_all("td", {"data-stat": "conf_champ_post"})]

        # use tournament champs if any are populated, otherwise fall back
        # to regular season champs (mid-season prediction case)
        has_tourney_data = any(c for c in tourney_champs)
        if has_tourney_data:
            for conf, champ in zip(conf_names, tourney_champs):
                all_data.append((year, conf, champ))
        else:
            reg_season_champs = [tag.text.strip() for tag in soup.find_all("td", {"data-stat": "conf_champ_reg"})]
            for conf, champ in zip(conf_names, reg_season_champs):
                all_data.append((year, conf, champ))

    df = pd.DataFrame(all_data, columns=['year', 'conference', 'postseason_champion'])

    # remove independent conferences
    df = df[df['conference'] != 'Independent']

    # map conference names to KenPom abbreviations
    df['conference'] = df['conference'].map(CONFERENCE_NAME_MAPPING)

    # normalize team names
    df['postseason_champion'] = _normalize_team_names(df['postseason_champion'])

    # apply manual overrides for ineligible teams
    for (override_year, conf), replacement in CONFERENCE_CHAMP_OVERRIDES.items():
        mask = (df['year'] == override_year) & (df['conference'] == conf)
        df.loc[mask, 'postseason_champion'] = replacement

    return df


def _calc_win_pct(record):
    """Calculate win percentage from a 'W-L' record string."""
    wins, losses = map(int, record.split('-'))
    total = wins + losses
    return 0 if total == 0 else wins / total


def scrape_quad_records(dates):
    """
    Scrape quad records from bracketologists.com for the given dates.

    Args:
        dates: list of date strings in 'YYYY-MM-DD' format

    Returns DataFrame with columns:
        team, year, record, quad_1-4_record, non_d1_record,
        wins, losses, win_percentage, quad_1-4_win_percentage
    """
    all_data = []

    for date in dates:
        url = f"https://bracketologists.com/date/{date}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        for team_div in soup.find_all('div', class_='teamBarSmall'):
            team_name = team_div.find('span', class_='teamNameSmall').a.contents[0].strip()
            records = [span.text.strip() for span in team_div.find_all('span', class_='teamRecordSmall')]

            all_data.append({
                'team': team_name,
                'year': int(date[:4]),
                'record': records[0],
                'quad_1_record': records[1],
                'quad_2_record': records[2],
                'quad_3_record': records[3],
                'quad_4_record': records[4],
                'non_d1_record': records[5],
            })

    df = pd.DataFrame(all_data)

    # normalize team names (State -> St., hyphens -> spaces, then mapping)
    df['team'] = df['team'].str.replace('State', 'St.', regex=False)
    df['team'] = df['team'].str.replace('-', ' ', regex=False)
    df['team'] = df['team'].map(TEAM_NAME_MAPPING).fillna(df['team'])

    # parse overall record
    df[['wins', 'losses']] = df['record'].str.split('-', expand=True).astype(int)
    df['win_percentage'] = df['wins'] / (df['wins'] + df['losses'])

    # compute quad win percentages
    for i in range(1, 5):
        df[f'quad_{i}_win_percentage'] = df[f'quad_{i}_record'].apply(_calc_win_pct)

    return df


def scrape_tournament_seeds(years):
    """
    Scrape historical tournament fields from Wikipedia.
    Only needed for training (ground truth labels).

    Returns DataFrame with columns: year, seed, team
    """
    all_data = []

    for year in years:
        url = f"https://en.wikipedia.org/wiki/{year}_NCAA_Division_I_men%27s_basketball_tournament"
        response = requests.get(url, headers={'User-Agent': 'MarchMadnessClassifier/1.0'})
        soup = BeautifulSoup(response.content, 'html.parser')

        tables = soup.find_all('table', class_='wikitable sortable plainrowheaders')

        for table in tables:
            caption = table.find('caption')
            if not caption or 'Regional' not in caption.get_text():
                continue

            rows = table.find_all('tr')
            if len(rows) <= 1:
                continue

            # track current seed for rowspan handling (play-in games)
            current_seed = None

            for row in rows[1:]:
                columns = row.find_all(['td', 'th'])

                if len(columns) < 2:
                    continue

                first_cell = columns[0]

                if 'rowspan' in first_cell.attrs:
                    current_seed = first_cell.get_text(strip=True).replace('*', '')
                    seed = current_seed
                    school_idx = 1
                elif current_seed:
                    seed = current_seed
                    school_idx = 0
                else:
                    seed = first_cell.get_text(strip=True).replace('*', '')
                    school_idx = 1

                if len(columns) <= school_idx:
                    continue

                all_data.append({
                    'year': year,
                    'seed': seed,
                    'team': columns[school_idx].get_text(strip=True),
                })

                # reset current_seed after second row of a rowspan
                if current_seed and 'rowspan' not in first_cell.attrs:
                    current_seed = None

    df = pd.DataFrame(all_data)

    # normalize team names
    df['team'] = _normalize_team_names(df['team'])

    return df
