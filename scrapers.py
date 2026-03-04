import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from kenpompy.utils import login
from kenpompy.misc import get_pomeroy_ratings

from config import CONFERENCE_NAME_MAPPING, CONFERENCE_SLUG_MAPPING, TEAM_NAME_MAPPING, CONFERENCE_CHAMP_OVERRIDES

_SR_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
}

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


def scrape_torvik(years):
    """
    Fetch team ratings from Bart Torvik for the given years.
    Downloads CSV files directly from barttorvik.com (no authentication required).

    Returns DataFrame with columns:
        team, year, conference, record, sos
    """
    from io import StringIO

    frames = []
    for year in years:
        url = f'https://barttorvik.com/{year}_team_results.csv'
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        df['year'] = year
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    # select and rename columns
    df = df[['team', 'year', 'conf', 'record', 'sos']]
    df = df.rename(columns={'conf': 'conference'})

    # normalize team names to match other scrapers
    df['team'] = _normalize_team_names(df['team'])

    return df


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
    data exists at all for a year (e.g. mid-season), falls back to scraping
    individual conference standings pages for the current leader by conference
    win percentage.

    Returns DataFrame with columns: year, conference, postseason_champion
    """
    tourney_data = []
    leaders_frames = []

    for year in years:
        url = f"https://www.sports-reference.com/cbb/seasons/men/{year}.html"
        response = requests.get(url, headers=_SR_HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")

        conf_names = [tag.text.strip() for tag in soup.find_all("td", {"data-stat": "conf_name"})]
        tourney_champs = [tag.text.strip() for tag in soup.find_all("td", {"data-stat": "conf_champ_post"})]

        has_tourney_data = any(c for c in tourney_champs)
        if has_tourney_data:
            for conf, champ in zip(conf_names, tourney_champs):
                tourney_data.append((year, conf, champ))
        else:
            # mid-season: scrape conference standings for current leaders
            print(f'  No tournament data for {year}, scraping conference standings...')
            leaders_frames.append(scrape_conference_leaders(year))

    # process tournament champion data (full conference names, raw team names)
    frames = []

    if tourney_data:
        df = pd.DataFrame(tourney_data, columns=['year', 'conference', 'postseason_champion'])
        df = df[df['conference'] != 'Independent']
        df['conference'] = df['conference'].map(CONFERENCE_NAME_MAPPING)
        df['postseason_champion'] = _normalize_team_names(df['postseason_champion'])

        for (override_year, conf), replacement in CONFERENCE_CHAMP_OVERRIDES.items():
            mask = (df['year'] == override_year) & (df['conference'] == conf)
            df.loc[mask, 'postseason_champion'] = replacement

        frames.append(df)

    # leaders data is already in KenPom format from scrape_conference_leaders
    frames.extend(leaders_frames)

    if frames:
        return pd.concat(frames, ignore_index=True)

    return pd.DataFrame(columns=['year', 'conference', 'postseason_champion'])


def scrape_conference_leaders(year):
    """
    Scrape current conference standings from sports-reference.com to find
    the team leading each conference by conference win percentage.

    Used as a mid-season fallback when conference tournament champions
    are not yet determined.

    Returns DataFrame with columns: year, conference, postseason_champion
    """
    import time
    all_data = []

    for kenpom_abbr, slug in CONFERENCE_SLUG_MAPPING.items():
        time.sleep(3)  # rate limit: sports-reference blocks rapid requests
        url = f"https://www.sports-reference.com/cbb/conferences/{slug}/men/{year}.html"
        try:
            response = requests.get(url, headers=_SR_HEADERS)
            # retry once after longer wait on rate limit
            if response.status_code == 429:
                print(f'    {slug}: rate limited, retrying in 30s...')
                time.sleep(30)
                response = requests.get(url, headers=_SR_HEADERS)
            if response.status_code != 200:
                print(f'    {slug}: HTTP {response.status_code}')
                continue
        except requests.RequestException:
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        best_team = None
        best_pct = -1.0

        for row in soup.find_all("tr"):
            school_cell = row.find("td", {"data-stat": "school_name"})
            wins_cell = row.find("td", {"data-stat": "wins_conf"})
            losses_cell = row.find("td", {"data-stat": "losses_conf"})

            if not school_cell or not wins_cell or not losses_cell:
                continue

            team = school_cell.get_text(strip=True)
            try:
                wins = int(wins_cell.get_text(strip=True))
                losses = int(losses_cell.get_text(strip=True))
            except ValueError:
                continue

            total = wins + losses
            if total == 0:
                continue

            pct = wins / total
            if pct > best_pct:
                best_pct = pct
                best_team = team

        if best_team:
            all_data.append((year, kenpom_abbr, best_team))

    df = pd.DataFrame(all_data, columns=['year', 'conference', 'postseason_champion'])

    # normalize team names (hyphens -> spaces, then standard normalization)
    df['postseason_champion'] = df['postseason_champion'].str.replace('-', ' ', regex=False)
    df['postseason_champion'] = _normalize_team_names(df['postseason_champion'])

    print(f'  Found leaders for {len(df)} conferences')

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

    # if no data was found, try the previous day (bracketologists.com
    # may not have today's data yet)
    if not all_data and len(dates) == 1:
        from datetime import datetime, timedelta
        original = datetime.strptime(dates[0], '%Y-%m-%d')
        prev_date = (original - timedelta(days=1)).strftime('%Y-%m-%d')
        print(f'  No quad data for {dates[0]}, trying {prev_date}...')
        return scrape_quad_records([prev_date])

    df = pd.DataFrame(all_data)

    if df.empty:
        return df

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
        response = requests.get(url, headers=_SR_HEADERS)
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
