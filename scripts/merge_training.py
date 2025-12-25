import pandas as pd
import numpy as np
from pathlib import Path

# 1. DYNAMIC PATH SETUP
ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / 'data' / 'processed'
MODELING_DIR = ROOT_DIR / 'data' / 'modeling'

MODELING_DIR.mkdir(parents=True, exist_ok=True)

def load_raw_data():
    files = [
        'races.csv', 'results.csv', 'drivers.csv', 'constructors.csv',
        'qualifying.csv', 'constructor_standings.csv', 'driver_standings.csv',
        'circuits.csv', 'status.csv'
    ]
    data = {}
    for f in files:
        path = PROCESSED_DIR / f
        if path.exists():
            data[f.replace('.csv', '')] = pd.read_csv(path)
        else:
            print(f"Warning: {f} not found!")
    return data

def make_merged_df(races_df, all_data):
    # Local references
    results = all_data['results']
    drivers = all_data['drivers']
    constructors = all_data['constructors'].copy()
    qualifying = all_data['qualifying'].copy()
    driver_st = all_data['driver_standings'].copy()
    const_st = all_data['constructor_standings'].copy()
    circuits = all_data['circuits']
    status = all_data['status']

    # --- PRE-MERGE RENAMING (Explicitly naming to avoid conflicts) ---
    races_df = races_df.rename(columns={'name': 'race_name'})
    constructors = constructors.rename(columns={'name': 'constructor_name'})
    driver_st = driver_st.rename(columns={'points': 'points_ds', 'position': 'position_ds', 'wins': 'wins_ds'})
    const_st = const_st.rename(columns={'points': 'points_cs', 'position': 'position_cs', 'wins': 'wins_cs'})
    qualifying = qualifying.rename(columns={'position': 'qualifying_pos'})

    # --- MERGING ---
    # 1. Races + Circuits
    races_with_circuits = races_df.merge(circuits[['circuitId', 'alt']], on='circuitId', how='left')
    
    # 2. Results + Races
    df = results.merge(races_with_circuits[['raceId', 'year', 'round', 'race_name', 'alt']], on='raceId', how='inner')
    
    # 3. Add Standings
    df = df.merge(driver_st[['raceId', 'driverId', 'points_ds', 'position_ds', 'wins_ds']], on=['raceId','driverId'], how='left')
    df = df.merge(const_st[['raceId', 'constructorId', 'points_cs', 'position_cs', 'wins_cs']], on=['raceId','constructorId'], how='left')
    
    # 4. Add Driver/Constructor Names
    df = df.merge(drivers[['driverId', 'driverRef']], on='driverId', how='left')
    df = df.merge(constructors[['constructorId', 'constructor_name']], on='constructorId', how='left')
    
    # 5. Add Qualifying
    df = df.merge(qualifying[['raceId', 'driverId', 'constructorId', 'qualifying_pos']], 
                  on=['raceId', 'driverId', 'constructorId'], how='left')
    
    # 6. Add Status
    df = df.merge(status[['statusId', 'status']], on='statusId', how='left')

    # --- FEATURE ENGINEERING ---
    df = df.sort_values(['year', 'round', 'positionOrder'])

    # Shifted Features (Pre-Race only)
    df['driver_points_pre_race'] = df.groupby('driverId')['points_ds'].shift(1).fillna(0)
    df['driver_pos_pre_race'] = df.groupby('driverId')['position_ds'].shift(1).fillna(20)
    df['driver_wins_pre_race'] = df.groupby('driverId')['wins_ds'].shift(1).fillna(0)
    df['constructor_points_pre_race'] = df.groupby('constructorId')['points_cs'].shift(1).fillna(0)
    df['constructor_pos_pre_race'] = df.groupby('constructorId')['position_cs'].shift(1).fillna(10)

    # Momentum (Form)
    df['driver_form'] = df.groupby('driverId')['positionOrder'].transform(lambda x: x.ewm(alpha=0.3).mean().shift(1)).fillna(15)
    
    team_avg = df.groupby(['raceId', 'constructorId'])['positionOrder'].mean().reset_index()
    team_avg['team_form'] = team_avg.groupby('constructorId')['positionOrder'].transform(lambda x: x.ewm(alpha=0.3).mean().shift(1)).fillna(15)
    df = df.merge(team_avg[['raceId', 'constructorId', 'team_form']], on=['raceId', 'constructorId'], how='left')

    # Targets
    df['top3'] = (df['positionOrder'] <= 3).astype(int)
    df['top10'] = (df['positionOrder'] <= 10).astype(int)
    df['DNF'] = (~df['status'].str.contains('Finished|\\+|=', na=False)).astype(int)
    
    df['qualifying_pos'] = df['qualifying_pos'].fillna(df['grid'])
    df['grid_penalty'] = df['grid'] - df['qualifying_pos']
    df['pos_gain_loss'] = df['grid'] - df['positionOrder']

    return df

if __name__ == "__main__":
    print("Loading data...")
    raw_data = load_raw_data()
    
    train_races = raw_data['races'][raw_data['races']['year'].between(2017, 2022)]
    test_races = raw_data['races'][raw_data['races']['year'].between(2023, 2024)]

    print("Building training set...")
    train_df = make_merged_df(train_races, raw_data)
    print("Building testing set...")
    test_df = make_merged_df(test_races, raw_data)

    # Final columns to keep - UPDATED to match our new names
    core_cols = [
    'year', 'race_name', 'circuitRef', 'alt', 'driverRef', 'constructor_name', 
    'grid', 'qualifying_pos', 'grid_penalty', 'driver_points_pre_race', 
    'driver_pos_pre_race', 'driver_form', 'team_form', 
    'constructor_points_pre_race', 'top3', 'top10', 'DNF', 'pos_gain_loss'
]

    # Save
    train_df[core_cols].to_csv(MODELING_DIR / 'train_f1_2017_2022.csv', index=False)
    test_df[core_cols].to_csv(MODELING_DIR / 'test_f1_2023_2024.csv', index=False)
    
    print("-" * 30)
    print(f"SUCCESS! Files saved to {MODELING_DIR}")