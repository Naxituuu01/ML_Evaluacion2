# src/f1kedro/pipelines/ingestion/nodes.py
import pandas as pd
import numpy as np

def merge_datasets(races: pd.DataFrame, drivers: pd.DataFrame, constructors: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Merge raw csvs into a single DataFrame with basic cleaning and features."""
    # merge results with races
    df = results.merge(races[['raceId','year','name','date']], on='raceId', how='left')
    # merge drivers (keep dob, name)
    drivers_sel = drivers[['driverId','forename','surname','dob','nationality']].copy()
    df = df.merge(drivers_sel, on='driverId', how='left')
    # merge constructors
    constructors_sel = constructors[['constructorId','name','nationality']].rename(columns={'name':'constructor_name','nationality':'constructor_nationality'})
    df = df.merge(constructors_sel, on='constructorId', how='left')

    # parse dates and compute age
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df['age_at_race'] = (df['date'] - df['dob']).dt.days / 365.25

    # numeric conversions
    df['points'] = pd.to_numeric(df['points'], errors='coerce').fillna(0)
    df['positionOrder'] = pd.to_numeric(df['positionOrder'], errors='coerce')
    df['grid'] = pd.to_numeric(df['grid'], errors='coerce').fillna(99).astype(int)

    # basic cleaning: drop rows without race/driver
    df = df.dropna(subset=['raceId','driverId'])

    return df