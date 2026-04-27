import pandas as pd
from pathlib import Path


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV file."""
    df = pd.read_csv(filepath)
    return df


def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Fix messy column names."""
    df = df.rename(columns={
        'PatientId':        'patient_id',
        'AppointmentID':    'appointment_id',
        'Gender':           'gender',
        'ScheduledDay':     'scheduled_day',
        'AppointmentDay':   'appointment_day',
        'Age':              'age',
        'Neighbourhood':    'neighbourhood',
        'Scholarship':      'scholarship',
        'Hipertension':     'hypertension',
        'Diabetes':         'diabetes',
        'Alcoholism':       'alcoholism',
        'Handcap':          'handicap',
        'SMS_received':     'sms_received',
        'No-show':          'no_show'
    })
    return df


def fix_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """Convert No-show text to binary 1/0."""
    # 'Yes' means they DID skip = 1 (positive class)
    # 'No' means they showed up = 0
    df['no_show'] = df['no_show'].map({'Yes': 1, 'No': 0})
    return df


def remove_bad_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with invalid data."""
    # Remove negative ages
    df = df[df['age'] >= 0]

    # Remove ages above 120 (impossible)
    df = df[df['age'] <= 120]

    # Remove rows where appointment is before scheduled day
    df = df[df['appointment_day'] >= df['scheduled_day'].str[:10]]

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new useful features."""

    # Convert to datetime
    df['scheduled_day'] = pd.to_datetime(df['scheduled_day'])
    df['appointment_day'] = pd.to_datetime(df['appointment_day'])

    # Feature 1: days_in_advance (required by exam)
    # How many days before did they book?
    df['days_in_advance'] = (
        df['appointment_day'] - df['scheduled_day']
    ).dt.days

    # Fix any negative values (same day booking = 0)
    df['days_in_advance'] = df['days_in_advance'].clip(lower=0)

    # Feature 2: appointment day of week
    # Monday=0, Sunday=6 — weekends may have different no-show rates
    df['appointment_weekday'] = df['appointment_day'].dt.dayofweek

    return df


def get_features_and_target(df: pd.DataFrame):
    """Return X (features) and y (target) for model training."""
    feature_cols = [
        'age',
        'scholarship',
        'hypertension',
        'diabetes',
        'alcoholism',
        'handicap',
        'sms_received',
        'days_in_advance',
        'appointment_weekday'
    ]

    X = df[feature_cols]
    y = df['no_show']
    return X, y


def preprocess(filepath: str):
    """Full pipeline — load, clean, engineer features."""
    df = load_data(filepath)
    df = fix_column_names(df)
    df = fix_target_column(df)
    df = remove_bad_data(df)
    df = engineer_features(df)
    return df


if __name__ == "__main__":
    # Quick test to make sure everything works
    data_path = Path(__file__).parent.parent / "data" / "KaggleV2-May-2016.csv"
    df = preprocess(str(data_path))
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("No-show distribution:")
    print(df['no_show'].value_counts())
