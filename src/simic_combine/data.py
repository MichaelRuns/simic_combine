"""Data loading and cleaning for hypothyroid kitten study."""

from pathlib import Path
import pandas as pd
import numpy as np

# Control status definitions
CONTROL_STATUS = {
    1: "Controlled (normal TT4, normal TSH)",
    2: "Undertreated (low TT4 and/or high TSH)",
    3: "Overtreated (high TT4, normal TSH)",
    4: "Elevated TT4 and TSH",
    5: "Normal TT4 and elevated TSH",
}


def load_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Load and clean the hypothyroid kitten study data.

    Args:
        filepath: Path to CSV file. If None, uses default data location.

    Returns:
        Cleaned DataFrame with numeric columns and derived features.
    """
    if filepath is None:
        # Default to data directory relative to package
        # __file__ is src/simic_combine/data.py -> go up 3 levels to repo root
        filepath = Path(__file__).parent.parent.parent / "data" / "Hypothyroid kitten study - _Clinical Cases UTD (11_30_25).csv"

    df = pd.read_csv(filepath)

    # Drop the reference range row (row index 0 after header)
    data = df.drop(index=0).copy()

    # Reset index for clean iteration
    data = data.reset_index(drop=True)

    # Convert key columns to numeric
    numeric_cols = ["Weight", "Age ", "T4", "Frequency", "Control"]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Rename 'Age ' to 'Age' (has trailing space in CSV)
    if "Age " in data.columns:
        data = data.rename(columns={"Age ": "Age"})

    # Also convert Dosage T4 column (already calculated as mcg/kg)
    if "Dosage T4" in data.columns:
        data["Dosage T4"] = pd.to_numeric(data["Dosage T4"], errors="coerce")

    return data


def get_treatment_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to treatment observations with valid dose, weight, and control status.

    Args:
        df: Raw dataframe from load_data()

    Returns:
        DataFrame with only valid treatment observations and derived features.
    """
    # Filter to rows where treatment has started (T4 dose > 0)
    treatment = df[df["T4"] > 0].copy()

    # Require weight and control status
    treatment = treatment.dropna(subset=["Weight", "Control"])

    # Calculate daily dose (T4 is single dose, Frequency is times per day)
    # Default frequency to 2 if missing (most common)
    treatment["Frequency"] = treatment["Frequency"].fillna(2)
    treatment["daily_dose_mcg"] = treatment["T4"] * treatment["Frequency"]

    # Calculate dose per kg
    treatment["dose_per_kg"] = treatment["daily_dose_mcg"] / treatment["Weight"]

    # Ensure control is integer
    treatment["Control"] = treatment["Control"].astype(int)

    return treatment


def get_controlled_cases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to only controlled (Control=1) cases for model fitting.

    Args:
        df: Treatment dataframe from get_treatment_data()

    Returns:
        DataFrame with only Control=1 observations.
    """
    return df[df["Control"] == 1].copy()


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for the treatment data.

    Args:
        df: Treatment dataframe from get_treatment_data()

    Returns:
        Dictionary of summary statistics.
    """
    n_kittens = df["Animal ID"].nunique() if "Animal ID" in df.columns else df.iloc[:, 0].nunique()
    n_observations = len(df)

    control_counts = df["Control"].value_counts().sort_index()

    return {
        "n_kittens": n_kittens,
        "n_observations": n_observations,
        "control_distribution": control_counts.to_dict(),
        "weight_range": (df["Weight"].min(), df["Weight"].max()),
        "age_range": (df["Age"].min(), df["Age"].max()) if "Age" in df.columns else None,
        "dose_per_kg_range": (df["dose_per_kg"].min(), df["dose_per_kg"].max()),
    }
