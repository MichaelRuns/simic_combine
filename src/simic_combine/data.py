"""Data loading and cleaning for hypothyroid kitten study."""

from pathlib import Path
import pandas as pd
import numpy as np

# Animal ID normalization mapping: variant names → canonical names
ANIMAL_ID_NORMALIZATION: dict[str, str | None] = {
    "Bauhaus(Waldorf)": "Waldorf",
    "Pancake Circus aka Kronk": "Kronk",
    "Stratus aka Figaro": "Figaro",
    "Ozzy (oral T3 started)": "Ozzy",
    "Mackere/Hedgie": "Mackerel/Hedgie",
    "Kitten/Hedgie": "Hedgie",
    "Misty OKP": "Misty",
    "# first visits": None,  # Not an animal — drop this row
}

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


def normalize_animal_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize variant animal ID names to canonical forms.

    Applies ANIMAL_ID_NORMALIZATION mapping and drops rows where the
    canonical name is None (e.g., non-animal entries like '# first visits').

    Args:
        df: DataFrame with 'Animal ID' column.

    Returns:
        DataFrame with normalized animal IDs (rows with None mapping removed).
    """
    id_col = "Animal ID" if "Animal ID" in df.columns else df.columns[0]
    df = df.copy()

    # Strip whitespace from animal IDs
    df[id_col] = df[id_col].astype(str).str.strip()

    # Apply normalization mapping
    df[id_col] = df[id_col].map(lambda x: ANIMAL_ID_NORMALIZATION.get(x, x))

    # Drop rows where animal ID mapped to None
    df = df[df[id_col].notna()].copy()

    return df


def get_mixed_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare treatment data for mixed-effects model fitting.

    Takes the output of get_treatment_data() and adds log-transformed
    columns and control status indicator needed for the LMM.

    Args:
        df: Treatment dataframe from get_treatment_data().

    Returns:
        DataFrame with added columns: log_dose, log_weight, is_controlled,
        and normalized Animal_ID.
    """
    id_col = "Animal ID" if "Animal ID" in df.columns else df.columns[0]
    df = normalize_animal_ids(df)

    # Require positive values for log transform
    df = df[(df["daily_dose_mcg"] > 0) & (df["Weight"] > 0)].copy()

    df["log_dose"] = np.log(df["daily_dose_mcg"])
    df["log_weight"] = np.log(df["Weight"])
    df["is_controlled"] = (df["Control"] == 1).astype(int)
    df["Animal_ID"] = df[id_col]

    return df


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
