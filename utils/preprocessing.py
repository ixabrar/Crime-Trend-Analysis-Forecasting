import pandas as pd
from pathlib import Path


def load_core_ipc_datasets(base_path: str) -> dict:
    """
    Load district-wise IPC crime datasets (2001–2014).
    """
    base_path = Path(base_path)

    datasets = {
        "2001_2012": pd.read_csv(
            base_path / "01_district_wise_crimes_committed_ipc_2001_2012.csv"
        ),
        "2013": pd.read_csv(
            base_path / "01_district_wise_crimes_committed_ipc_2013.csv"
        ),
        "2014": pd.read_csv(
            base_path / "01_district_wise_crimes_committed_ipc_2014.csv"
        ),
    }

    return datasets

def normalize_columns(df):
    """
    Normalize column names by converting to lowercase and replacing spaces with underscores.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("&","and")
        .str.replace("/", "_")
        .str.replace(" ", "_")
        .str.replace("_+" , "_" , regex=True)
    )
    return df

def standardize_ipc_2001_2013(df):
    """
    Standardize the IPC dataset from 2001 to 2013.
    """
    df = normalize_columns(df)

    out = df[["state_ut", "district", "year"]].copy()
    
    out["murder"] = df["murder"]
    out["attempt_to_murder"] = df["attempt_to_murder"]
    out["culpable_homicide"] = df["culpable_homicide_not_amounting_to_murder"]
    out["rape"] = df["rape"]
    out["kidnapping_abduction"] = df["kidnapping_and_abduction"]
    out["dacoity"] = df["dacoity"]
    out["robbery"] = df["robbery"]
    out["burglary"] = df["burglary"]
    out["theft"] = df["theft"]
    out["auto_theft"] = df["auto_theft"]
    out["riots"] = df["riots"]
    out["hurt_grievous_hurt"] = df["hurt_grevious_hurt"]
    out["dowry_deaths"] = df["dowry_deaths"]
    out["assault_on_women"] = df["assault_on_women_with_intent_to_outrage_her_modesty"]
    out["insult_to_modesty_of_women"] = df["insult_to_modesty_of_women"]
    out["cruelty_by_husband_or_relatives"] = df["cruelty_by_husband_or_his_relatives"]
    out["importation_of_girls"] = df["importation_of_girls_from_foreign_countries"]
    out["causing_death_by_negligence"] = df["causing_death_by_negligence"]
    out["other_ipc_crimes"] = df["other_ipc_crimes"]
    out["total_ipc_crimes"] = df["total_ipc_crimes"]
    
    return out

def standardize_ipc_2014(df):
    """
    Standardize the IPC dataset from 2014.
    """
    df =   normalize_columns(df)

    out = df[["states_uts", "district","year"]].copy()
    out = out.rename(columns={"states_uts": "state_ut"})

    out["murder"] = df["murder"]
    out["attempt_to_murder"] = df["attempt_to_commit_murder"]
    out["culpable_homicide"] = df["culpable_homicide_not_amounting_to_murder"]

    rape_cols = [
        "rape",
        "custodial_rape",
        "rape_other_than_custodial"
    ]
    out["rape"] = df[rape_cols].sum(axis=1)

    kidnap_cols = [c for c in df.columns if c.startswith("kidnapping")]
    out["kidnapping_abduction"] = df[kidnap_cols].sum(axis=1)

    out["burglary"] = df["criminal_trespass_burglary"]
    out["theft"] = df["theft"]
    out["auto_theft"] = df["auto_theft"]

    out["dacoity"] = df.filter(like="dacoity").sum(axis=1)
    out["robbery"] = df["robbery"]
    out["riots"] = df.filter(like="riots").sum(axis=1)
    out["hurt_grievous_hurt"] = df[["hurt", "grievous_hurt"]].sum(axis=1)

    out["dowry_deaths"] = df["dowry_deaths"]
    out["assault_on_women"] = df["assault_on_women_with_intent_to_outrage_her_modesty"]
    out["insult_to_modesty_of_women"] = df["insult_to_the_modesty_of_women"]
    out["cruelty_by_husband_or_relatives"] = df["cruelty_by_husband_or_his_relatives"]
    out["importation_of_girls"] = df["importation_of_girls_from_foreign_country"]

    # Other
    out["causing_death_by_negligence"] = df["causing_death_by_negligence"]
    out["other_ipc_crimes"] = df["other_ipc_crimes"]
    out["total_ipc_crimes"] = df["total_cognizable_ipc_crimes"]

    return out


def enforce_canonical_schema(df):
    """
    Enforce canonical schema by selecting and ordering columns consistently.
    """
    canonical_columns = [
        'state_ut', 'district', 'year',
        'murder', 'attempt_to_murder', 'culpable_homicide', 'rape',
        'kidnapping_abduction', 'dacoity', 'robbery', 'riots',
        'hurt_grievous_hurt', 'dowry_deaths',
        'burglary', 'theft', 'auto_theft',
        'assault_on_women', 'insult_to_modesty_of_women',
        'cruelty_by_husband_or_relatives', 'importation_of_girls',
        'causing_death_by_negligence', 'other_ipc_crimes', 'total_ipc_crimes'
    ]
    
    # Select only the canonical columns that exist in the dataframe
    available_cols = [col for col in canonical_columns if col in df.columns]
    return df[available_cols].copy()

def standardize_women_crime_2001_2013(df):
    df = normalize_columns(df)

    out = df[["state_ut", "district", "year"]].copy()
    
    out["rape"] = df["rape"]
    out["kidnapping_abduction"] = df["kidnapping_and_abduction"]
    out["dowry_deaths"] = df["dowry_deaths"]
    out["assault_on_women"] = df["assault_on_women_with_intent_to_outrage_her_modesty"]
    out["insult_to_modesty_of_women"] = df["insult_to_modesty_of_women"]
    out["cruelty_by_husband_or_relatives"] = df["cruelty_by_husband_or_his_relatives"]
    out["importation_of_girls"] = df["importation_of_girls"]

    return out


def standardize_women_crime_2014(df):
    df = normalize_columns(df)

    out = df[["states_uts", "district", "year"]].rename(
        columns={"states_uts": "state_ut"}
    )

    # Rape
    rape_cols = [
        "rape",
        "custodial_rape",
        "custodial_gang_rape",
        "custodial_other_rape",
        "rape_other_than_custodial",
        "rape_gang_rape",
        "rape_others",
    ]
    rape_cols = [c for c in rape_cols if c in df.columns]
    out["rape"] = df[rape_cols].sum(axis=1)

    # Kidnapping & Abduction
    kidnap_cols = [
        c for c in df.columns
        if "kidnap" in c and "attempt" not in c
    ]
    out["kidnapping_abduction"] = df[kidnap_cols].sum(axis=1)

    # Dowry deaths
    out["dowry_deaths"] = df["dowry_deaths"]

    # Assault / Insult
    out["assault_on_women"] = df["assault_on_women_with_intent_to_outrage_her_modesty_total"]
    out["insult_to_modesty_of_women"] = df["insult_to_the_modesty_of_women_total"]

    # Cruelty / Importation
    out["cruelty_by_husband_or_relatives"] = df["cruelty_by_husband_or_his_relatives"]
    out["importation_of_girls"] = df["importation_of_girls_from_foreign_country"]

    return out


def standardize_children_crime_2001_2012(df):
    """
    Standardize children crime dataset from 2001-2012.
    """
    df = normalize_columns(df)
    
    out = df[["state_ut", "district", "year"]].copy()
    
    out["murder"] = df["murder"]
    out["rape"] = df["rape"]
    out["kidnapping_abduction"] = df["kidnapping_and_abduction"]
    out["foeticide"] = df["foeticide"]
    out["abetment_of_suicide"] = df["abetment_of_suicide"]
    out["exposure_and_abandonment"] = df["exposure_and_abandonment"]
    out["procuration_of_minor_girls"] = df["procuration_of_minor_girls"]
    out["buying_of_girls_for_prostitution"] = df["buying_of_girls_for_prostitution"]
    out["selling_of_girls_for_prostitution"] = df["selling_of_girls_for_prostitution"]
    out["prohibition_of_child_marriage_act"] = df["prohibition_of_child_marriage_act"]
    out["other_crimes"] = df["other_crimes"]
    out["total"] = df["total"]
    
    return out


def standardize_children_crime_2013(df):
    """
    Standardize children crime dataset from 2013.
    In 2013, murder is split into infanticid + other_murder.
    """
    df = normalize_columns(df)
    
    out = df[["state_ut", "district", "year"]].copy()
    
    # Combine infanticid + other_murder into murder
    out["murder"] = df["infanticid"] + df["other_murder"]
    out["rape"] = df["rape"]
    out["kidnapping_abduction"] = df["kidnapping_and_abduction"]
    out["foeticide"] = df["foeticide"]
    out["abetment_of_suicide"] = df["abetment_of_suicide"]
    out["exposure_and_abandonment"] = df["exposure_and_abandonment"]
    out["procuration_of_minor_girls"] = df["procuration_of_minor_girls"]
    out["buying_of_girls_for_prostitution"] = df["buying_of_girls_for_prostitution"]
    out["selling_of_girls_for_prostitution"] = df["selling_of_girls_for_prostitution"]
    out["prohibition_of_child_marriage_act"] = df["prohibition_of_child_marriage_act"]
    out["other_crimes"] = df["other_crimes"]
    out["total"] = df["total"]
    
    return out

