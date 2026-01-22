"""
Data Preparation Script
Generates standardized CSV files from raw data using preprocessing utilities.
Run this ONCE before using the dashboard or clustering.
"""

import pandas as pd
import sys
from pathlib import Path

# Import preprocessing functions
from utils.preprocessing import (
    load_core_ipc_datasets,
    standardize_ipc_2001_2013,
    standardize_ipc_2014,
    enforce_canonical_schema,
    standardize_women_crime_2001_2013,
    standardize_women_crime_2014,
    standardize_children_crime_2001_2012,
    standardize_children_crime_2013
)

def prepare_core_ipc():
    """Prepare and save core IPC crime data."""
    print("\n1. Processing Core IPC Crimes...")
    
    core = load_core_ipc_datasets("data/raw/core_crime")
    
    df = pd.concat([
        standardize_ipc_2001_2013(core["2001_2012"]),
        standardize_ipc_2001_2013(core["2013"]),
        standardize_ipc_2014(core["2014"]),
    ], ignore_index=True)
    
    core_ipc = enforce_canonical_schema(df)
    
    # Save
    output_path = Path("data/processed/core_crime")
    output_path.mkdir(parents=True, exist_ok=True)
    core_ipc.to_csv(output_path / "core_ipc_standardized.csv", index=False)
    
    print(f"   ✓ Saved {len(core_ipc)} records to data/processed/core_crime/core_ipc_standardized.csv")
    return core_ipc

def prepare_women_crime():
    """Prepare and save women crime data."""
    print("\n2. Processing Women Crimes...")
    
    women_2001_2012 = pd.read_csv("data/raw/women_crime/42_district_wise_crimes_committed_against_women_2001_2012.csv")
    women_2013 = pd.read_csv("data/raw/women_crime/42_district_wise_crimes_committed_against_women_2013.csv")
    women_2014 = pd.read_csv("data/raw/women_crime/42_district_wise_crimes_committed_against_women_2014.csv")
    
    w2001_2012 = standardize_women_crime_2001_2013(women_2001_2012)
    w2013 = standardize_women_crime_2001_2013(women_2013)
    w2014 = standardize_women_crime_2014(women_2014)
    
    women_crime = pd.concat([w2001_2012, w2013, w2014], ignore_index=True)
    
    # Save
    output_path = Path("data/processed/women_crime")
    output_path.mkdir(parents=True, exist_ok=True)
    women_crime.to_csv(output_path / "women_crime_standardized.csv", index=False)
    
    print(f"   ✓ Saved {len(women_crime)} records to data/processed/women_crime/women_crime_standardized.csv")
    return women_crime

def prepare_children_crime():
    """Prepare and save children crime data."""
    print("\n3. Processing Children Crimes...")
    
    children_2001_2012 = pd.read_csv("data/raw/children_crime/03_district_wise_crimes_committed_against_children_2001_2012.csv")
    children_2013 = pd.read_csv("data/raw/children_crime/03_district_wise_crimes_committed_against_children_2013.csv")
    
    c2001_2012 = standardize_children_crime_2001_2012(children_2001_2012)
    c2013 = standardize_children_crime_2013(children_2013)
    
    children_crime = pd.concat([c2001_2012, c2013], ignore_index=True)
    
    # Save
    output_path = Path("data/processed/children_crime")
    output_path.mkdir(parents=True, exist_ok=True)
    children_crime.to_csv(output_path / "children_crime_standardized.csv", index=False)
    
    print(f"   ✓ Saved {len(children_crime)} records to data/processed/children_crime/children_crime_standardized.csv")
    return children_crime

def main():
    print("=" * 60)
    print("Data Preparation for Crime Analysis System")
    print("=" * 60)
    
    try:
        core_ipc = prepare_core_ipc()
        women_crime = prepare_women_crime()
        children_crime = prepare_children_crime()
        
        print("\n" + "=" * 60)
        print("✓ Data preparation complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Run: python models/kmeans_clustering.py")
        print("  2. Run: streamlit run app.py")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
