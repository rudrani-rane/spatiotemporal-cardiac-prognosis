import os
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

LABELS_PATH = os.path.join(PROJECT_ROOT, "data", "metadata", "video_labels.csv")


def validate_labels():

    print("Validating labels...")

    df = pd.read_csv(LABELS_PATH)

    print(f"Total samples: {len(df)}")

    # Check missing values
    if df.isnull().sum().sum() > 0:
        print("⚠ Missing values detected")
    else:
        print("✔ No missing values")

    # EF range check
    if not df["EF"].between(0, 100).all():
        print("⚠ EF out of range detected")
    else:
        print("✔ EF values valid")
        
    # Split Check 
    if not df["split"].isin([0,1,2]).all():
        print("⚠ Invalid split encoding detected")
    else: 
        print("✔ Split encoding valid")

    # SV should not be negative
    if not (df["SV"] >= 0).all():
        print("⚠ Negative SV values detected")
    else:
        print("✔ SV values valid")

    print("Split distribution:")
    print(df["split"].value_counts())


if __name__ == "__main__":
    validate_labels()
