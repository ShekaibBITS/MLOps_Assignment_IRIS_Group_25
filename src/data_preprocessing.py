from sklearn.datasets import load_iris
# import pandas as pd
import os


def load_data():
    data = load_iris(as_frame=True)
    df = data.frame
    return df


def save_data(filepath="data/iris.csv"):
    """Save the Iris dataset to CSV for DVC tracking."""
    df = load_data()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure 'data' folder exists
    df.to_csv(filepath, index=False)
    print(f"Iris dataset saved to {filepath}")


if __name__ == "__main__":
    save_data()
