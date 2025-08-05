from sklearn.datasets import load_iris
import pandas as pd

def load_data():
    data = load_iris(as_frame=True)
    df = data.frame
    return df
