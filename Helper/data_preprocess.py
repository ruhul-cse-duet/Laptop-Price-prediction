from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as pd

def prepare_data(path_data):
    # Read data from path
    df = pd.read_csv(path_data)

    