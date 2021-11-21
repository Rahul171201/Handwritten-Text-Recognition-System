import numpy as np
import pandas as pd
# %matplotlib inline
import matplotlib
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(tol=0.1)

dt = pd.read_csv("A_Z Handwritten Data.csv");
df = pd.DataFrame(dt);
print(df);
