# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("agg")

# Importing the dataset
iris_dataset = pd.read_csv("datasets/iris.csv")
print(iris_dataset.head())

