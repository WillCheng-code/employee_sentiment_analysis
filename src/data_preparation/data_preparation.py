# Section 0: Imports and Initial Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from textblob import TextBlob   # For sentiment analysis (replace with transformer-based LLM as needed)
import os

# Load dataset
df = pd.read_csv("data/raw/test.csv")
df.head()
