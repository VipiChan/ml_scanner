# Final project cqf
# Subject: Deep Learning
# Author: Vipin Chandra

# ==========================================
# 1. Standard Python Libraries
# ==========================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
import time
import urllib.parse
import datetime as dt
from pathlib import Path
from pprint import pprint
import warnings

# Suppress warnings early
warnings.filterwarnings("ignore")

# ==========================================
# 2. Data Manipulation & Feature Selection
# ==========================================
import pandas as pd
import numpy as np
import pandas_ta as ta
from boruta import BorutaPy

# ==========================================
# 3. Visualization
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Set global plotting theme
sns.set_theme(style="darkgrid", palette="colorblind")

# ==========================================
# 4. Scikit-Learn (Machine Learning)
# ==========================================
# Base & Pipelines
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder

# Model Selection
from sklearn.model_selection import (
    train_test_split, TimeSeriesSplit, cross_val_score,
    GridSearchCV, RandomizedSearchCV
)

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    auc, roc_auc_score, roc_curve, RocCurveDisplay
)

# ==========================================
# 5. TensorFlow & Keras (Deep Learning)
# ==========================================
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Accuracy, AUC, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.utils import plot_model

# ==========================================
# 6. Keras Tuner & TensorBoard (Hyperparameter Tuning)
# ==========================================
import keras_tuner as kt
from keras_tuner import Hyperband, BayesianOptimization, RandomSearch, HyperParameters
from tensorboard import program

# ==========================================
# 7. Environment Specific & Custom Scripts
# ==========================================
try:
    from google.colab import output
except ImportError:
    output = None

from source.helper import *

# end of file