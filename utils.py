import torch
import random
import numpy as np
import pandas as pd


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)


def preprocess(df):
    pass