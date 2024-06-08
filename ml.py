from pathlib import Path

csv_path_stt = Path('./stt.csv')
csv_path_all = Path('./all.csv')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from tqdm.notebook import tqdm
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

from sklearn.preprocessing import StandardScaler


df_all = pd.read_csv(csv_path_all)      # a DataFrame object
df_all = df_all.sort_values(by='date')  # csv are not guarenteed to be ordered by date
df_all.head(10)