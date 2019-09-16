#%%
from __future__ import division

import itertools
import pickle
import numpy as np 
import scipy.io as sio
import pandas as pd
import scipy.sparse as ss 
import scipy.spatial.distance as ssd

from collections import defaultdict
from sklearn.preprocessing import normalize

#%%
dpath = "./MachineLearning/Music_Recommendation/Data/"
f = pd.read_csv(dpath+"ceshi.csv")
f.head()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
LE_f = le.fit(f["A"])
f["A"] = LE_f.transform(f["A"])
LE_f = le.fit(f["B"].astype(str))
f["B"] = LE_f.transform(f["B"].astype(str))
print(LE_f)

#%%
