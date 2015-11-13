
# coding: utf-8

### Definitions

# In[2]:

get_ipython().magic(u'matplotlib inline')
import operator
from sklearn.cluster import *

import functools
import cStringIO
from IPython.display import display, clear_output
from matplotlib.pyplot import imshow
from scipy.stats.stats import pearsonr  
import pylab,copy
from datetime import date
import numpy as np
import pandas as pd
import PIL
from scipy.spatial.distance import *
from PIL import Image ,ImageFilter
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,Normalizer,KernelCenterer,StandardScaler
def movingaverage(values,window):
    weigths = np.repeat(1.0, window)/window
    smas = np.convolve(values, weigths, 'valid')
    return smas # as a numpy array
from pandas.stats.moments import expanding_apply
from __future__ import division
def mean(a):
    return sum(a) / len(a)
def getav(X):
    from operator import add
    a=np.array(X)
    b=list((list((y) for y in x)) for x in a)
    df=pd.DataFrame()
    for x in range(len(b)):
        df[x]=b[x][0]
    df['avg']=float(0)
    for x in range(df.shape[0]):
        df['avg'][x]=sum(df.loc[x])/(len(df.loc[x])-1)
    return df['avg']

def getPerc(X):
        res1=pd.Series(X.shift(-1).Close/X.Close)
        res2=expanding_apply(res1,lambda p:functools.reduce(operator.mul, p, 1)).shift(1)
        return res2.dropna()
#def getRefined(f):
#    truevect=(f.index==f.index[-1])|(((f.High>f.shift(1).High) & (f.High>f.shift(-1).High)& (f.High>f.shift(-2).High)& (f.High>f.shift(2).High))|(((f.Low<f.shift(1).Low) & (f.Low<f.shift(-1).Low)& (f.Low<f.shift(2).Low)& (f.Low<f.shift(-2).Low))))
#    g=f.loc[truevect]
#    return pd.DataFrame([pd.Series(index=pd.date_range(f.index[0],f.index[-1])),g.Close]).transpose().interpolate().fillna(method='bfill')
print 'done'


### Load data

# In[3]:

import pandas.io.data as web
import datetime
import pandas as pd
import math
start = datetime.datetime(1980,1, 1)
end = datetime.datetime(2016, 1, 27)
f=web.DataReader("^GSPC", 'yahoo', start, end)
#getRefined(f).Close.plot()
f.Close.plot(figsize=(16,8))


### Set properties

# In[4]:

pattern_day_start=date(2005,1,18)
pattern_day_end=date(2005,4,5)
project_into_future=80
threshold=0.005


### Define pattern

# In[5]:


dataset_pattern=f.loc[pd.date_range(pattern_day_start, pattern_day_end)].dropna()
#dataset_pattern_refined=getRefined(dataset_pattern)
#dataset_pattern_refined=dataset_pattern_refined[dataset_pattern_refined.index.isin(dataset_pattern.index)]


test_range=f.loc[pd.date_range(f.index[0], pattern_day_start)].dropna()
#test_range_refined=getRefined(test_range)
#test_range_refined=test_range_refined[test_range_refined.index.isin(test_range.index)]

diff_u=pd.Series()
gpc1=getPerc(dataset_pattern)

len_dataset_pattern=len(dataset_pattern)
len_test_range=(len(test_range)-len_dataset_pattern)
dataset_pattern.Close.plot()


### Filling dict of datasets

# In[6]:

mylist=[]
for u in range(len_test_range):
    clear_output(wait=True)
    print 'Filling dict'
    print str(u)+' of '+str(len_test_range)
    mylist+=[pd.Series(getPerc(test_range[u:u+len_dataset_pattern]))]
mylist=np.array(mylist)


# In[ ]:

from sklearn.mixture import VBGMM
from copy import copy
from sklearn.neural_network import BernoulliRBM
kmeans = BernoulliRBM ()
clusters=pd.Series(kmeans.fit_transform(mylist),name='clusters')


# In[ ]:


c=pd.DataFrame(mylist).join(clusters)
c
d=c.loc[((c.clusters==c.clusters.iloc[-1]))]
d
d=d.iloc[:,:-1]
for x in range(len(d)):
    a=2
    d.iloc[[x,-1]].transpose().plot(linewidth=1)
d.iloc[[x,-1],:]
#.iloc[:,[x,-1]]


# In[ ]:



