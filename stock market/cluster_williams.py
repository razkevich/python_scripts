
# coding: utf-8

# In[2]:

import pandas as pd
class Densifier(object):
    def fit(self, X, y=None):
        pass
    def fit_transform(self, X, y=None):
        return self.transform(X)
    def transform(self, X, y=None):
        return X.toarray()
def getRefined(f):
    truevect=(f.index==f.index[-1])|(((f.High>f.shift(1).High) & (f.High>f.shift(-1).High)& (f.High>f.shift(-2).High)& (f.High>f.shift(2).High))|(((f.Low<f.shift(1).Low) & (f.Low<f.shift(-1).Low)& (f.Low<f.shift(2).Low)& (f.Low<f.shift(-2).Low))))
    g=f.loc[truevect]
    return g# pd.DataFrame([pd.Series(index=pd.date_range(f.index[0],f.index[-1])),g.Close]).transpose()#.interpolate().fillna(method='bfill')
def getRefined2(f):
    truevect=(f.index==f.index[-1])|(((f>f.shift(1))&(f>f.shift(-1)))|(((f<f.shift(1))&(f<f.shift(-1)))))
    g=f.loc[truevect]
    return g# pd.DataFrame([pd.Series(index=pd.date_range(f.index[0],f.index[-1])),g.Close]).transpose()#.interpolate().fillna(method='bfill')


# In[3]:

import pandas.io.data as web
import datetime
start = datetime.datetime(1970, 1, 1)
end = datetime.datetime(2013, 1, 27)
f=web.DataReader("F", 'yahoo', start, end)
f['ret']=f.shift(-10).Close/f.shift(-4).Open
re=getRefined(f['2011':])
re.Close.plot(figsize=(16,16))


# In[4]:

re


# In[ ]:




# In[5]:

re=getRefined(f)
mydata=pd.DataFrame()
#
mydata['PriceLen']=re.ret
mydata['DateLen']=list((x.days) if hasattr(x,'days') else 0 for x in (re.Date-re.shift(1).Date))
mydata.index=re.Date
mydata.tail()


# In[8]:

from sklearn.cluster import *
from sklearn.mixture import *
km=KMeans(n_clusters=50)
mydata=mydata.dropna()
clusters=km.fit_predict(mydata)
re['cluster']=0
re['cluster'].iloc[1:]=list(x for x in clusters)
re.tail()


# In[32]:

import random
g=pd.DataFrame()
g['return0']=re['ret']>1
g['r']=list((random.randint(0,1000)) for r in xrange(g.shape[0]))
#g['letter1']=re.cluster.dropna()
#g['letter2']=re.shift(1).cluster.dropna()
#g['letter3']=re.shift(2).cluster.dropna()
#g['letter4']=re.shift(3).cluster.dropna()
#g['letter5']=re.shift(4).cluster.dropna()
#g['letter6']=re.shift(5).cluster.dropna()
g


# In[38]:

from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import copy
g.return0=g.return0>0
#y1.return0=y1.return0.apply(float)
from sklearn.ensemble import *
from sklearn.dummy import *
from sklearn.svm import LinearSVC
gbc=GradientBoostingClassifier()
g=g.dropna()

ohc=Pipeline([('b',OneHotEncoder()),('a',Densifier())])
X=g[['r']]
Y=g['return0']

X=ohc.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)
gbc.fit(x_train,y_train)
pr=list((max(x)>0.65) for x in gbc.predict_proba(x_test))
gbc.score(x_test,y_test,pr)


# In[10]:

f['2011':]


# In[ ]:



