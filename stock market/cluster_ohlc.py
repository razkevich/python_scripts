# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 14:45:05 2015

@author: razkevich
"""

import pandas.io.data as web
import datetime
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2016, 1, 27)
f=web.DataReader("F", 'yahoo', start, end)

import pandas as pd
f2=pd.DataFrame()
f2['a1']=f.Open/f.shift(1).Close
f2['a2']=f.High/f.shift(1).Close
f2['a3']=f.Low/f.shift(1).Close
f2['a4']=f.Close/f.shift(1).Close
#f2['a1']=f.Close/f.shift(1).Close
#f2['a2']=f.Close/pd.rolling_mean(f.Close,5)
#f2['a3']=pd.rolling_mean(f.Close,5)/pd.rolling_mean(f.Close,13)
#f2['a5']=f.Volume/pd.rolling_mean(f.Volume,21)


#f2['C/MA']=f.Close/pd.rolling_mean(f.Close,5)
f2.tail()
from sklearn.cluster import *
from sklearn.mixture import *
km=KMeans(n_clusters=50)
f2=f2.fillna(method='bfill')
clusters=km.fit_predict(f2)
tmp=list(x for x in clusters)
f.loc[:,'cluster']=tmp
f
f0=f
f0['c2']=f.shift(-1).cluster
f0['c3']=f.shift(-2).cluster
#f0['c4']=f.shift(-3).cluster
#f0['c5']=f.shift(-4).cluster
myif=(f0.cluster==8)&(f0.c2==13)&(f0.c3==13)
f0[myif|myif.shift(1)|myif.shift(2)]



g=pd.DataFrame()
g['return0']=f.shift(-1).Close/f.shift(-1).Open>1
g['letter1']=f.cluster.dropna()
#g['r']=list((random.randint(0,30)) for r in xrange(g.shape[0]))
g['letter2']=f.shift(1).cluster.dropna()
g['letter3']=f.shift(2).cluster.dropna()
#g['letter4']=f.shift(3).cluster.dropna()
#g['letter5']=f.shift(4).cluster.dropna()
#g['letter3']=f.shift(2).cluster.dropna()

g=g.dropna()
g.tail()
class Densifier(object):
    def fit(self, X, y=None):
        pass
    def fit_transform(self, X, y=None):
        return self.transform(X)
    def transform(self, X, y=None):
        return X.toarray()
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline    
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
import copy

#y1.return0=y1.return0.apply(float)
from sklearn.ensemble import *
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
gbc=GaussianNB()
ohc=Pipeline([('b',OneHotEncoder()),('a',Densifier())])

X=g.iloc[:,1:]
Y=g.iloc[:,0]

X=ohc.fit_transform(X)
t=0.9
pr=list((max(x)>t) for x in gbc.predict_proba(x_test))
#gbc.score(x_test,y_test,pr)
[gbc.score(x_test,y_test,pr),

sum(list((max(x)>t) for x in gbc.predict_proba(x_test)))]
h=g.groupby(by=['letter1','letter2','letter3','letter4','letter5']).sum()
h2=g.groupby(by=['letter1','letter2','letter3','letter4','letter5']).mean()
h[h.return0>2]