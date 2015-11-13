
# coding: utf-8

# In[40]:

import Quandl
a=Quandl.get('ODA/POILBRE_USD',trim_start='2000-01-01')


# In[41]:

def dst(a,b):
    return (sum(map(lambda x, y: ((x - y)/x)**2, a, b))/n)**0.5


# In[42]:

# timeseries transformation from Price to % change
import pandas as pd
n=len(a)
n_days=5
price_ch =pd.Series(a['Value']/a.Value.shift(-1)-1).dropna()
result=[price_ch[u:u+n_days] for u in range(0,n-n_days+1)]


# In[47]:


l=pd.DataFrame([sum(result[0].reset_index().Value-x.reset_index().Value)**2,x.reset_index().Date[0].date()] for x in result)
l.index=l[1]


# In[55]:

l[l[0]==min(l[0][1:])]


# In[65]:

l[0][0:4].plot()
l[0][datetime.date(2002,9,30):datetime.date('2002-10-5')].plot()


# In[64]:

l[0].ix

