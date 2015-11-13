
# coding: utf-8

# In[43]:
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    import matplotlib.pyplot as plt
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
   
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
import matplotlib as plt
from sklearn.preprocessing import *
import pandas as pd,random,Quandl
from sklearn.metrics import confusion_matrix
cot=pd.DataFrame()
brent=pd.DataFrame()
def loaddata(str_1, str_2):
    global cot,brent
    cot=Quandl.get(str_1, authtoken="3xoNoz5iCMsyH3NngTz5")
    brent=Quandl.get(str_2, authtoken="3xoNoz5iCMsyH3NngTz5")
    
def svm_classify(threshold):
    
    data=pd.DataFrame()
    i=0
    xprev=0
    xprev2=0
    for x in cot.columns[:-1]:
        data[x]=cot[x]/pd.rolling_mean(cot[x],5)
        data[x+'_polynomial2']=data[x]*data[x]
        data[x+'_polynomial3']=data[x]*data[x]*data[x]
        if (xprev!=0):
            data[x+'_polynomial_x_2']=data[x]*data[xprev]
        if (xprev2!=0):
            data[x+'_polynomial_x_3']=data[x]*data[xprev2]*data[xprev]
        i=i+1
        xprev=x
        xprev2=xprev
    
    data['return']=((brent.shift(-5).Rate/brent.shift(-1).Rate)-1)>0
    data=data[8:].dropna(1)
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:-1,:-1], data.iloc[:-1,-1], test_size=0.5)
    gbc=SVC (kernel='rbf',probability=True,C=1)
    gbc.fit(x_train,y_train)
    #min_max_scaler=MinMaxScaler()
    #mms=min_max_scaler.fit(list(max(a) for a in gbc.predict_proba(x_train)))
    pr=list(max(a) for a in gbc.predict_proba(x_test))
    Y=pd.DataFrame()
    Y['actual']=y_test
    Y['predicted']=gbc.predict(x_test)
    Y['P']=mms.transform(list(max(a) for a in gbc.predict_proba(x_test)))
    Y_filtered=Y[Y.P>threshold]
    cm=confusion_matrix(Y_filtered.actual,Y_filtered.predicted)
    return [gbc.score(x_test,y_test,pr>threshold),cm,'Prediction of UP is %s; P = %s' %(gbc.predict(data.iloc[-1:,:-1])[0],
     list((max(x)) for x in gbc.predict_proba(data.iloc[-1:,:-1]))[0]
     ),brent]


# In[45]:
loaddata("OFDP/FUTURE_COT_190","CURRFX/USDRUB")
# In[46]:

#a=svm_classify("CFTC/GC_FO_ALL","OFDP/FUTURE_GC2",-1000)
#a=svm_classify("CFTC/BZC_FO_ALL","SCF/ICE_B1_FW",0.55)
a=svm_classify(0.55)
#plt.matshow(a[1])



# In[45]:

plot_confusion_matrix(a[1])


# In[ ]:



