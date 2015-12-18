# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 14:49:44 2015

@author: razkevich
"""
import pandas as pd
#igt=pd.read_json('d:\\test.json')
#ig=igt.iloc[:,1:2]
#ig=ig.applymap(lambda x:','.join(x))
#ig_t=list(((" ".join(x))+' '+(" ".join(list(y.replace(' ','_') for y in x)))) for x in ig['ingredients'])
#ig_t=transformer.transform(vectorizer.transform(ig.iloc[:,0].tolist()))


Y_pred1=classifier.predict(transformer.transform(vectorizer.transform(ig_t)))
igt['cuisine']=Y_pred1


igt.drop('ingredients',axis=1).to_csv('D:\\OUT2.CSV',index=False)