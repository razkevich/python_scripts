# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 12:19:45 2015

@author: razkevich
"""

from nolearn.dbn import DBN

net = DBN(
    [784, 300, 10],
    learn_rates=0.3,
    learn_rate_decays=0.9,
    epochs=10,
    verbose=1,
    )
    
    
    
import csv
import numpy as np

with open('D:\\train.csv', 'rb') as f:
	data = list(csv.reader(f))
	
train_data = np.array(data[1:])
labels = train_data[:, 0].astype('float')
train_data = train_data[:, 1:].astype('float') / 255.0

net.fit(train_data, labels)



with open('D:\\test.csv', 'rb') as f:
	data = list(csv.reader(f))

test_data = np.array(data[1:]).astype('float') / 255.0
preds = net.predict(test_data)

with open('D:\\submission.csv', 'wb') as f:
    fieldnames = ['"ImageId"', '"Label"']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    i = 1
    for elem in preds:
        writer.writerow({'"ImageId"': i, '"Label"': str(elem)})
        i += 1