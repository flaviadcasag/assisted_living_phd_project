# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 10:24:49 2018

@author: flacas_b
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pickle
import time

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, CuDNNLSTM, Dropout
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import SKOYENdata_new as SKOYENdata

import TreePPM
import SPEED
import LeZi

##############AUX FUNCTIONS SPEED AND ALZ
    
def computeAcc(sequence, optimalWindow, tree, algo, testing, sensorsID, time = 0):
    '''
    computeAcc returns the accuracies everytime a new event is predicted
    accuracy if final accuracy and accuracies at every event
    ''' 
    correctPred = 0
    totalPred   = 0
    total       = []
    accuracies  = []    
    if testing == False:
        for j in range(len(sequence)-optimalWindow):
            window = sequence[j:optimalWindow+j]
            nextEvent = sequence[optimalWindow+j]
            predictedEvent = tree.predictNextEvent(window)
            if predictedEvent == nextEvent:
                correctPred = correctPred + 1
            totalPred = totalPred + 1
            total.append(totalPred)
            accuracies.append(correctPred/totalPred*100)       
    else:      
        for j in range(len(sequence)-optimalWindow):
            window = sequence[j:optimalWindow+j]
            nextEvent = sequence[optimalWindow+j]
            predictedEvent = tree.predictNextEvent(window)
            if predictedEvent == nextEvent:
                correctPred = correctPred + 1
            totalPred = totalPred + 1
            total.append(totalPred)
            accuracies.append(correctPred/totalPred*100)
    accuracy = correctPred/totalPred*100
    return accuracy, accuracies

def findOptWindow(sequence, tree, maxEpisodeLength, algo, time = 0):
    '''
    returns the optimal number of events to predict from 
    it computes the accuracy for predicting from last x events, where 1<x<maxEpisodeLength
    '''  
    accuracies = []
    for w in range(1,maxEpisodeLength+1):
        accuracy, other  = computeAcc(sequence, w, tree, algo, False, 0)
        accuracies.append(accuracy) 
    optimalWindow   = accuracies.index(max(accuracies)) + 1
    return optimalWindow, accuracies

########################## RNN
    

def RNNmodel(nNeurons, nEpochs, bS,  Xtrain, Ytrain, Xval, Yval, Xtest, Ytest):
             
    # create and fit the model
    model = Sequential()
    model.add(CuDNNLSTM(nNeurons, return_sequences=False,input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
    model.add(Dropout(0.5))
    model.add(Dense(Ytrain.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')
    bestModel = ModelCheckpoint('best.hdf5', save_best_only=True, monitor='val_acc', save_weights_only=True, mode='max')    
    callbacks_list = [earlystop, bestModel]
    history = model.fit(Xtrain, Ytrain, epochs=nEpochs, batch_size=bS, callbacks=callbacks_list, validation_data=(Xval,Yval))
    testResults = model.evaluate(x=Xtest, y=Ytest, batch_size=bS)
    
    accT   = history.history['acc'][-1]
    accV   = history.history['val_acc'][-1]
    accTe  = testResults[-1]

    return accT, accV, accTe, history
    
############## SPEED accuracy
def doSpeed(trainSpeed, valSpeed, testSpeed, sensorsID):
    contexts, maxEpisodeLength, maxEpisode = SPEED.SPEED(trainSpeed)
    tree = TreePPM.Tree(TreePPM.Node(None,0,0))
    tree = tree.createTree(contexts)
    optWindow, accuraciesVal = findOptWindow(valSpeed, tree, maxEpisodeLength, 'SPEED')
    accuracyTest, accuraciesTest = computeAcc(testSpeed, optWindow, tree, 'SPEED', True, sensorsID)
    return tree, optWindow, accuraciesVal, accuracyTest, accuraciesTest

############## ALZ accuracy
def doALZ(trainALZ, valALZ, testALZ, sensorsID):
    contexts, maxEpisodeLength = LeZi.LeZi(trainALZ) 
    tree = TreePPM.Tree(TreePPM.Node(None,0,0))
    tree = tree.createTree(contexts)
    optWindow, accuraciesVal = findOptWindow(valALZ, tree, maxEpisodeLength, 'ALZ')
    accuracyTest, accuraciesTest = computeAcc(testALZ, optWindow, tree, 'ALZ',  True, sensorsID)
    return tree, optWindow, accuraciesVal, accuracyTest, accuraciesTest

def splitKs(seq,test_size,val_size):
    
    ks = {0: {'train': seq[:-(test_size+val_size)], 'val': seq[-(test_size+val_size):-test_size], 'test': seq[-test_size:]},
          1: {'train': seq[test_size+val_size:], 'val': seq[:test_size], 'test': seq[test_size:test_size+val_size]},
          2: {'train': seq[test_size:-val_size], 'val': seq[-test_size:], 'test': seq[:test_size]},
    }   
    return ks
    
def splitKs_LSTM(dataX, dataY, nEvents, n_classes, val_size, test_size):

    try:
        k12_X, k3_X, k12_Y, k3_Y = train_test_split(dataX, dataY, test_size=test_size, stratify=dataY)
    except:
        k12_X, k3_X, k12_Y, k3_Y = train_test_split(dataX, dataY, test_size=test_size)
    
    try:
        k1_X, k2_X, k1_Y, k2_Y = train_test_split(k12_X, k12_Y, test_size=val_size, stratify=k12_Y)
    except:
        k1_X, k2_X, k1_Y, k2_Y = train_test_split(k12_X, k12_Y, test_size=val_size)

    k1_X = np_utils.to_categorical(k1_X, num_classes = n_classes)
    k1_Y = np_utils.to_categorical(k1_Y, num_classes = n_classes)
    k2_X = np_utils.to_categorical(k2_X, num_classes = n_classes)
    k2_Y = np_utils.to_categorical(k2_Y, num_classes = n_classes)
    k3_X = np_utils.to_categorical(k3_X, num_classes = n_classes)
    k3_Y = np_utils.to_categorical(k3_Y, num_classes = n_classes)

    ks = {'Xtrain': k1_X, 'Xval': k2_X, 'Xtest': k3_X, 'Ytrain': k1_Y, 'Yval': k2_Y, 'Ytest': k3_Y}

    if nEvents==1:
        mat = ks['Xtrain']
        ks['Xtrain'] = np.reshape(mat, (mat.shape[0], 1, mat.shape[1]))
        mat = ks['Xval']
        ks['Xval'] = np.reshape(mat, (mat.shape[0], 1, mat.shape[1]))
        mat = ks['Xtest']
        ks['Xtest'] = np.reshape(mat, (mat.shape[0], 1, mat.shape[1]))
    
    return ks
    
def createDataset(sequence, nEvents):

    #create mapping of unique chars to integers
    chars = sorted(list(set(sequence)))
    char_to_int = dict((c, i) for i, c in enumerate(chars)) 
    
    n_chars = len(sequence)
    # prepare the dataset of input to output pairs encoded as integers
    seq_length = nEvents
    ### prepare the dataset of input to output pairs encoded as integers
    dataX = []
    dataY = []
    for i in range(n_chars-seq_length):
    	seq_in = sequence[i:i + (seq_length)]
    	seq_out = sequence[i + (seq_length)]
    	dataX.append([char_to_int[char] for char in seq_in])
    	dataY.append([char_to_int[seq_out]])   
    dataX = np.array(dataX)
    dataY = np.array(dataY)
    
    return dataX, dataY, len(chars)
    
########################## MAIN

filename = "apt1.csv"
opt      = "all"
fix      = True
error    = 2

speedSeq, sensorsIDspeed = SKOYENdata.getSequence(filename, opt, fix, error)
alzSeq, sensorsIDalz     = SKOYENdata.getLeZiDataset(filename, opt, fix, error)

val_size  = 3000
test_size = 3000

k = 3
ks_speed = splitKs(speedSeq, test_size, val_size)
ks_alz   = splitKs(alzSeq, test_size, val_size)

plt.figure(1)

#SPEED
acc = []
windows = []
for n in range(k):     
    tree, optWindow, accuraciesVal, accuracyTest, accuraciesTest = doSpeed(ks_speed[n]['train'], ks_speed[n]['val'], ks_speed[n]['test'], sensorsIDspeed)
    acc.append(accuraciesVal)
    windows.append(optWindow)

min_events = 100000
for ac in acc:
    if len(ac)<min_events:
        min_events = len(ac)
new_acc = []
for ac in acc:
    new_acc.append(ac[:min_events])      

acc = np.array(new_acc)
acc = np.mean(acc, axis=0)
if len(acc)<30:
    plt.plot(list(range(1,len(acc)+1)),acc,label='SPEED')
else:
    plt.plot(list(range(1,31)),acc[:30],label='SPEED')
saveVariables('SPEED', [windows, acc])

#ALZ
acc = []
windows = []
for n in range(k):
    #ALZ
    tree, optWindow, accuraciesVal, accuracyTest, accuraciesTest = doALZ(ks_alz[n]['train'], ks_alz[n]['val'], ks_alz[n]['test'], sensorsIDalz)
    acc.append(accuraciesVal)
    windows.append(optWindow)
min_events = 100000
for ac in acc:
    if len(ac)<min_events:
        min_events = len(ac)
new_acc = []
for ac in acc:
    new_acc.append(ac[:min_events])      

acc = np.array(new_acc)
acc = np.mean(acc, axis=0)        
if len(acc)<31:
   plt.plot(list(range(1,len(acc)+1)),acc,'.-',label='ALZ')
else:
   plt.plot(list(range(1,31)),acc[:30],'.-',label='ALZ')
saveVariables('ALZ', [windows, acc])


#RNN SPEED
allVAccs = []
allTeAccs = []
nE       = list(range(1,31))
nNeurons = 64
nEpochs  = 200
bS       = 512
for nEvents in nE:
    dataX_speed, dataY_speed, nClasses = createDataset(speedSeq, nEvents)
    accsT  = []
    accsV  = []
    accsTe = []
    for n in range(k):
        ks = splitKs_LSTM(dataX_speed, dataY_speed, nEvents, nClasses, val_size, test_size)
        accT, accV, accTe, history, = RNNmodel(nNeurons, nEpochs, bS, ks['Xtrain'], ks['Ytrain'], ks['Xval'], ks['Yval'], ks['Xtest'], ks['Ytest'])        
        accsT.append(accT)
        accsV.append(accV)
        accsTe.append(accTe)
    meanAccsT  = sum(accsT)/k
    meanAccsV  = sum(accsV)/k
    meanAccsTe = sum(accsTe)/k
    allVAccs.append(meanAccsV)
    allTeAccs.append(meanAccsTe)

plt.plot(nE,np.array(allVAccs)*100,'.-', label='LSTM SPEED') 
saveVariables('LSTM_SPEED', [allVAccs,allTeAccs])

#RNN ALZ
allVAccs = []
allTeAccs = []
nE       = list(range(1,31))
nNeurons = 64
nEpochs  = 50
bS       = 512
for nEvents in nE:
    dataX_alz, dataY_alz, nClasses = createDataset(alzSeq, nEvents)    
    accsT  = []
    accsV  = []
    accsTe = []
    for n in range(k):
        ks = splitKs_LSTM(dataX_alz, dataY_alz, nEvents, nClasses, val_size, test_size)
        accT, accV, accTe, history, = RNNmodel(nNeurons, nEpochs, bS, ks['Xtrain'], ks['Ytrain'], ks['Xval'], ks['Yval'], ks['Xtest'], ks['Ytest'])
        accsT.append(accT)
        accsV.append(accV)
        accsTe.append(accTe)
    meanAccsT  = sum(accsT)/k
    meanAccsV  = sum(accsV)/k
    meanAccsTe = sum(accsTe)/k
    allVAccs.append(meanAccsV)
    allTeAccs.append(meanAccsTe)

plt.plot(nE,np.array(allVAccs)*100,'.-',label='LSTM ALZ') 
saveVariables('LSTM_ALZ', [allVAccs,allTeAccs])

#plt.title("Accuracy vs Memory Length (All sensors)")
plt.legend(loc='upper right')
plt.xlabel('Memory Length')
plt.ylabel('Accuracy (%)')
axes = plt.gca()
axes.set_ylim([0,100])
plt.savefig("AccVSmemory.png") 





