# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 22:17:03 2019

@author: flacas_b
"""

import SKOYENdata_new as SKOYENdata
import numpy as np
import string
import matplotlib.pyplot as plt

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import CuDNNLSTM, LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

import pickle

def prepareData(sequence, seq_length, X_char_to_int, Y_char_to_int):
    
    n_chars = len(sequence)
    if Y_char_to_int == 0: #predicting time
        dataX = []
        dataY = []
        for i in range(n_chars-seq_length):
            seq_in = sequence[i:i + (seq_length)]
            seq_out = sequence[i + (seq_length)]
            dataX.append([X_char_to_int[char] for char in seq_in])
            dataY.append([X_char_to_int[seq_out]])   
        dataX = np.array(dataX)
        dataY = np.array(dataY)
    else:
        dataX = []
        dataY = []
        for i in range(n_chars-seq_length):
            seq_in = sequence[i:i + (seq_length)]
            seq_out = sequence[i + (seq_length)]
            dataX.append([X_char_to_int[char] for char in seq_in])
            dataY.append([Y_char_to_int[seq_out[0]]])   
        dataX = np.array(dataX)
        dataY = np.array(dataY)
    
    return dataX, dataY

def getData(apts_data, seq_length, training_apts, validation_apt, X_chars, X_char_to_int, Y_chars, Y_char_to_int):
    
    #training data
    Xtrain = np.empty((0, seq_length), dtype=np.int)
    Ytrain = np.empty((0,1), dtype=np.int)
    for apt in training_apts:
        x_temp, y_temp  = prepareData(apts_data[apt], seq_length, X_char_to_int, Y_char_to_int)
        Xtrain = np.concatenate((Xtrain, x_temp), axis=0)
        Ytrain = np.concatenate((Ytrain, y_temp), axis=0)
    
    #validation data
    Xval, Yval =  prepareData(apts_data[validation_apt], seq_length, X_char_to_int, Y_char_to_int)
    
    #transform to categorical data
    train_x_classes = len(X_chars)
    train_y_classes = train_x_classes if Y_chars==0 else len(Y_chars)
    Xtrain = np_utils.to_categorical(Xtrain, num_classes = train_x_classes)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(Ytrain), Ytrain[:,0])
    Ytrain = np_utils.to_categorical(Ytrain, num_classes = train_y_classes)
    
    return Xtrain, Ytrain, Xval, Yval, class_weights

def createSourceModel(input_shape, output_shape):
    
    model = Sequential()
    #model.add(Dense(32, input_shape=(input_shape[1], input_shape[2])))
    model.add(CuDNNLSTM(64, input_shape=(input_shape[1], input_shape[2])))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))
    
    opt = 'adam'       
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, verbose=0, mode='auto')
    bestModel = ModelCheckpoint('bestTrain.hdf5', save_best_only=True, monitor='val_acc', mode='max')
    callbacks_list = [earlystop, bestModel]
    return model, callbacks_list

def createTargetModel(input_shape, output_shape, weights):
    
    
    model = Sequential()
    #model.add(Dense(32, input_shape=(input_shape[1], input_shape[2])))
    model.add(CuDNNLSTM(64, input_shape=(input_shape[1], input_shape[2])))
    model.add(Dropout(0.5))
    model.add(Dense(output_shape, activation='softmax'))
    
    opt = Adam(lr=0.001)
    model.set_weights(weights)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, verbose=0, mode='auto')
    bestModel = ModelCheckpoint('bestRetrain.hdf5', save_best_only=True, monitor='val_acc', mode='max')
    callbacks_list = [earlystop, bestModel]
    return model, callbacks_list
    

def doKmeans(data):
	
    sensors = sorted(list(set(data[:,0])))
            
    ##find the most common durations
    max_clusters = 8
    cs = range(2, max_clusters+1)

    sensors_periods = {}
    sensors_centroids = {}
    cluster_sizes = {}
    new_data = np.empty((0, data.shape[1]+1), dtype=np.int)
    opt_clusters = [5,4,4,4,4,4,4,4,4,4,4,4,4]
    k_models = []
    scalars  = []
    for oc, s in enumerate(sensors):
        
        indices = np.where(data[:,0]==s)
        sensor_data = data[indices]
        X = sensor_data[:,-2:]
        
        #scale data
        Xscaler = StandardScaler()
        X = Xscaler.fit_transform(X)
        
        
        max_sil_score = 0
        best_cs = -1
        sse = []
        #for i, n in enumerate(cs):
        for i, n in enumerate([opt_clusters[oc]]):
            # cluster the samples
            if len(X)<n:
                k = KMeans(n_clusters=1, random_state=42 ) 
                best_cs = 1
            else:
                k = KMeans(n_clusters=n, random_state=42 )
                best_cs = n                
            k.fit(X)

            sse.append(k.inertia_)
            
        
        # plt.figure(plt.gcf().number + 1)
        # plt.plot(cs[:len(sse)], sse)
        # plt.xlabel("Number of cluster")
        # plt.ylabel("SSD")
        # plt.savefig("SSE_" + str(apt) + "_Sensor_" + str(s))
        
        k = KMeans(n_clusters=best_cs)
        k_models.append(k)
        scalars.append(Xscaler)
        results = k.fit_predict(X)	
        centroids = k.cluster_centers_
        labels = np.reshape(k.labels_, (k.labels_.shape[0],1))
        cluster_sizes[s] = {}
        outliers = -1
        for c in range(n):
            cluster_sizes[s][c] = sum(k.labels_ == c)
            if sum(k.labels_ == c) <0.05*len(k.labels_):
                outliers = c
        sensors_centroids[s] = centroids
        
        #X_original = Xscaler.inverse_transform(X)
        #plot clustering graphs
        # plt.figure(plt.gcf().number + 1)
        

        # plt.scatter(np.array(X_original[:,0])/3600, np.array(X_original[:,1])/60, c=k.labels_.astype(np.float), edgecolor='k')
        # #plt.scatter(np.array(k.cluster_centers_[:,0])/3600 ,np.array(k.cluster_centers_[:,1])/60, color='black', marker='x', linewidths=4)  
        # plt.xlabel('Time of day (hours)')
        # plt.ylabel('Elapsed time (minutes)')
        # plt.title('K-means clustering of samples')   
        # plt.savefig(str(apt) + '_Sensor_' + str(s) + "_best_k_" + str(best_cs))
        # saveVariables(str(apt) + '_Sensor_' + str(s) + "_best_k_" + str(best_cs), [X, k.labels_.astype(np.float)])
        
        sensor_data = np.append(sensor_data, labels, axis=1)
        
        if outliers != -1:
            ind_outliers = np.where(sensor_data[:,-1]==c)
            np.delete(sensor_data, ind_outliers, axis=0)

        new_data = np.concatenate((new_data, sensor_data), axis=0)
        
        
    new_data = new_data[new_data[:,0].argsort()]    
    return k_models, scalars

def getSequence(data):
    #create a dictionary with every sensor id assigned to an alphabet symbol
    alphabet = list(string.ascii_lowercase)
    sensors = {}
    sensors_IDs = sorted(list(set(data[:,1])))
    for s, a in zip(sensors_IDs, alphabet):
        sensors[s] = a
            
    sequence = []
    for d in data:
        sensor   = d[1]
        msg      = d[2]
        label    = d[-1]
        event = sensors[sensor] if msg==0 else sensors[sensor].upper()
        event += str(label)
        sequence.append(event)
        
    return sequence

def getChars(apts_data):

    #training aparts datasets
    chars = []
    for k,v in apts_data.items():
        chars.extend(list(set(v))) 
    X_chars = sorted(chars)
    X_char_to_int = dict((c, i) for i, c in enumerate(X_chars))  
    Ytrain_chars = list(set([c[0] for c in X_chars]))
    Ytrain_char_to_int = dict((c, i) for i, c in enumerate(Ytrain_chars))

    return X_chars, X_char_to_int, Ytrain_chars, Ytrain_char_to_int

############# MAIN
apts    = [1,2,3,4,5,6,7,8]

#parameters
seq_length = 10
nEpochs    = 200
bS         = 512

predict_time = True

data = np.empty((0, 5), dtype=np.int)
opt      = 'all' 
for apt in apts:   
    data_tmp = SKOYENdata.getDataDuration(str(apt) + ".csv", opt, fix=True, error=2, mapping=True) #id, msg, weekday, time in seconds, durations
    data     = np.concatenate((data, data_tmp), axis=0)
k_models, scalars = doKmeans(data)

#keep data of each apt
original_apts_data = {}
for apt in apts:
    data = SKOYENdata.getDataDuration(str(apt) + ".csv", opt, fix=True, error=2, mapping=True) #id, msg, weekday, time in seconds, durations
    
    IDs = np.array(range(len(data)))
    IDs = IDs.reshape((IDs.shape[0],1))
    data = np.append(IDs, data, axis=1) #line_id, id, msg, weekday, time in seconds, durations
    
    sensors = sorted(list(set(data[:,1])))
    
    new_data = np.empty((0, data.shape[1]+1), dtype=np.int)
    for oc, s in enumerate(sensors):
        indices = np.where(data[:,1]==s)
        sensor_data = data[indices]
        X = sensor_data[:,-2:]
        if len(X)>=len(k_models[oc].cluster_centers_):
            Xscaler = StandardScaler()
            X = scalars[oc].fit_transform(X)
            results = k_models[oc].fit_predict(X)	
            labels = np.reshape(k_models[oc].labels_, (k_models[oc].labels_.shape[0],1))
            sensor_data = np.append(sensor_data, labels, axis=1)
            new_data = np.concatenate((new_data, sensor_data), axis=0)
         
    new_data = new_data[new_data[:,0].argsort()] 
        
    sequence = getSequence(new_data)
    original_apts_data[apt] = sequence
      
dataset_sizes = [200, 400, 600, 800, 1000, 1300, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 7500, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 50000, 60000, 80000, 100000, 120000, 160000, 200000] 
n_test_events = 3000
X_chars, X_char_to_int, Y_chars, Y_char_to_int = getChars(original_apts_data)

#cross-validation loop
for k in range(1):
    
    training_apts = apts[:k] + apts[k+1:]
    validation_apt =  apts[k]
    
    allAccs = []
    apts_data = original_apts_data.copy()           
    apts_data[validation_apt] = apts_data[validation_apt][:100] + apts_data[validation_apt][-n_test_events:]  
    for ds in dataset_sizes:
        
        if predict_time == True:
            Xtrain, Ytrain, Xval, Yval, class_weights = getData(apts_data, seq_length, training_apts, validation_apt, X_chars, X_char_to_int, 0, 0)
        else:
            Xtrain, Ytrain, Xval, Yval, class_weights = getData(apts_data, seq_length, training_apts, validation_apt, X_chars, X_char_to_int, Y_chars, Y_char_to_int)
        
        train_model, callbacks_list   = createSourceModel(Xtrain.shape, Ytrain.shape[1])    
        Xtrain_train, Xtrain_val, Ytrain_train, Ytrain_val = train_test_split(Xtrain, Ytrain, test_size=0.2, shuffle=True)
        if len(Xtrain_train)>ds:
            train_history = train_model.fit(Xtrain_train[:ds,:], Ytrain_train[:ds], epochs=nEpochs, batch_size=bS, callbacks=callbacks_list, validation_data=(Xtrain_val,Ytrain_val))

            train_model.load_weights(filepath = 'bestTrain.hdf5')
            
            Xval_train, Yval_train = Xval[:-n_test_events,:], Yval[:-n_test_events]
            Xval_val, Yval_val     = Xval[-n_test_events:,:], Yval[-n_test_events:]
            
            test_x_classes = len(X_chars)
            test_y_classes = test_x_classes if (predict_time == True) else len(Y_chars)
            Xval_val       = np_utils.to_categorical(Xval_val, num_classes = test_x_classes)
            Yval_val       = np_utils.to_categorical(Yval_val, num_classes = test_y_classes)
            
            accsK = []
            for n_i in range(3):   

                Xval_train, Yval_train = shuffle(Xval_train, Yval_train) 
                Xval_train_cat = np_utils.to_categorical(Xval_train, num_classes = test_x_classes)
                Yval_train_cat = np_utils.to_categorical(Yval_train, num_classes = test_y_classes)
                
                retrain_model, callbacks_list = createTargetModel(Xval_train_cat.shape, Yval_train_cat.shape[1], train_model.get_weights())

                val_train_history = retrain_model.fit(Xval_train_cat, Yval_train_cat, epochs=nEpochs, batch_size=bS, callbacks=callbacks_list, validation_data=(Xval_val,Yval_val))

                retrain_model.load_weights(filepath = 'bestRetrain.hdf5')

                accTe = retrain_model.evaluate(x=Xval_val, y=Yval_val, batch_size=bS)
                accsK.append(accTe[-1])

            allAccs.append(np.mean(accsK))

    saveVariables("allAccs" + str(validation_apt), [allAccs, dataset_sizes])        
    