# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 11:32:08 2018

@author: flacas_b
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 19:37:20 2018

@author: flacas_b
"""

import numpy as np
import pandas as pd
import datetime 
import string
import aux_functions as aux
import math
from itertools import product
import random

##########################################################################################
##################################### DATA PROCESSING ####################################
##########################################################################################
        
def getEventsDataset(filename, opt, fix=True, error=0, mapping=False):
    '''
    getEventsDataset returns an np.array with the csv file information
    opt defines which (group of) sensors that will be taken in account
    '''
    #Read .csv file
    df      = pd.read_csv(filename,names=['timestamp', 'sensorID', 'sensorMsg'],sep=';',skiprows=1)
    
    if opt == 'all':
        df = df[df.sensorID.isin([1,2,3,4,5,32,6,9,10,11,12,31,19,20,21,22,23,24,25,26,27,29,30])]
    elif opt == 'pirs': #get only PIRs
        df = df[df.sensorID.isin([1,2,3,4,5,6,32])]
    elif opt == 'pirsMinusTwo': #get only PIRs
        df = df[df.sensorID.isin([1,2,4,5,6])]
    elif opt == 'pirsMags': #get PIRs + magnetics
        df = df[df.sensorID.isin([1,2,3,4,5,6,32,9,10,11,12,31])]
    elif opt == 'pirsMagsMinusTwo': #get PIRs + magnetics
        df = df[df.sensorID.isin([1,2,4,5,6,9,10,11,12,31])]
    elif opt == 'pirsPowers': #get PIRs + power
        df = df[df.sensorID.isin([1,2,3,4,5,6,32,19,20,21,22,23,24,25,26,27,29,30])]
    elif opt == 'pirsPowersMinusTwo': #get PIRs + power
        df = df[df.sensorID.isin([1,2,4,5,6,19,20,21,22,23,24,25,26,27,29,30])]
    elif opt == 'powers': #get power
        df = df[df.sensorID.isin([19,20,21,22,23,24,25,26,27,29,30])]
    elif opt == 'mags': #get magnetics
        df = df[df.sensorID.isin([9,10,11,12,31])]
    elif opt == 'magsPowers': # get mag + power
        df = df[df.sensorID.isin([9,10,11,12,31,19,20,21,22,23,24,25,26,27,29,30])]
    elif opt == 'allMinusTwo':
        df = df[df.sensorID.isin([1,2,4,5,6,9,10,11,12,31,19,20,21,22,23,24,25,26,27,29,30])] 

    #separate date and time from the same column
    df['date'], df['time'] = zip(*df['timestamp'].map(lambda x: x.split(' ')))
    df = df.drop('timestamp',1) 
    
    if mapping == True:
        df = mapSensors(df)
    
    #get data in nparray
    ds = df.values
    
    #read values and put in dataset variable as sensorID, sensorMSG, weekDay 0-6, hour in seconds
    dataset = []
    for i in ds:
        dataset.append([i[0], i[1], getWeekDay(i[2]), getSeconds(i[3])])
    
    #remove consecutive same sensors events
    dataset = removeConsecutiveEqualEvents(dataset)
    
    #remove noise
    if fix:
        dataset = fixDataset(dataset, opt, error)
    
    return np.array(dataset)

def mapSensors(df):
    
    sensor2sensor = { 12: 50,
                    29: 50,
                    30: 50,
                    31: 50,
                    21: 52,
                    24: 52,                    
    }
    delete = [11,19,20,22,25,26,27]
    df = df[~df['sensorID'].isin(delete)]
    df['sensorID'] = df['sensorID'].replace(sensor2sensor)
    
    return df
    
def removeConsecutiveEqualEvents(dataset):
    '''
    returns a list without consecutive same events, receives a dataset in list form
    '''
    newDataset = []
    newDataset.append(dataset[0])
    for i in range(1,len(dataset)):
        if dataset[i][0] <= 6 or dataset[i][0]==32:
            if dataset[i][0]!=dataset[i-1][0]:
                newDataset.append(dataset[i])
        else: #mag or power
            newDataset.append(dataset[i])
    
    return newDataset

def cleanRawData(dataset):
    
    possible = {1: [2,6], 2: [4,1,20,11,32,3], 32: [2,3], 3: [32,2], 4: [2,6,5,23,27,9], 5: [4,21,31], 6: [10,1,4], 9: [4,9], 10: [6,10], 11: [2,11], 20: [2,20], 21: [31,5,21], 23: [4,27,9,23], 27: [4,23,9,27], 31: [31,5,21]}
   
    newDataset = []
    wrongData = {}
    deleteIndex = []
    incorrect  = 0
    for i in range(len(dataset)-2):
        correct = True
        if not dataset[i+1][0] in possible.get(dataset[i][0]):
            correct = False
            incorrect = incorrect + 1
            wrongSeq = tuple((dataset[i][0], dataset[i+1][0]))
            if wrongData.get(wrongSeq) == None:
                wrongData[wrongSeq] = 1
            else:
                wrongData[wrongSeq] = wrongData.get(wrongSeq) + 1  
        if not correct:
            deleteIndex.append(i)
    
    return newDataset

def fixDataset(dataset, opt, error=0):
    '''
    returns a dataset in which new events are added to make the sequence correct, the time added is in between
    the consecutive events
    error == 2 means we are fixing both errors (place and missing event), otherwise only place error
    '''
    newDataset = []
    
    missingCounter = 0
    placeCounter   = 0
    
    if opt == 'all':
        for i in range(len(dataset)-1):
            currentE = dataset[i][0]        
            nextE    = dataset[i+1][0]
            newDataset.append(dataset[i])
            if nextE == 4 and currentE == 32:                                    #(32, 4): 180
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([2, 1, dataset[i][2], time])
            elif nextE == 4 and currentE == 31:                                  #(31, 4): 40
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([5, 1, dataset[i][2], time])
            elif nextE == 1 and currentE == 32:                                  #(32, 1): 202
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([2, 1, dataset[i][2], time])
            elif nextE == 20 and currentE == 32:                                  #(32, 20): 16
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([2, 1, dataset[i][2], time])
            elif nextE == 1 and currentE == 32:                                  #(32, 1): 202
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([2, 1, dataset[i][2], time])                
            elif nextE == 4 and currentE == 21:                                  #(21, 4): 30
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([2, 1, dataset[i][2], time])                 
            elif nextE == 32 and currentE == 1:                                  #(1, 32): 31
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([2, 1, dataset[i][2], time])                  
            elif nextE == 9 and currentE == 6:                                  #(6, 9): 5
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time]) 
            elif nextE == 10 and currentE == 9:                                  #(9, 10): 4
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time])  
                newDataset.append([6, 1, dataset[i][2], time]) 
            elif nextE == 4 and currentE == 10:                                  #(10, 4): 28
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([6, 1, dataset[i][2], time])                  
            elif nextE == 9 and currentE == 5:                                  #(5, 9): 2
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time])   
            elif nextE == 31 and currentE == 4:                                  #(4, 31): 21
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([5, 1, dataset[i][2], time]) 
            elif nextE == 27 and currentE == 5:                                  #(5, 27): 7
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time]) 
            elif nextE == 2 and currentE == 5:                                  #(5, 2): 18
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time]) 
            elif nextE == 23 and currentE == 2:                                  #(2, 23): 7
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time]) 
            elif nextE == 32 and currentE == 23:                                  #(23, 32): 5
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([2, 1, dataset[i][2], time]) 
            elif nextE == 11 and currentE == 1:                                  #(1, 11): 6
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([2, 1, dataset[i][2], time]) 
            elif nextE == 1 and currentE == 11:                                  #(11, 1): 13
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([2, 1, dataset[i][2], time]) 
            elif nextE == 6 and currentE == 5:                                  #((5, 6): 8
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time]) 
            elif nextE == 10 and currentE == 4:                                  #(4, 10): 10
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([6, 1, dataset[i][2], time]) 
            elif nextE == 5 and currentE == 6:                                  #(6, 5): 22
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time]) 
            elif nextE == 6 and currentE == 23:                                  #(23, 6): 10
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time]) 
            elif nextE == 23 and currentE == 32:                                  #(32, 23): 5
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time])                 
            elif nextE == 32 and currentE == 4:                                  #(4, 32): 10
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([2, 1, dataset[i][2], time])                 
            elif nextE == 2 and currentE == 9:                                  #(9, 2): 7
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time])
            
    elif opt == 'pirs': #get only PIRs
        for i in range(len(dataset)-1):
            currentE = dataset[i][0]        
            nextE    = dataset[i+1][0]
            newDataset.append(dataset[i])
            if nextE == 4 and currentE == 32:                                    #(32, 4): 187
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([2, 1, dataset[i][2], time])          
            if nextE == 1 and currentE == 3:                                    #(3, 1): 26
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([2, 1, dataset[i][2], time]) 
            if nextE == 1 and currentE == 32:                                    #(32, 1): 210
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([2, 1, dataset[i][2], time]) 
            if nextE == 32 and currentE == 1:                                    #(1, 32): 35
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([2, 1, dataset[i][2], time])                 
            if nextE == 2 and currentE == 5:                                    #(5, 2): 23
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time])
            if nextE == 6 and currentE == 5:                                    #(5, 6): 10
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time])
            if nextE == 5 and currentE == 6:                                    #(6, 5): 22
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time])                
            if nextE == 4 and currentE == 3:                                    #(3, 4): 4
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([2, 1, dataset[i][2], time])


    elif opt == 'pirsMinusTwo': #get only PIRs
        for i in range(len(dataset)-1):
            currentE = dataset[i][0]        
            nextE    = dataset[i+1][0]
            newDataset.append(dataset[i])
            if nextE == 2 and currentE == 5:                                    #(5, 2): 24
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time])
                missingCounter = missingCounter + 1
            if nextE == 6 and currentE == 5:                                    #(5, 6): 10
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time])
                missingCounter = missingCounter + 1                
            if nextE == 5 and currentE == 6:                                    #(6, 5): 22
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time]) 
                missingCounter = missingCounter + 1                
            if nextE == 5 and currentE == 2:                                    #(2, 5): 4
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time])  
                missingCounter = missingCounter + 1 
                
    elif opt == 'pirsMags': #get PIRs + magnetics
        df = df[df.sensorID.isin([1,2,3,4,5,6,32,9,10,11,12,31])]
    elif opt == 'pirsMagsMinusTwo': #get PIRs + magnetics
        df = df[df.sensorID.isin([1,2,4,5,6,9,10,11,12,31])]
    elif opt == 'pirsPowers': #get PIRs + power
        df = df[df.sensorID.isin([1,2,3,4,5,6,32,19,20,21,22,23,24,25,26,27,29,30])]
    elif opt == 'pirsPowersMinusTwo': #get PIRs + power
        df = df[df.sensorID.isin([1,2,4,5,6,19,20,21,22,23,24,25,26,27,29,30])]
    elif opt == 'powers': #get power
        df = df[df.sensorID.isin([19,20,21,22,23,24,25,26,27,29,30])]
    elif opt == 'mags': #get magnetics
        df = df[df.sensorID.isin([9,10,11,12,31])]
    elif opt == 'magsPowers': # get mag + power
        df = df[df.sensorID.isin([9,10,11,12,31,19,20,21,22,23,24,25,26,27,29,30])]
    
    elif opt == 'allMinusTwo':
        for i in range(len(dataset)-1):
            currentE = dataset[i][0]        
            nextE    = dataset[i+1][0]
            newDataset.append(dataset[i])
            if nextE == 4 and currentE == 31:                                    #(31, 4): 40
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i][3]
                newDataset.append([5, 1, dataset[i][2], time])
                placeCounter = placeCounter + 1
            if nextE == 4 and currentE == 21:                                    #(21, 4): 30
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i][3]
                newDataset.append([5, 1, dataset[i][2], time])
                placeCounter = placeCounter + 1
            if nextE == 1 and currentE == 11:                                    #(11, 1): 16
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i][3]
                newDataset.append([2, 1, dataset[i][2], time])
                placeCounter = placeCounter + 1
            if nextE == 4 and currentE == 1 and error == 2:                                    #(1, 4): 10
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                possible = [2,6]
                newDataset.append([random.choice(possible), 1, dataset[i][2], time]) 
                missingCounter = missingCounter + 1
            if nextE == 9 and currentE == 6:                                    #(6, 9): 5
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i+1][3]
                newDataset.append([4, 1, dataset[i][2], time])  
                placeCounter = placeCounter + 1
            if nextE == 10 and currentE == 9 and error == 2:                                    #(9, 10): 4
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time])                  
                newDataset.append([6, 1, dataset[i][2], time])  
                missingCounter = missingCounter + 2
            if nextE == 4 and currentE == 10:                                    #(10, 4): 28
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i][3]
                newDataset.append([6, 1, dataset[i][2], time]) 
                placeCounter = placeCounter + 1
            if nextE == 9 and currentE == 5:                                    #(5, 9): 2
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i+1][3]
                newDataset.append([4, 1, dataset[i][2], time]) 
                placeCounter = placeCounter + 1
            if nextE == 23 and currentE == 9:                                    #(9, 23): 1,
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i+1][3]
                newDataset.append([4, 1, dataset[i][2], time]) 
                placeCounter = placeCounter + 1
            if nextE == 2 and currentE == 23:                                    #(23, 2): 17
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i][3]
                newDataset.append([4, 1, dataset[i][2], time]) 
                placeCounter = placeCounter + 1
            if nextE == 31 and currentE == 4:                                    #(4, 31): 21
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i+1][3]
                newDataset.append([5, 1, dataset[i][2], time])
                placeCounter = placeCounter + 1
            if nextE == 27 and currentE == 5:                                    #(5, 27): 7
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i+1][3]
                newDataset.append([4, 1, dataset[i][2], time])
                placeCounter = placeCounter + 1
            if nextE == 2 and currentE == 5 and error == 2:                                    #(5, 2): 19
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time])
                missingCounter = missingCounter + 1
            if nextE == 23 and currentE == 2:                                    #(2, 23): 12
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                dataset[i+1][3]
                newDataset.append([4, 1, dataset[i][2], time]) 
                placeCounter = placeCounter + 1
            if nextE == 1 and currentE == 23:                                    #(23, 1): 2
                time = dataset[i][3]
                newDataset.append([4, 1, dataset[i][2], time])
                if error == 2:
                    time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                    possible = [2,6]
                    newDataset.append([random.choice(possible), 1, dataset[i][2], time]) 
                missingCounter = missingCounter + 1
                placeCounter = placeCounter + 1
            if nextE == 11 and currentE == 1:                                    #(1, 11): 8,
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i+1][3]
                newDataset.append([2, 1, dataset[i][2], time]) 
                placeCounter = placeCounter + 1                
            if nextE == 4 and currentE == 11:                                    #(11, 4): 5,
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([2, 1, dataset[i][2], time])
                placeCounter = placeCounter + 1
            if nextE == 23 and currentE == 10:                                    #(10, 23): 1
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i+1][3]
                newDataset.append([6, 1, dataset[i][2], time]) 
                newDataset.append([4, 1, dataset[i][2], time]) 
            if nextE == 6 and currentE == 5 and error == 2:                                    #(5, 6): 8,
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time]) 
                missingCounter = missingCounter + 1
            if nextE == 27 and currentE == 21:                                    #(21, 27): 1
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i][3]
                newDataset.append([5, 1, dataset[i][2], time]) 
                time = dataset[i+1][3]
                newDataset.append([4, 1, dataset[i][2], time]) 
                placeCounter = placeCounter + 2
            if nextE == 2 and currentE == 31:                                    #(31, 2): 2
                time = dataset[i][3]
                newDataset.append([5, 1, dataset[i][2], time])
                if error == 2:
                    time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                    newDataset.append([4, 1, dataset[i][2], time]) 
                missingCounter = missingCounter + 1
                placeCounter = placeCounter + 1
            if nextE == 5 and currentE == 9:                                    #(9, 5): 1,
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i][3]
                newDataset.append([4, 1, dataset[i][2], time])  
                placeCounter = placeCounter + 1
            if nextE == 10 and currentE == 4:                                    #(4, 10): 10
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i+1][3]
                newDataset.append([6, 1, dataset[i][2], time])  
                placeCounter = placeCounter + 1
            if nextE == 5 and currentE == 10:                                    #(10, 5): 3
                time = dataset[i][3]
                newDataset.append([6, 1, dataset[i][2], time]) 
                if error == 2:
                    time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                    newDataset.append([4, 1, dataset[i][2], time])
                missingCounter = missingCounter + 1
                placeCounter = placeCounter + 1
            if nextE == 21 and currentE == 6:                                    #(6, 21): 2
                if error == 2:
                    time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                    newDataset.append([4, 1, dataset[i][2], time]) 
                time = dataset[i+1][3]
                newDataset.append([5, 1, dataset[i][2], time])
                missingCounter = missingCounter + 1
                placeCounter = placeCounter + 1
            if nextE == 5 and currentE == 6 and error == 2:                                    #(6, 5): 22
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time]) 
                missingCounter = missingCounter + 1
            if nextE == 27 and currentE == 6:                                    #(6, 27): 3
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i+1][3]
                newDataset.append([4, 1, dataset[i][2], time])
                placeCounter = placeCounter + 1
            if nextE == 20 and currentE == 1:                                    # (1, 20): 2
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i+1][3]
                newDataset.append([2, 1, dataset[i][2], time])  
                placeCounter = placeCounter + 1
            if nextE == 31 and currentE == 10:                                    # (10, 31): 1
                time = dataset[i][3]
                newDataset.append([6, 1, dataset[i][2], time]) 
                if error == 2:
                    time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                    newDataset.append([4, 1, dataset[i][2], time]) 
                    time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                    newDataset.append([5, 1, dataset[i][2], time])
                placeCounter = placeCounter + 1
                missingCounter = missingCounter + 2
            if nextE == 6 and currentE == 31:                                    # (31, 6): 1
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i][3]
                newDataset.append([5, 1, dataset[i][2], time])
                time = dataset[i+1][3]
                newDataset.append([4, 1, dataset[i][2], time]) 
                placeCounter = placeCounter + 2
            if nextE == 6 and currentE == 2 and error == 2:                                    #(2, 6): 6
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                possible = [1,4]
                newDataset.append([random.choice(possible), 1, dataset[i][2], time]) 
                missingCounter = missingCounter + 1
            if nextE == 1 and currentE == 4 and error == 2:                                    #4, 1): 8
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                possible = [2,6]
                newDataset.append([random.choice(possible), 1, dataset[i][2], time]) 
                missingCounter = missingCounter + 1
            if nextE == 1 and currentE == 5 and error == 2:                                    #(5, 1): 1
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time]) 
                possible = [2,6]
                missingCounter = missingCounter + 1
                newDataset.append([random.choice(possible), 1, dataset[i][2], time]) 
            if nextE == 23 and currentE == 5:                                    #(5, 23): 2
                time = dataset[i+1][3]
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time])   
                placeCounter = placeCounter + 1
            if nextE == 20 and currentE == 23:                                    #(23, 20): 1
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i][3]
                newDataset.append([4, 1, dataset[i][2], time]) 
                time = dataset[i+1][3]
                newDataset.append([2, 1, dataset[i][2], time])   
                placeCounter = placeCounter + 2
            if nextE == 6 and currentE == 23:                                    #(23, 6): 10
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i][3] 
                newDataset.append([4, 1, dataset[i][2], time]) 
                placeCounter = placeCounter + 1
            if nextE == 5 and currentE == 2 and error == 2:                                    #(2, 5): 4
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                newDataset.append([4, 1, dataset[i][2], time]) 
                missingCounter = missingCounter + 1
            if nextE == 2 and currentE == 9:                                    #(9, 2): 8
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i][3]
                newDataset.append([4, 1, dataset[i][2], time]) 
                placeCounter = placeCounter + 1
            if nextE == 23 and currentE == 6:                                    #(6, 23): 2
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i+1][3]
                newDataset.append([4, 1, dataset[i][2], time])
                placeCounter = placeCounter + 1                 
            if nextE == 6 and currentE == 9:                                    #(9, 6): 2
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i+1][3]
                newDataset.append([4, 1, dataset[i][2], time])  
                placeCounter = placeCounter + 1
            if nextE == 4 and currentE == 20:                                    #(20, 4): 5
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i][3]
                newDataset.append([2, 1, dataset[i][2], time]) 
                placeCounter = placeCounter + 1                
            if nextE == 9 and currentE == 2:                                    #(2, 9): 5
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i+1][3]
                newDataset.append([4, 1, dataset[i][2], time])
                placeCounter = placeCounter + 1
            if nextE == 21 and currentE == 4:                                    #(4, 21): 2
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i+1][3]
                newDataset.append([5, 1, dataset[i][2], time])
                placeCounter = placeCounter + 1
            if nextE == 6 and currentE == 27:                                    #(27, 6): 2
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i][3]
                newDataset.append([4, 1, dataset[i][2], time]) 
                placeCounter = placeCounter + 1
            if nextE == 2 and currentE == 10:                                    #(10, 2): 5
                time = dataset[i][3]
                newDataset.append([6, 1, dataset[i][2], time]) 
                possible = [1,4]
                if error == 2:
                    time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                    newDataset.append([random.choice(possible), 1, dataset[i][2], time])
                placeCounter = placeCounter + 1
                missingCounter = missingCounter + 1
            if nextE == 2 and currentE == 6 and error == 2:                                    #(6, 2): 16
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                possible = [1,4]
                newDataset.append([random.choice(possible), 1, dataset[i][2], time])
                missingCounter = missingCounter + 1                
            if nextE == 10 and currentE == 2:                                    #(2, 10): 2
                if error == 2:
                    time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                    possible = [1,4]
                    newDataset.append([random.choice(possible), 1, dataset[i][2], time])
                time = dataset[i+1][3]
                newDataset.append([6, 1, dataset[i][2], time])
                placeCounter = placeCounter + 1
                missingCounter = missingCounter + 1
            if nextE == 5 and currentE == 1 and error == 2:                                    #(1, 5): 3
                time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                possible = [2,6]
                newDataset.append([random.choice(possible), 1, dataset[i][2], time])
                newDataset.append([4, 1, dataset[i][2], time])
                missingCounter = missingCounter + 2
            if nextE == 31 and currentE == 6:                                    #(6, 31): 1
                if error == 2:
                    time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                    newDataset.append([4, 1, dataset[i][2], time])  
                time = dataset[i+1][3]
                newDataset.append([5, 1, dataset[i][2], time])
                placeCounter = placeCounter + 1
                missingCounter = missingCounter + 1
            if nextE == 20 and currentE == 9:                                    # (9, 20): 1
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i][3]
                newDataset.append([4, 1, dataset[i][2], time])   
                time = dataset[i+1][3]
                newDataset.append([2, 1, dataset[i][2], time]) 
                placeCounter = placeCounter + 2
            if nextE == 20 and currentE == 9:                                    # (20, 1): 1
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i][3]
                newDataset.append([4, 1, dataset[i][2], time])
                time = dataset[i+1][3]
                newDataset.append([2, 1, dataset[i][2], time]) 
                placeCounter = placeCounter + 2
            if nextE == 11 and currentE == 6:                                    #(6, 11): 1
                if error == 2:
                    time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                    possible = [1,4]
                    newDataset.append([random.choice(possible), 1, dataset[i][2], time])
                time = dataset[i+1][3]
                newDataset.append([2, 1, dataset[i][2], time])
                placeCounter = placeCounter + 1
                missingCounter = missingCounter + 1
            if nextE == 9 and currentE == 20:                                    # (20, 9): 1
                #time = int(dataset[i][3] + (dataset[i+1][3]-dataset[i][3])/2)
                time = dataset[i][3]
                newDataset.append([2, 1, dataset[i][2], time]) 
                time = dataset[i+1][3]
                newDataset.append([4, 1, dataset[i][2], time]) 
                placeCounter = placeCounter + 2
               
    return newDataset

def getToggleStateDataset(filename, delta, opt, fix, error):
    '''
    getStateDataset returns states of the aparmtent in which each value 0 or 1 corresponds 
    to one sensor if it was activated at that certain slot, a state every delta seconds, opt is for (group of) sensors
    '''
    dataset = getEventsDataset(filename, opt, fix, error)    
    dataset = changeForContinuousTime(dataset)
    
    sensors = sorted(list(set(dataset[:,0])))
    nSensors = len(sensors)
    
    #get when the dataset starts and ends
    startH   = dataset[0,3]
    endH     = dataset[len(dataset)-1,3]
    state    = np.zeros(nSensors)
    
    times           = np.arange(startH,endH+delta,delta)
    #create states dataset for every x sec
    newDataset      = np.zeros((len(times), 1 + nSensors))
    newDataset[:,0] = times
    i = 0

    for j in range(len(newDataset)-1):
        start = newDataset[j,0]
        end = newDataset[j+1, 0]        
        while (i < len(dataset)) and (dataset[i,3] >= start) and (dataset[i,3] < end):
            #find which sensor activated and put in state
            s = sensors.index(int(dataset[i,0]))
            state[s] = 1 
            i = i + 1
        newDataset[j,1:] = state
        state            = np.zeros(nSensors)
    return newDataset
    
def getStateDataset(filename, delta, opt, fix=True, error=0):
    '''
    getStateDataset returns states of the aparmtent in which each value 0 or 1 corresponds 
    to one sensor, a state every sec seconds
    '''
    dataset = getEventsDataset(filename, opt, fix, error)
    dataset = changeForContinuousTime(dataset)
    
    sensors = sorted(list(set(dataset[:,0])))
    nSensors = len(sensors)
    
    #get when the dataset starts and ends
    startH   = dataset[0,3]
    endH     = dataset[len(dataset)-1,3]
    state    = np.zeros(nSensors)
    
    times           = np.arange(startH,endH+delta,delta)
    #create states dataset for every x sec
    newDataset      = np.zeros((len(times), 1 + nSensors))
    newDataset[:,0] = times
    
    lastRoom = -1
    out      = 0
    lastState = np.zeros(nSensors)
    powerSensors = [19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30]
    i = 0

    for j in range(len(newDataset)-1):
        
        start = newDataset[j,0]
        end = newDataset[j+1, 0]        
       
        while (i < len(dataset)) and (dataset[i,3] >= start) and (dataset[i,3] < end):
            
            s = sensors.index(int(dataset[i,0]))
            
            if dataset[i,1] == 1:
                state[s] = 1 
            
            #keep track of last room
            if dataset[i,0] in [1,2,4,5,6]:
                lastRoom = int(dataset[i,0])
                
            #keep track of power sensors
            if dataset[i,0] in powerSensors:
                lastState[s] = dataset[i,1]
            
            #check if person is out of the apt
            if dataset[i,0] == 10 and dataset[i,1] == 1 and dataset[i+2,0] == 10 and dataset[i+2,1] == 1:
                out = 1
            elif dataset[i,0] == 10 and dataset[i,1] == 1 and dataset[i-2,0] == 10 and dataset[i-2,1] == 1:
                out = 0
            
            i = i + 1
            
        #if person did not change rooms
        if (not np.any(state[:6])) and (out == 0):
            if lastRoom != -1:
                s = sensors.index(lastRoom)
                state[s] = 1
        
        for k in powerSensors:
            if k in sensors:
                pS = sensors.index(k)
                if (lastState[pS] == 1):
                    state[pS] = 1
            
        newDataset[j,1:] = state
        state            = np.zeros(nSensors)
    return newDataset

def getSensorDurationDataset(filename, delta, opt, fix=True, error=0):
    '''
    getSensorDurationDataset returns the duration each sensor is on in order of occurrence. The duration is d*delta seconds
    '''
    dataset = getEventsDataset(filename, opt, fix, error)
    dataset = changeForContinuousTime(dataset)
    
    sensors = sorted(list(set(dataset[:,0])))
    nSensors = len(sensors)
    motionS  = [1,2,4,5,6]

    newDataset = []
    inp = np.zeros((nSensors))
    
    for s in range(len(dataset)):
        
        if dataset[s,0] in motionS:
            for i in range(s+1,len(dataset)):
                if dataset[i,0] in motionS and not dataset[i,0]==dataset[s,0]:
                    duration = int((dataset[i,3]-dataset[s,3])/delta)
                    if duration == 0:
                        break                       
                    inp[sensors.index(int(dataset[s,0]))] = duration
                    newDataset.append(inp)
                    inp = np.zeros((nSensors))
                    break
        elif dataset[s,0] in sensors and dataset[s,1]==1:
            for i in range(s+1,len(dataset)):
                if dataset[i,0]==dataset[s,0]:
                    duration = int((dataset[i,3]-dataset[s,3])/delta)
                    if duration == 0:
                        break  
                    inp[sensors.index(int(dataset[s,0]))] = duration
                    newDataset.append(inp)
                    inp = np.zeros((nSensors))
                    break            
    
    newDataset = np.array(newDataset)
    data       = []
    
    i = 0
    while i<=len(newDataset)-1:
        j = 1
        while (i+j)<=len(newDataset)-1 and np.nonzero(newDataset[i,:])==np.nonzero(newDataset[i+j,:]):
            j = j+1
        if j > 1:
            newLine = sum(newDataset[i:i+j,:])
        else:
            newLine = newDataset[i,:]
        data.append(newLine.tolist())
        i = i+j

    data = np.array(data)
    return data

def getToggleTimeDataset(filename, opt, fix=True, error=0):
    '''
    returns two arrays: one with toggle and other with time
    '''
    
    dataset = getEventsDataset(filename, opt, fix, error)
        
    sensors = sorted(list(set(dataset[:,0])))
    nSensors = len(sensors)
    
    toggleData = []
    timeData   = []
    inpToggle  = np.zeros((nSensors))
    inpTime    = np.zeros((nSensors))
    
    lastRoom = -1
    rooms = [1,2,4,5,6]
    
    for s in range(len(dataset)):
        
        if dataset[s,0] != 3 and dataset[s,0] != 32:
            if dataset[s,1] == 1:
                if (dataset[s,0] in rooms) and dataset[s,0]!=lastRoom:
                    if (lastRoom!=-1):
                        inpToggle[sensors.index(int(lastRoom))] = -1
                        toggleData.append(inpToggle)
                        inpToggle = np.zeros((nSensors))
                        inpTime[sensors.index(int(dataset[s,0]))] = int(dataset[s,3])
                        timeData.append(inpTime)
                        inpTime    = np.zeros((nSensors))
                        lastRoom = int(dataset[s,0])
                    else:
                        lastRoom = int(dataset[s,0])
                    inpToggle[sensors.index(int(dataset[s,0]))] = 1
                elif dataset[s,0] not in rooms:
                     inpToggle[sensors.index(int(dataset[s,0]))] = 1
            elif dataset[s,1] == 0:
                inpToggle[sensors.index(int(dataset[s,0]))] = -1
            
            if inpToggle.any():
                toggleData.append(inpToggle)
                inpToggle = np.zeros((nSensors))
                
                inpTime[sensors.index(int(dataset[s,0]))] = int(dataset[s,3])
                timeData.append(inpTime)
                inpTime    = np.zeros((nSensors))
    
    toggleData = np.array(toggleData)
    timeData   = np.array(timeData)
    
    return toggleData, timeData

def getPositiveToggleDataset(filename, opt, fix=True, error=0):
    '''
    returns array with positive toggle in order
    '''
    dataset = getEventsDataset(filename, opt, fix, error)   
    
    sensors = sorted(list(set(dataset[:,0])))
    nSensors = len(sensors)
    
    toggleData = []
    inpToggle  = np.zeros((nSensors))
    
    for s in range(len(dataset)):
        if dataset[s,0] != 3 and dataset[s,0] != 32:
            if dataset[s,1] == 1:
                inpToggle[sensors.index(int(dataset[s,0]))] = 1
                toggleData.append(inpToggle)
                inpToggle = np.zeros((nSensors))

    toggleData = np.array(toggleData)
    newToggleData = []
    newToggleData.append(toggleData[0,:])
    for s in range(1,len(toggleData)):
        if not (toggleData[s,:]==toggleData[s-1,:]).all():
            newToggleData.append(toggleData[s,:])
        
    newToggleData = np.array(newToggleData)    
    return newToggleData

def getPositiveToggleWithTime(filename, opt, fix=True, error=0):
    '''
    returns array with positive toggle and value is time
    '''
    dataset = getEventsDataset(filename, opt, fix, error)  

    sensors = sorted(list(set(dataset[:,0])))       
    nSensors = len(sensors)
    
    toggleData = []
    inpToggle  = np.zeros((nSensors+1))
    
    for s in range(len(dataset)):
        if dataset[s,1] == 1:
            inpToggle[sensors.index(int(dataset[s,0]))] = 1
            inpToggle[-1] = dataset[s,3]/(60*60)
            toggleData.append(inpToggle.tolist())
            inpToggle = np.zeros((nSensors+1))

    toggleData = np.array(toggleData)
    newToggleData = []
    newToggleData.append(toggleData[0])
    for s in range(1,len(toggleData)):
        if not (toggleData[s]==toggleData[s-1]).all():
            newToggleData.append(toggleData[s])
        
    newToggleData = np.array(newToggleData)    
    return newToggleData, sensors

def getBinaryCodes(filename, opt, algo, fix=True, error=2):
    '''
    returns dataset as binary code of each number in sequence of letter + period of day
    '''
    
    if algo == "ALZ":
        data, sensors = getLeZiDatasetWithConcatPeriod(filename, opt, fix, error)
    elif algo == "SPEED":
        data, sensors = getSequenceConcatPeriod(filename, opt, fix, error)
        
    variables = sorted(list(set(data)))
    nVar      = len(variables)
    nBits     = math.ceil(math.log2(nVar))
    
    #generate codes
    codes = []
    for bits in product("01", repeat=nBits): 
        codes.append(''.join(bits)) 
        
    #assign a code for each variable
    varCodes = {}
    i = 0
    for var in variables:
        varCodes[var] = [int(x) for x in codes[i]]
        i             = i + 1

    codeSeq = []
    for e in data:
        codeSeq.append(varCodes.get(e))
    
    codeSeq = np.array(codeSeq)
    
    return codeSeq, varCodes, data, sensors

        
def getSequence(filename, opt, fix=True, error=0,mapping=False):
    """
    getSequence returns an array with the data from Skøyen data as sequence of events 
    as upper and lower case events
    the returned sequence is the input for SPEED algorithm 
    """
    
    #read file and get events
    data = getEventsDataset(filename, opt, fix, error,mapping)
    data = changeForContinuousTime(data)
            
    #create a dictionary with every sensor id assigned to an alphabet symbol
    alphabet = list(string.ascii_lowercase)
    sensors = {}
    sensors_IDs = sorted(list(set(data[:,0])))
    for s, a in zip(sensors_IDs, alphabet):
        sensors[s] = a
          
    #create the sequence and add first event
    sequence = []
    first    = data[0]
    sensor   = first[0]
    state    = first[1]
    value    = sensors.get(sensor)
    room     = ''
    
    if sensor == 1 or sensor == 2 or sensor == 3 or sensor == 4 or sensor == 5 or sensor == 6 or sensor == 32:
        room = value
        
    #for sensors ON, letter is upper case
    if state == 1 : value = value.upper()
    
    #add sensor event to sequence
    sequence.append(value)
    
    #add all next events
    for l in range(1,len(data)):
        line   = data[l]
        sensor = line[0]
        state  = line[1]
        value  = sensors.get(sensor)
        
        if state == 1 : value = value.upper()
        
        #skip event if it is already the last one that happened 
        if  (not sequence[-1] == value) and (room != value.swapcase()) and (room != value) :
            if sensor == 1 or sensor == 2 or sensor == 3 or sensor == 4 or sensor == 5 or sensor == 6 or sensor == 32:
                sequence.append(room.lower())
                room = value
            sequence.append(value)
    
    sequence = cleanSequence(sequence)
        
    return sequence, sensors


def getSequencePeriod(filename, opt, fix=True, error=0):
    """
    getSequence returns an array with the data from Skøyen data as sequence of events 
    as upper and lower case events plus the period of day of that event, period is individual value
    the returned sequence is the input for SPEED algorithm 
    """
    
    #read file and get events
    data = getEventsDataset(filename, opt, fix, error)
    #data = aux.continuousTime(data)
        
    #generate alphabet symbols from a, ..., z
    alphabet = list(string.ascii_lowercase)

    #create a dictionary with every sensor id assigned to an alphabet symbol
    sensors  = {}
    letter = 0
    for line in data:
        sensor = int(line[0])
        #if sensor id is not in dictionary yet, add it with the assigned symbol
        if  sensors.get(sensor) == None:
            sensors[sensor] = alphabet[letter]
            letter          = letter + 1
          
    #create the sequence and add first event
    sequence = []
    first    = data[0]
    sensor   = first[0]
    state    = first[1]
    time   = line[3]/3600
    if time >= 7 and time <= 12:         #morning
        time = '1'
    elif time > 12 and time <= 17:       #afternoon
        time = '2'
    elif time > 17 and time <= 22:       #evening
        time = '3'
    elif time > 22 and time < 7:         #night
        time = '4'
    value    = sensors.get(sensor)
    room     = ''
    
    if sensor == 1 or sensor == 2 or sensor == 3 or sensor == 4 or sensor == 5 or sensor == 6 or sensor == 32:
        room = value
        
    #for sensors ON, letter is upper case
    if state == 1 : value = value.upper()
    
    #add sensor event to sequence
    sequence.append(value)
    sequence.append(time)
    
    #add all next events
    for l in range(1,len(data)):
        line   = data[l]
        sensor = line[0]
        state  = line[1]
        time   = line[3]/3600
        if time >= 7 and time <= 12:         #morning
            time = '1'
        elif time > 12 and time <= 17:       #afternoon
            time = '2'
        elif time > 17 and time <= 22:       #evening
            time = '3'
        elif time > 22 or time < 7:         #night
            time = '4'
            
        value  = sensors.get(sensor)
        
        if state == 1 : value = value.upper()
        
        #skip event if it is already the last one that happened 
        if  (not sequence[-1] == value) and (room != value.swapcase()) and (room != value) :
            if sensor == 1 or sensor == 2 or sensor == 3 or sensor == 4 or sensor == 5 or sensor == 6 or sensor == 32:
                sequence.append(room.lower())
                sequence.append(time)
                room = value
            sequence.append(value)
            sequence.append(time)
    
    sequence = cleanSequencePeriod(sequence)
        
    return sequence

def getSequenceConcatPeriod(filename, opt, fix=True, error=0):
    """
    getSequence returns an array with the data from Skøyen data as sequence of events 
    as upper and lower case events, plus the period of day of that event, period is concatenated with letter
    the returned sequence is the input for SPEED algorithm 
    """
    
    #read file and get events
    data = getEventsDataset(filename, opt, fix, error)
    #data = aux.continuousTime(data)
        
    #generate alphabet symbols from a, ..., z
    alphabet = list(string.ascii_lowercase)

    #create a dictionary with every sensor id assigned to an alphabet symbol
    sensors  = {}
    letter = 0
    for line in data:
        sensor = int(line[0])
        #if sensor id is not in dictionary yet, add it with the assigned symbol
        if  sensors.get(sensor) == None:
            sensors[sensor] = alphabet[letter]
            letter          = letter + 1
          
    #create the sequence and add first event
    sequence = []
    first    = data[0]
    sensor   = first[0]
    state    = first[1]
    time   = line[3]/3600
    if time >= 7 and time <= 12:         #morning
        time = '1'
    elif time > 12 and time <= 17:       #afternoon
        time = '2'
    elif time > 17 and time <= 22:       #evening
        time = '3'
    elif time > 22 or time < 7:         #night
        time = '4'
    value    = sensors.get(sensor)
    room     = ''
    
    if sensor == 1 or sensor == 2 or sensor == 3 or sensor == 4 or sensor == 5 or sensor == 6 or sensor == 32:
        room = value
        
    #for sensors ON, letter is upper case
    if state == 1 : value = value.upper()
    
    #add sensor event to sequence
    sequence.append(value+time)
    
    #add all next events
    for l in range(1,len(data)):
        line   = data[l]
        sensor = line[0]
        state  = line[1]
        time   = line[3]/3600
        if time >= 7 and time <= 12:         #morning
            time = '1'
        elif time > 12 and time <= 17:       #afternoon
            time = '2'
        elif time > 17 and time <= 22:       #evening
            time = '3'
        elif time > 22 or time < 7:         #night
            time = '4'
            
        value  = sensors.get(sensor)
        
        if state == 1 : value = value.upper()
        
        #skip event if it is already the last one that happened 
        if  (not sequence[-1] == value) and (room != value.swapcase()) and (room != value) :
            if sensor == 1 or sensor == 2 or sensor == 3 or sensor == 4 or sensor == 5 or sensor == 6 or sensor == 32:
                sequence.append(room.lower()+time)
                room = value
            sequence.append(value+time)
    
    sequence = cleanSequence(sequence)
        
    return sequence, sensors


def getDataDuration(filename, opt, fix=True, error=0, mapping=False):
    """
    getSequence returns an array with the data from Skøyen data as sequence of events 
    as upper and lower case events, plus the duration of the sensor's activation or deactivation
    duration is concatenated with letter
    """
    #read file and get events
    data = getEventsDataset(filename, opt, fix, error, mapping)
    #data = aux.continuousTime(data)
    data = np.asarray(data, dtype=object)
    data = addOffEvents(data)

    durations = np.ediff1d(data[:,-1])
    durations = np.reshape(durations, (durations.shape[0],1))
    neg_durations = np.where(durations[:,0]<0)
    for n in range(len(neg_durations[0])):
        index = neg_durations[0][n]
        durations[index,0] = 24*3600 - data[index,3] + data[index+1,3]
    
    data = np.append(data[:-1,:], durations, axis=1 )
            
    return data
        


def getSequenceT(filename, opt, fix=True, error=0):
    """
    getSequence returns an array as sequence of events 
    as upper and lower case events and also the duration in minutes (int) of the episodes after each letter
    the returned sequence is the input for M-SPEED algorithm    
    """
    
    #read file and get events
    data = getEventsDataset(filename, opt, fix, error)
    data = aux.continuousTime(data)

    #generate alphabet symbols from a, ..., z
    alphabet = list(string.ascii_lowercase)
    
    #create a dictionary with every sensor id assigned to an alphabet symbol
    sensors  = {}
    letter = 0
    for line in data:
        sensor = int(line[0])
        #if sensor id is not in dictionary yet, add it with the assigned symbol
        if  sensors.get(sensor) == None:
            sensors[sensor] = alphabet[letter]
            letter          = letter + 1
          
    #create the sequence and add first event
    sequence = []
    first    = data[0]
    sensor   = first[0]
    state    = first[1]
    value    = sensors.get(sensor)
    
    #for sensors ON, letter is upper case
    if state == 1 : value = value.upper()
    
    #add sensor event to sequence
    sequence.append(value)
    
    if sensor == 1 or sensor == 2 or sensor == 3 or sensor == 4 or sensor == 5 or sensor == 6 or sensor == 32:
        room = value
        roomT = first[3]
        
    
    #add all next events
    for l in range(1,len(data)):
        line   = data[l]
        sensor = line[0]
        state  = line[1]
        value  = sensors.get(sensor)
        
        if state == 1 : value = value.upper()
        
        #skip event if it is already the last one that happened 
        if  (not sequence[-1] == value) and (room != value.swapcase()) and (room != value) :
            
            if sensor == 1 or sensor == 2 or sensor == 3 or sensor == 4 or sensor == 5 or sensor == 6 or sensor == 32:
                sequence.append(room.lower())
                duration = line[3]-roomT
                if int(duration)>0: sequence.append(str(int(duration)))
                prevSensor = {k:v for k, v in sensors.items() if v == room}
                lastOpposite, = np.where(data[:l-1,0] == prevSensor) 
                if lastOpposite.size:
                    duration     = int((data[l,3] - data[lastOpposite[-1],3])*60)
                    if duration>0:
                        sequence.append(str(duration))
                room = value
                roomT = line[3]
                
            sequence.append(value)
            
            #find the time of the episode  
            lastOpposite, = np.where(data[:l,0] == sensor) 
            if lastOpposite.size:
                duration     = int((data[l,3] - data[lastOpposite[-1],3])*60)
                if int(duration)>0:
                    sequence.append(str(duration))
#    sequence = cleanSequenceT(sequence)      
    return sequence


def getLeZiDataset(filename, opt, fix=True, error=0):
    '''
    returns a sequence of letters corresponding to each sensorID
    '''
    data = getEventsDataset(filename, opt, fix, error)
    
    #generate alphabet symbols from a, ..., z
    alphabet = list(string.ascii_lowercase)

    #create a dictionary with every sensor id assigned to an alphabet symbol
    sensors  = {}
    letter = 0
    for line in data:
        sensor = int(line[0])
        #if sensor id is not in dictionary yet, add it with the assigned symbol
        if  sensors.get(sensor) == None:
            sensors[sensor] = alphabet[letter]
            letter             = letter + 1
    
    #create the sequence and add first event
    sequence = []
    sensor   = data[0,0]
    sequence.append(sensors.get(sensor))
    
    #add all next events
    for l in range(1,len(data)):
        sensor = data[l,0]
        value  = sensors.get(sensor)  
        #skip event if it is already the last one that happened 
        if  not sequence[-1] == value and data[l,1]==1:
            sequence.append(value)

    return sequence, sensors

def getSensorsEvents(filename, opt, fix=True, error=0):
    '''
    returns a sequence of letters corresponding to each sensorID
    '''
    data = getEventsDataset(filename, opt, fix, error)
    
    #create the sequence and add first event
    sequence = []
    sensor   = data[0,0]
    sequence.append(sensor)
    
    #add all next events
    for l in range(1,len(data)):
        sensor = data[l,0]
        #skip event if it is already the last one that happened 
        if  not sequence[-1] == sensor:
            sequence.append(sensor)

    return sequence

def getLeZiDatasetWithPeriod(filename, opt, fix=True, error=0):
    '''
    returns a sequence of letters corresnponding to sensorID and period of day
    '''
    data = getEventsDataset(filename, opt, fix, error)
    
    #generate alphabet symbols from a, ..., z
    alphabet = list(string.ascii_lowercase)

    #create a dictionary with every sensor id assigned to an alphabet symbol
    sensors  = {}
    letter = 0
    for line in data:
        sensor = int(line[0])
        #if sensor id is not in dictionary yet, add it with the assigned symbol
        if  sensors.get(sensor) == None:
            sensors[sensor] = alphabet[letter]
            letter             = letter + 1
    
    #create the sequence and add first event
    sequence = []
    sensor   = data[0,0]
    sequence.append(sensors.get(sensor))
    
    time   = data[0,3]/3600
    if time >= 7 and time <= 12:         #morning
        time = '1'
    elif time > 12 and time <= 17:       #afternoon
        time = '2'
    elif time > 17 and time <= 22:       #evening
        time = '3'
    elif time > 22 or time < 7:         #night
        time = '4'
    sequence.append(time)
    
    #add all next events
    for l in range(1,len(data)):
        sensor = data[l,0]
        value  = sensors.get(sensor)  
        #skip event if it is already the last one that happened 
        if  not sequence[-2] == value:
            sequence.append(value)
            time   = data[l,3]/3600
            if time >= 7 and time <= 12:         #morning
                time = '1'
            elif time > 12 and time <= 17:       #afternoon
                time = '2'
            elif time > 17 and time <= 22:       #evening
                time = '3'
            elif time > 22 or time < 7:         #night
                time = '4'
            sequence.append(time)

    return sequence, sensors

def getLeZiDatasetWithConcatPeriod(filename, opt, fix=True, error=0):
    '''
    returns a sequence of letters corresnponding to sensorID and period of day
    '''
    sequence, sensors = getLeZiDatasetWithPeriod(filename, opt, fix, error)
    
    new_sequence = []
    for i in range(0,len(sequence),2):
        new_sequence.append(sequence[i]+sequence[i+1])
        

    return new_sequence, sensors

##########################################################################################
##################################### AUXILIAR FUNCTIONS #################################
##########################################################################################
    
def addOffEvents(data):
    '''
    adds off events of pir sensors to dataset
    '''
    pirs     = [1,2,3,4,5,6,32]
    newData  = []
    newData.append(data[0].tolist())
    lastRoom = data[0][0] if data[0][0] in pirs else 0
    for i in range(1,len(data)-1):
        if (data[i][0] in pirs) and (data[i][0] != lastRoom): 
            newData.append([lastRoom, 0, data[i][2], data[i][3]])
        if not (data[i][0] == lastRoom):
            newData.append(data[i].tolist())
        if data[i][0] in pirs:
            lastRoom = data[i][0]
    return np.array(newData)

def getWeekDay(dt):
    """
    getWeekDay returns the corresponding day of the week from 0 to 6 (Mon to Sun)        
    """
    year, month, day = (int(x) for x in dt.split('-')) 
    weekDay          = datetime.date(year, month, day).weekday()    
    return weekDay


def getHour(t):
    """
    getHour returns the time in seconds       
    """
    hr, m, s = (int(float(x)) for x in t.split(':'))
    time = int(hr + m/60 + s/3600)
    return time

def getSeconds(t):
    """
    getHour returns the time in seconds       
    """
    hr, m, s = (int(float(x)) for x in t.split(':')) 
    time = hr*3600+m*60+s
    return time

def cleanSequence(sequence):
    sensors = {}
    newSequence = []
    for e in sequence:
        #if not in dictionary, add. Else, check if last one was different case
        if  sensors.get(e.lower()) == None: 
            sensors[e.lower()] = e
            newSequence.append(e)
        else:
            if e != sensors[e.lower()]:
                newSequence.append(e)
                sensors[e.lower()] = e
                
    return newSequence

def cleanSequencePeriod(sequence):
    sensors = {}
    newSequence = []
    for event in range(0,len(sequence),2):
        e = sequence[event]
        #if not in dictionary, add. Else, check if last one was different case
        if  sensors.get(e.lower()) == None: 
            sensors[e.lower()] = e
            newSequence.append(e)
            newSequence.append(sequence[event+1])
        else:
            if e != sensors[e.lower()]:
                newSequence.append(e)
                newSequence.append(sequence[event+1])
                sensors[e.lower()] = e
                
    return newSequence

def cleanSequenceT(sequence):
    
    newSequence = []
    for e in range(len(sequence)):
        n = sequence[e]
        if n.isdigit():
            p = sequence[e-1]
            op = p.swapcase()
            for i in range(len(sequence[:e-1]), -1, -1):
                check = sequence[i]
                if check == op:
                    seq = sequence[i:e-1]
                    seq = [x for x in seq if (x.isdigit())]
                    sumD = sum(list(map(int,seq)))
                    if sumD < int(n):
                        newSequence.append(n)
                    break
        else:
            newSequence.append(n)
                    
    return newSequence

def splitDayPeriods(filename, opt, ap, fix=True, error=0):
    '''
    returns different datasets for different periods of the day, according to fixed time
    '''  
    
    dataset = getEventsDataset(filename, opt, fix, error)
    
    morningData   = []
    afternoonData = []
    eveningData   = []
    nightData     = []
    
    for i in dataset:
        hour = i[3]%24
        if hour > 7 and hour < 12:
            morningData.append(i)
        elif hour >=12 and hour < 18:
            afternoonData.append(i)
        elif hour >= 18 and hour < 22:
            eveningData.append(i)
        else:
            nightData.append(i)
    
#    morningData   = getLeZiDataset(ap, np.array(morningData))
#    afternoonData = getLeZiDataset(ap, np.array(afternoonData)) 
#    eveningData   = getLeZiDataset(ap, np.array(eveningData)) 
#    nightData     = getLeZiDataset(ap, np.array(nightData)) 
    
    return np.array(morningData),  np.array(afternoonData),  np.array(eveningData), np.array(nightData)

def changeForContinuousTime(data):
    
    diffs   = np.ediff1d(data[:,3])
    indexes = np.where(diffs<0)
    indexes = np.array(list(map(lambda x:x+1, indexes)))
    indexes = np.append(indexes,len(data))
    dataset = np.copy(data)
    
    for i in range(len(indexes)-1):
        start = indexes[i]
        end   = indexes[i+1]
        seconds  = 24*3600*(i+1)
        for j in range(start,end):
            dataset[j,3] = dataset[j,3] + seconds
    
    return dataset
