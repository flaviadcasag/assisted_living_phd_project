# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:22:03 2018

@author: flacas_b
"""

import numpy as np

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

def addLocation(data, apt):
    '''
    adds a column to the array with the location in the apartment
    '''
    rooms = {
            -1: "out",
            1: "bathroom",
            3: "bedroom",
            32: "bedroom",
            2: "bedroom",
            4: "living_room",
            5: "kitchen",
            6: "entrance",
            19: "living_room",
            20: "bedroom",
            21: "kitchen",
            22: "kitchen",
            23: "living_room",
            24: "kitchen",
            25: "bathroom",
            26: "kitchen",
            27: "living_room",
            29: "kitchen",
            30: "kitchen",
            9: "living_room",
            10: "entrance",
            11: "bedroom",
            12: "kitchen",
            31: "kitchen",
            }
    
    data = np.append(data, np.zeros((len(data),1)), axis=1)
    
    
    if apt != "507":
        for i in range(len(data)-1):
            e = data[i]
            ne = data[i+1]
            nb = data[i-1]            
            if e[0]==10 and e[1]==0 and ne[0]==10 and ne[1]==1 and (ne[3]-e[3])>=60:
                e[-1]  = rooms[-1]
                nb[-1] = rooms[-1]
                ne[-1] = rooms[-1]
            else:
                if e[-1]==0:
                    e[-1] = rooms[e[0]]
    else:
        for i in range(len(data)-1):
            e = data[i]
            ne = data[i+1]        
            if e[0]==6 and (ne[3]-e[3])>=5*60:
                e[-1]  = rooms[-1]
            else:
                if e[-1]==0:
                    e[-1] = rooms[e[0]]        
    
    return data

def removeConsecutiveEqualEvents(dataset,c):
    #remove consecutive equal events
    newDataset = []
    newDataset.append(dataset[0])
    for i in range(1,len(dataset)):
        if not np.array_equal(dataset[i,c:],dataset[i-1,c:]):
            newDataset.append(dataset[i].tolist())        
    data = np.array(newDataset)
    return data

def computeDurations(data, posTime):
    
    durations = np.ediff1d(data[:,posTime])
    durations = np.reshape(durations, (durations.shape[0],1))
    neg_durations = np.where(durations[:,0]<0)
    for n in range(len(neg_durations[0])):
        index = neg_durations[0][n]
        durations[index,0] = 24*3600 - data[index,posTime] + data[index+1,posTime]
    return durations
   
def getData(filename, opt, keepTransitions=True, treatTransitions=True):

    apt = filename.split("/")[-1].split(".")[0]
    
    original_data = SKOYENdata.getEventsDataset(filename, opt, fix=True, error=2)
    data = np.asarray(original_data, dtype=object)
    data = addLocation(data, apt)
    
    state = {
            23: 0,         #tv
            19: 0,         #radio
            20: 0,         #night_lamp
            21: 0,         #coffee_machine
            22: 0,         #stove
            24: 0,         #kettle
            25: 0,         #bathroom_lamp
            26: 0,         #kitchen_lamp
            29: 0,         #microwave
            30: 0,         #toaster
            27: 0,         #living_room_lamp
            11: 0,         #wardrobe
            12: 0,         #fridge
            31: 0          #cupboard     
            }
    
    power = [19,20,21,22,23,24,25,26,27,29,30]
    mag   = [9,10,11,12,31]
      
    for i in range(len(data)):
        sensor = data[i][0]
        msg    = data[i][1]
        room   = data[i][4]
        if sensor in power or sensor in mag:
            state[sensor] = msg
            if sensor == 23 and state[sensor] == 1 and room == "living_room":
                data[i][-1] = "watching_tv"           
        elif sensor == 4:
            if state[23] == 1:
                data[i][-1] = "watching_tv"
                         
    #detecting in bed
    i = 0
    bed_sensors = [2,3,32,20]
    
    while i <= len(data)-7:
        flag = 0
        if data[i+4][3]-data[i][3] > 300: #more than 5 min    
            sensors = data[i:i+5,0]
            for v in sensors:
                if not v in bed_sensors:
                    flag = 1
                    break
            if flag == 0:
                data[i:i+5,-1] = "in_bed" 
        i = i+1
    
    data = removeConsecutiveEqualEvents(data[:,2:],2)
    
    ##put together pirs with activity
    
    #compute durations and: exclude pirs with duration < 1min or keep as transition activities
    durations = computeDurations(data,1)
    data = np.append(data[:-1,:], durations, axis=1)  
    
    #cluster locations with activities
    for i in range(len(data)-1):
        current_act = data[i,2]
        current_dur = data[i,-1]
        next_act  = data[i+1,2]
        next_dur  = data[i+1,-1]
        
        if current_act == "living_room" and next_act == "watching_tv" and current_dur<60*3:
            data[i,2] = next_act
        elif current_act == "watching_tv" and next_act == "living_room" and next_dur<60*3:
            data[i+1,2] = current_act
        elif current_act == "entrance" and next_act == "out" and current_dur<60*3:
            data[i,2] = next_act
        elif current_act == "out" and next_act == "entrance" and next_dur<60*3:
            data[i+1,2] = current_act
        elif current_act == "bedroom" and next_act == "in_bed" and current_dur<60*3:
            data[i,2] = next_act
        elif current_act == "in_bed" and next_act == "bedroom" and next_dur<60*3:
            data[i+1,2] = current_act
            
    data = removeConsecutiveEqualEvents(data[:,:-1],2)
    durations = computeDurations(data,1)
    data = np.append(data[:-1,:], durations, axis=1)  
    
    if treatTransitions:
    
        transitionTimes = {'bedroom': 5*60,
                           'living_room': 5*60,
                           'bathroom': 2*60,
                           'entrance': 500000,
                           'kitchen': 60,
                           }
    
    
        #identify transitions
        for i in data:
            activity = i[-2]
            dur      = i[-1]
            if activity in transitionTimes and transitionTimes[activity]>dur:
                i[-2] = "transition_" + activity
            if activity == "entrance":
                i[-2] = "transition_entrance"
        
        data = data[:,:-1]
        
        if not keepTransitions:
            new_data = []
            for i in data:
                if i[-1][:10] == "transition":
                    pass
                else:
                    new_data.append(i)
            data = np.array(new_data)
            #remove consecutive equal events
            data = removeConsecutiveEqualEvents(data,2)
            durations = computeDurations(data,1)
            data = np.append(data[:-1,:], durations, axis=1)
            for i in range(len(data)-1):
                current_act = data[i,2]
                current_dur = data[i,-1]
                next_act  = data[i+1,2]
                next_dur  = data[i+1,-1]
                
                if current_act == "living_room" and next_act == "watching_tv" and current_dur<60*3:
                    data[i,2] = next_act
                elif current_act == "watching_tv" and next_act == "living_room" and next_dur<60*3:
                    data[i+1,2] = current_act
                elif current_act == "entrance" and next_act == "out" and current_dur<60*3:
                    data[i,2] = next_act
                elif current_act == "out" and next_act == "entrance" and next_dur<60*3:
                    data[i+1,2] = current_act
                elif current_act == "bedroom" and next_act == "in_bed" and current_dur<60*3:
                    data[i,2] = next_act
                elif current_act == "in_bed" and next_act == "bedroom" and next_dur<60*3:
                    data[i+1,2] = current_act
            data = removeConsecutiveEqualEvents(data[:,:-1],2)
       
        #recompute durations 
        durations = computeDurations(data,1)
        data = np.append(data[:-1,:], durations, axis=1)
            
    return data   