# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:08:40 2019

@author: flacas_b
"""
import SKOYENdata_new as SKOYENdata
import numpy as np

SENSORS = {'TV': 23,
           'RADIO': 19,
           'NIGHT_LAMP': 20,
           'COFFEE_MACHINE': 21,
           'STOVE': 22,
           'KETTLE': 24,
           'BATHROOM_LAMP': 25,
           'KITCHEN_LAMP': 26,
           'MICROWAVE': 29,
           'TOASTER': 30,
           'LIV_ROOM_LAMP': 27,
           'WARDROBE': 11,
           'FRIDGE': 12,
           'CUPBOARD': 31,
           'ENT_DOOR': 10,
           'BALCONY_DOOR': 9,
           'BATHROOM': 1,
           'BEDROOM': 2,
           'OVER_BED': 3,
           'BY_BED': 32,
           'ENTRANCE': 6,
           'KITCHEN': 5,
           'LIV_ROOM': 4,
           'OUT': -1,
           }

ACTIVITIES = {SENSORS['COFFEE_MACHINE'] : {1: "in_kitchen_1", 0: "in_kitchen_0"},
              SENSORS['TV']:  {1: "watching_tv_1", 0 : "watching_tv_0"},
              SENSORS['RADIO']: { 1: "in_livroom_1", 0: "in_livroom_0"},
              SENSORS['STOVE']: {1: "in_kitchen_1", 0: "in_kitchen_0"},
              SENSORS['KETTLE']: {1: "in_kitchen_1", 0: "in_kitchen_0"},
              SENSORS['MICROWAVE']: {1: "in_kitchen_1", 0: "in_kitchen_0"},
              SENSORS['TOASTER']: {1: "in_kitchen_1", 0: "in_kitchen_0"},
              SENSORS['WARDROBE']: {1: "in_bedroom_1", 0: "in_bedroom_0"},
              SENSORS['FRIDGE']: {1: "in_kitchen_1", 0: "in_kitchen_0"},
              SENSORS['CUPBOARD']: {1: "in_kitchen_1", 0: "in_kitchen_0"},
              SENSORS['BEDROOM']: {1: "in_bedroom_1", 0: "in_bedroom_0"},
              SENSORS['BATHROOM']: {1: "in_bathroom_1", 0: "in_bathroom_0"},
              SENSORS['KITCHEN']: {1: "in_kitchen_1", 0: "in_kitchen_0"},
              SENSORS['LIV_ROOM']: {1: "in_livroom_1", 0: "in_livroom_0"},  
              SENSORS['ENTRANCE']: {1: "in_entrance_1", 0: "in_entrance_0"},  
              SENSORS['NIGHT_LAMP']: {1: "in_bedroom_1", 0: "in_bedroom_0"},
              SENSORS['LIV_ROOM_LAMP']: {1: "in_livroom_1", 0: "in_livroom_0"}, 
              SENSORS['BALCONY_DOOR']: {1: "in_livroom_1", 0: "in_livroom_0"}, 
              SENSORS['ENT_DOOR']: {1: "out_1", 0: "out_0"},
              }

ROOMS = {'LIV_ROOM': "living_room",
         'BATHROOM': "bathroom",
         'BEDROOM': "bedroom",
         'ENTRANCE': "entrance",
         'KITCHEN': "kitchen",
         'OUT': "out",        
        }

SENSOR_ROOMS = {SENSORS['TV']: ROOMS['LIV_ROOM'],
           SENSORS['RADIO']: ROOMS['LIV_ROOM'],
           SENSORS['NIGHT_LAMP']: ROOMS['BEDROOM'],
           SENSORS['COFFEE_MACHINE']: ROOMS['KITCHEN'],
           SENSORS['STOVE']: ROOMS['KITCHEN'],
           SENSORS['KETTLE']: ROOMS['KITCHEN'],
           SENSORS['BATHROOM_LAMP']: ROOMS['BATHROOM'],
           SENSORS['KITCHEN_LAMP']: ROOMS['KITCHEN'],
           SENSORS['MICROWAVE']: ROOMS['KITCHEN'],
           SENSORS['TOASTER']: ROOMS['KITCHEN'],
           SENSORS['LIV_ROOM_LAMP']: ROOMS['LIV_ROOM'],
           SENSORS['WARDROBE']: ROOMS['BEDROOM'],
           SENSORS['FRIDGE']: ROOMS['KITCHEN'],
           SENSORS['CUPBOARD']: ROOMS['KITCHEN'],
           SENSORS['ENT_DOOR']: ROOMS['ENTRANCE'],
           SENSORS['BALCONY_DOOR']: ROOMS['LIV_ROOM'],
           SENSORS['BATHROOM']: ROOMS['BATHROOM'],
           SENSORS['BEDROOM']: ROOMS['BEDROOM'],
           SENSORS['OVER_BED']: ROOMS['BEDROOM'],
           SENSORS['BY_BED']: ROOMS['BEDROOM'],
           SENSORS['ENTRANCE']: ROOMS['ENTRANCE'],
           SENSORS['KITCHEN']: ROOMS['KITCHEN'],
           SENSORS['LIV_ROOM']: ROOMS['LIV_ROOM'],
           SENSORS['OUT']: ROOMS['OUT'],
           }
    
POWER_SENSORS = ['TV', 'RADIO', 'NIGHT_LAMP', 'COFFEE_MACHINE', 'STOVE', 'KETTLE', 'BATHROOM_LAMP', 'KITCHEN_LAMP', 'MICROWAVE', 'TOASTER', 'LIV_ROOM_LAMP']
POWER_SENSORS = [SENSORS.get(s) for s in POWER_SENSORS]
MAG_SENSORS   = ['WARDROBE', 'FRIDGE', 'CUPBOARD', 'ENT_DOOR', 'BALCONY_DOOR']
MAG_SENSORS   = [SENSORS.get(s) for s in MAG_SENSORS]
PIRS          = ['BATHROOM', 'BEDROOM', 'OVER_BED', 'BY_BED', 'ENTRANCE', 'KITCHEN', 'LIV_ROOM']
PIRS          = [SENSORS.get(s) for s in PIRS]



def addLocation(data, apt):
    '''
    adds a column to the array with the location in the apartment for each event
    '''    
    data = np.append(data, np.zeros((len(data),1)), axis=1)
    
    
    #add out event
    if apt != "507":
        for i in range(len(data)-1):
            e = data[i]
            ne = data[i+1]   
            if e[0]==10 and e[1]==0 and ne[0]==10 and ne[1]==1 and (ne[3]-e[3])>=60:
                data[i-1][-1] = "out"
                data[i][-1] = "out"
                data[i+2][-1] = "out"
            else:
                if e[-1]==0:
                    e[-1] = SENSOR_ROOMS[e[0]]
    else:
        for i in range(len(data)-1):
            e = data[i]
            ne = data[i+1]        
            if e[0]==6 and (ne[3]-e[3])>=5*60:
                data[i][-1] = "out"
            else:
                if e[-1]==0:
                    e[-1] = SENSOR_ROOMS[e[0]]
                

    #check if events before and after are entrance to go out and put it to out
    out = np.where(data[:,-1]=="out")[0]
    for i in out:
        j = 1
        while data[i-j][-1]=="entrance":
            data[i-j][-1] = SENSOR_ROOMS[-1]
            j += 1
        j = 1
        while data[i+j][-1]=="entrance":
            data[i+j][-1] = SENSOR_ROOMS[-1]
            j += 1
        
    return data[:-1,:]

def addOnOffActivity(data):
    
    data = np.append(data, np.zeros((len(data),1)), axis=1)
    
    for event in data:
        sensor = event[0]
        msg    = event[1]
        if sensor==SENSORS['TV']:
            event[-1] = ACTIVITIES[sensor][msg]

    return data
    
    
def addInBed(data):
    
    #detecting in bed
    i = 0
    bed_sensors = [SENSORS['BEDROOM'], SENSORS['OVER_BED'], SENSORS['BY_BED'], SENSORS['NIGHT_LAMP']]
    thres = 5*60 #5 min
    while i <= len(data)-4:
        sensors = data[i:i+4]
        if np.all(np.in1d(sensors[:,0], bed_sensors)) and sensors[-1,3]-sensors[0,3] > thres:
            data[i,5] = "in_bed_1"     
            j = i+4
            while len(data)>j and data[j,0] in bed_sensors:
                j+=1
            data[j-1,5] = "in_bed_0"
            i = j
        else:
            i = i+1
       
    return data

def addInRoom(data):

    for i in data:
        act  = i[-1]
        room = i[-2]
        if act == 0:
            i[-1] = room + "_1"
    
    #delete consecutive repetitive activities
    toDelete = []
    for i in range(1, len(data)):
        cur_act = data[i,-1]
        prev_act = data[i-1,-1]
        if cur_act == prev_act:
            toDelete.append(i)           
    data = np.delete(data, toDelete, axis=0)
        
    return data

def getData(filename, opt="all", treatTransitions=False, keepTransitions=False, elapTime=True, durTime=False):

    apt = filename.split("/")[-1].split(".")[0]
    
    data = SKOYENdata.getEventsDataset(filename, opt, fix=True, error=2)
    data = np.asarray(data, dtype=object)
    data = addLocation(data, apt)
    data = addOnOffActivity(data)
    data = addInBed(data)
    data = addInRoom(data)

    #putting together room with activity
    priority = ['watching_tv_0', 'watching_tv_1', 'in_bed_0', 'in_bed_1', 'out_0', 'out_1']
    i = 0
    new_data = []
    while i<len(data)-2:
        sensors = data[i:i+2]
        acts = [a[:-2] for a in sensors[:,5]]
        if len(set(sensors[:,4]))==1 and len(set(acts))!=1:
            cur_room = sensors[0,4]
            j = i+2
            while len(data)>j and data[j,4]==cur_room:
                j+=1
            acts = set(data[i:j,-1])
            common = acts&set(priority)
            if len(common)>1:
                act = list(common)[0][:-2]
                new_data.append(np.append(data[i,:5],act+"_1"))
                new_data.append(np.append(data[j-1,:5],act+"_0"))
            elif len(common) == 1:
                act = list(common)[0]
                new_data.append(np.append(data[i,:5],act))
            else:
                act = list(acts)[0][:-2]
                new_data.append(np.append(data[i,:5],act+"_1"))
                new_data.append(np.append(data[j-1,:5],act+"_0"))
            i = j
        else:
            new_data.append(data[i])
            i+=1
    data = np.array(new_data)    
    
    rooms = ["bathroom", "bedroom", "kitchen", "living_room", "out", "entrance"]
    #include off of rooms
    new_data = []
    for i in range(len(data)-1):
        cur_act = data[i,-1][:-2]
        next_room = data[i+1,-2]
        if cur_act in rooms and next_room!=cur_act:
            new_data.append(data[i])
            new_data.append(np.append(data[i+1,:5],cur_act + "_0" ))
        else:
            new_data.append(data[i])
    data = np.array(new_data)
    
    #Elapsed time
    if elapTime:
        durations = np.ediff1d(data[:,3])
        durations = np.reshape(durations, (durations.shape[0],1))
        neg_durations = np.where(durations[:,0]<0)
        for n in range(len(neg_durations[0])):
            index = neg_durations[0][n]
            durations[index,0] = 24*3600 - data[index,3] + data[index+1,3]
        data = np.append(data[:-1,:], durations, axis=1)
    elif durTime:
        data = np.append(data, np.zeros((len(data),1)), axis=1)
        for i in range(len(data)-1):
            curr = data[i][5][:-1]
            msg  = int(data[i][5][-1])
            if msg == 1:
                opposite = 0 if msg == 1 else 1
                for j in range(i+1,len(data)):
                    nextCur = data[j][5][:-1]
                    nextMsg = int(data[j][5][-1])
                    if nextCur == curr and nextMsg == opposite:
                        dur = data[j][3] - data[i][3]
                        if dur < 0 :
                            dur = 24*3600 - data[i,3] + data[j,3]
                        data[i][-1] = dur
                        break
            else:
                nextCur = data[j][5][:-1]
                nextMsg = int(data[j][5][-1])
                dur = data[i+1][3] - data[i][3]
                if dur < 0 :
                    dur = 24*3600 - data[i,3] + data[i+1,3]
                data[i][-1] = dur

                
    transitionTimes = {'bedroom': 5*60,
                   'living_room': 5*60,
                   'bathroom': 2*60,
                   'entrance': 500000,
                   'kitchen': 60,
                   'out': 5*60,
                   }
    
    if treatTransitions:
        if keepTransitions:
            #recognize transitions
            for i in range(len(data)-1):
                curr = data[i][5][:-2]
                msg  = int(data[i][5][-1])
                nextCur = data[i+1][5][:-2]
                if (msg==1) and (nextCur == curr) and (curr in transitionTimes.keys()) and (data[i,-1]<transitionTimes[curr]):
                        data[i][5] = "transition_" + data[i][5] 
                        data[i+1][5] = "transition_" + data[i+1][5]
        else:
            deleteTransitions = []
            #recognize transitions
            for i in range(len(data)-1):
                curr = data[i][5][:-2]
                msg  = int(data[i][5][-1])
                nextCur = data[i+1][5][:-2]
                if (msg==1) and (nextCur == curr) and (curr in transitionTimes.keys()) and (data[i,-1]<transitionTimes[curr]):
                        deleteTransitions += [i, i+1]
            data = np.delete(data, deleteTransitions, axis=0)
            data = np.array(data)
            
    return data


