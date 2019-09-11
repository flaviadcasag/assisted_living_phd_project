# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:25:53 2018

@author: flacas_b
"""

import sys
sys.path.insert(0, 'C:/Users/flacas/Dropbox/Prediction')
sys.path.insert(0, 'C:/Users/flacas/Dropbox/Prediction/Bot')

import matplotlib.pyplot as plt
import TreePPM
import SKOYENdata_new as SKOYENdata
import time
import pandas as pd
import numpy as np
import pickle
import math
import time

def getContexts(episode,contexts):
    for j in range(len(episode)):
        for k in range(j+1,len(episode)+1):
            context = tuple(episode[j:k])
            if not context in contexts:
                contexts[context] = 1
            else:
                contexts[context] = contexts.get(context) + 1
                
def SPEED(sequence):
    
    maxEpisodeLength = 1
    window           = []
    contexts         = {}
    
    for event in sequence:
        window.append(event)
        oppositeEvent = event.upper() if event.islower() else event.lower()
        
        for i in range(len(window)-1, -1, -1):
            if window[i] == oppositeEvent:
                episode = window[i:]
                
                if len(episode) > maxEpisodeLength:
                    
                    maxEpisodeLength = len(episode)
                    maxEpisode = episode
                    
                window = window[len(window)-maxEpisodeLength:]                
                #add or update frequencies of all possible contexts within the episode
                #where max context length = maxEpisodeLength
                getContexts(episode, contexts)
                
                break

    return contexts, maxEpisodeLength, maxEpisode


#################  MAIN ##################


#sequence = ['A','B','b','D','C','c','a','B','C','b','d','c','A','D','a','B','A','d','a','b']
#sequence=['A','a','B','C','b','D','E','d','B','b','D','e','d','B']
#contexts, maxEpisodeLength, maxEpisode = SPEED(sequence)  
#filename = "mavlab/march.in"
#sequenceMarch = getSequenceData(filename)

#open and process CASAS data to input
#filename = "mavlab/april.in"
#sequenceApril = CASASdata.getSequence(filename)
#sequence = sequenceApril

#open and process SKØYEN data to input
#filename = "C:/Users/flacas/Dropbox/real data 2/203.csv"
#sequence = SKOYENdata.getSequence(filename,1)


##PAPER example
#sequence = ['A','a','B','C','b','D','E','d','B','b','D','e','d','B']
#contexts, maxEpisodeLength = SPEED(sequence)
#tree = TreePPM.Tree(TreePPM.Node(None,0,0))
#tree = tree.createTree(contexts)