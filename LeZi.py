# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 13:22:58 2018

@author: flacas_b
"""

import TreePPM
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import pickle

def getContexts(episode, contexts):
    for j in range(0,len(episode)):
        context = tuple(episode[-j-1:])
        if not context in contexts:
            contexts[context] = 1
        else:
            contexts[context] = contexts.get(context) + 1
                
def LeZi(sequence):
    
    dictionary  = {}
    contexts    = {}
    phrase      = []
    window      = []
    maxLZlength = 0

    for v in sequence:
        
        phrase.append(v)
        newContext = tuple(phrase)
        if not newContext in dictionary:
            dictionary[newContext] = 1
            if maxLZlength < len(phrase):
                maxLZlength = len(phrase)
            phrase = []     
 
        window.append(v)
        if len(window) > maxLZlength:
            #delete first element
            window = window[1:]
            
        getContexts(window, contexts)
    
    return contexts, maxLZlength

######## test data
#sequence = ['a','a','d','d','a','b','d','d','b','c','c','b','d','d','d','b','c','c','c','b','a','a','a','d','d','b','a','a','a','b','c','c','c','b','a','a','d']           
#sequence = ['a','a','a','b','a','b','b','b','b','b','a','a','b','c','c','d','d','c','b','a','a','a','a']

######## CASAS data       
#filename = "mavlab/march.in"
#sequenceMarch = getSequenceData(filename)
#filename = "mavlab/april.in"
#sequence = CASASdata.getSequence(filename)


####PAPERS example
#sequence = ['a','b','c','d','e','b','d','e','b']
#contexts, maxLZlength = LeZi(sequence)
#tree = TreePPM.Tree(TreePPM.Node(None,0,0))
#tree = tree.createTree(contexts)