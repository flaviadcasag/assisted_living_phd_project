# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:39:08 2018

@author: flacas
"""

class Node:
    """
    node in tree used for the PPM algorithm
    """

    def __init__(self, parent, value, frequency):
        """
        create a  new node from parent and frequency
        """
        self.parent    = parent
        self.frequency = frequency
        self.value     = value
        self.children  = {}
        self.escape    = 1

    def addChild(self, child):
        """
        add child to node
        """
        self.children[child.value] = child
        # update the escape probability at this node
        s = 0 # sum of children frequency
        for child in self.children.values():
            s = s + child.frequency
        self.escape = (self.frequency - s) / self.frequency

    def probability(self, value):
        """
        calculate probability of next symbol being value
        """
        if value in self.children:
            return self.children[value].frequency / self.frequency
        return 0.0

    def __str__(self, level=0):
        ret = "\t"*level+repr(self.value)+"("+str(self.frequency)+")"+"\n"
        for child in self.children.values():
            ret += child.__str__(level+1)
        return ret

    def __repr__(self):
        return '<tree node representation>'

class Tree:
    """
    tree used for PPM algorithm
    """

    def __init__(self,root):
        """
        init tree with root node 0
        """
        self.root    = root    # root node of the tree
        self.leaves  = []
        self.symbols = set([]) # set of all symbols represented by nodes in the tree

    def findNode(self, path):
        """
        node will return the node defined by path in the tree starting from
        root node.
        Expects the nodes to already be in place, i.e. path must exist in the
        tree already.
        """
        node = self.root
        for v in path:
            node = node.children[v]
        return node

    def createTree(self, contexts):
        """
        generates nodes of the tree
        """
        # create a tree with the contexts
        root = Node(None, 0, 100)
        self.root = root

        for k, v in contexts.items():
            self.addNode(k[:-1], k[-1], v)

        rootFrequency = 0
        for children in root.children.values():
            rootFrequency = rootFrequency + children.frequency
        root.frequency = rootFrequency
        
        return self

    def printTree(self):
        """
        prints the tree with values and frequencies
        """
        print(self.root)

    def addNode(self, seq, value, freq):
        """
        addNode creates a new node with value and frequency freq and adds it
        after the sequence defined by seq in the tree.
        """
        # find parent and add the new node to it's children
        parent = self.findNode(seq)
        node = Node(parent, value, freq)
        parent.addChild(node)

        # add to the set of symbols in the tree
        self.symbols.add(value)

        # update leaves
        # add the new node to the set of leaves and remove the parent node
        self.leaves.append(node)
        if parent in self.leaves:
            self.leaves.remove(parent)

    def calculateProbability(self, seq, v):
        """
        calculateProbability calculates the probability of a value v following
        a sequence seq.
        """
        node = self.root
        p = node.probability(v)
        for e in seq:
            if not e in node.children:
                break
            node = node.children[e]
            p = p * node.escape + node.probability(v)
        return p

    def calculateProbabilities(self, seq):
        """
        calculate probability of every symbol after a certain sequence
        """
        probs = {}
        for s in self.symbols:
            probs[s] = self.calculateProbability(seq, s)
        return probs

    def predictNextEvent(self, seq):
        """
        finds the event that has the highest probability of happening
        """
        probs = self.calculateProbabilities(seq)
        return max(probs, key=probs.get)
    
    def predictNextEventM(self, seq, timeStorage):
        """
        finds the event that has the highest probability of happening. if same probability, return with longest duration
        """
        probs = self.calculateProbabilities(seq)      
        return self.getLongestEvent(probs, timeStorage)
    
    def getLongestEvent(self, probs, timeStorage):
        maxValue = max(probs.values())
        keys     = [key for key in probs.keys() if probs[key]==maxValue]
        if len(keys) > 1:
            maxD = 0
            nextE = keys[0]
            for s in keys:
                if s in timeStorage and timeStorage.get(s) > maxD:
                    nextE = s
                    maxD = timeStorage.get(s)
            return nextE
        return keys[0]
