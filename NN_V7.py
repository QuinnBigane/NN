"""
Title: NN_V7.py

Authors:
Quinn Bigane - qbigane@sandiego.edu

Date
5/11/2020

Iteration 7:
This iteration will deconstruct the nueral network to be more object 
oriented/architectural so it can be more easily iterated on in the 
future


"""
import numpy
import random

class nueralNetwork:
    def __init__(self):
        pass

class inputNode:
    """Data Node"""
    def _init__(self):
        self.name = None
        self.length = 0
        self.data = []

    def addOne(self,x):
        """Adds the passed item to the list of data, increments length"""
        self.data.append(x)
        self.length += 1
    def addMulti(self,x):
        """Adds the passed list of data to the list of data, increments 
        length"""
        for n in x:
            self.addOne(n)

    def __str__(self):
        """Returns a string representing the class"""
        return str(self.data)
        
class hiddenLayer:
    def __init__(self):
        passd