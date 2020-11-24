"""
Author:  Quinn Bigane
Date: 4/1/2020
Program: This program will be an input adjustable, single output nueral network
"""
import numpy
import pandas
import itertools
import random
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

class Nueral_Network():
    """This class is one layer of a nueral network. Given 
    N inputs, it returns one output between 1 and 0"""
    def __init__ (self, excel_file = None, time_learn = 10000, learning_rate = .1, output = "Output", time_test = 10000):
        #dictionary that maps columns to their weights, randomized initially
        self.weights_dict = {}          #{col_name:weight} 
        #dictionary that maps column names to their data stored in a list
        self.inputs_dict = {}            #{col_name:[x1,...,xn]}
        #times that the training function will run
        self.time_learn = time_learn
        #times taht the test function will run
        self.time_test = time_test
        #rate at which weights are allowed to change 
        self.learning_rate = learning_rate
        #output key
        self.output = output
        #wrong count
        self.wrong_counter = None
    
        
        #if passed an excel file, 
        if excel_file != None:
            try:
                self.excel_df_number_gen(excel_file)
            except OSError:
                print("bad excel file")


        #run the training program
        self.train_dict(self.time_learn, self.learning_rate, self.inputs_dict, self.weights_dict)
        self.test_dict(self.time_test,self.inputs_dict,self.weights_dict)
        print(self.weights_dict)
        print("Percent Wrong: " + str(self.wrong_counter/self.time_test*100))

    def sigmoid(self, value):
        """Takes a number and noralizes it to be between 0 and 1"""
        return round(1/(1 + numpy.exp(-value)), 5)

    def excel_df_number_gen(self, excel_file):
        """Fed an excel file location, generates data for the NN"""
        
        #loops through columns in excel file populating inputs
        xl_df = pandas.read_excel(excel_file)
        for col in xl_df.columns:  
            self.inputs_dict[col] = xl_df[col]
    
         
        #loops through columns creating a weight for each, 
        for col in xl_df.columns:
            if col == self.output:
                self.weights_dict["bias"] = numpy.random.randn()
            else:
                self.weights_dict[col] = numpy.random.randn()

    def train_dict(self, time_learn, learning_rate, inputs_dict, weights_dict):
        #for time_learn
        self.costs = []
        for runtime_counter in range(time_learn):
            #pick a randomn data point
            datapoint = int(numpy.random.randint(0,len(inputs_dict[random.choice(list(inputs_dict.keys()))])))
            
            #calculate prediction
            z = 0
            for key in inputs_dict:
                if key == self.output:
                  z += weights_dict["bias"]  
                else:
                    z += inputs_dict[key][datapoint] * weights_dict[key]
            prediction = self.sigmoid(z)

            #cost calculation
            self.costs.append((prediction - inputs_dict[self.output][datapoint])**2)
                        
            #change weights
            for key in weights_dict.keys():
                if key == "bias":
                    #                                                   dcost_dpred                                             dpred_dz          dz_dweights             
                    weights_dict[key] -= (learning_rate * (2 * (prediction - inputs_dict[self.output][datapoint])) * (prediction * (1-prediction)) * (1))
                else:
                    #                                                   dcost_dpred                                             dpred_dz                dz_dweights             
                    weights_dict[key] -= (learning_rate * (2 * (prediction - inputs_dict[self.output][datapoint])) * (prediction * (1-prediction)) * inputs_dict[key][datapoint])
            
            #Program runn time display
            if (runtime_counter%(time_learn/10) == 0):
                percent_complete=round(runtime_counter*100/time_learn, 4)
                print("I'm training: " + str(percent_complete) + " Percent Complete")
        plt.plot(range(time_learn), self.costs)
        plt.show()
    
    def test_dict(self, time_test,inputs_dict,weights_dict):
        #for time_learn
        self.wrong_counter = 0
        for runtime_counter in range(time_test):
            #pick a randomn data point
            datapoint = int(numpy.random.randint(0,len(inputs_dict[random.choice(list(inputs_dict.keys()))])))
            #calculate prediction
            z = 0
            for key in inputs_dict:
                if key != self.output:
                    z += inputs_dict[key][datapoint] * weights_dict[key]
            z += weights_dict["bias"]
            prediction = self.sigmoid(z)

            if prediction >.5:
                prediction = 1
            else:
                prediction = 0
            self.wrong_counter += abs(prediction - inputs_dict[self.output][datapoint])
        
            #Program runn time display
            if (runtime_counter%(time_test/10) == 0):
                percent_complete=round(runtime_counter*100/time_test, 4)
                print("I'm testing: " + str(percent_complete) + " Percent Complete")
   
        

Test = Nueral_Network(excel_file="C:\\PythonDirectory\\NN_Project\\Stock_Training_Data.xlsx")

