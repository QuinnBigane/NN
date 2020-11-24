import numpy
import pandas
import itertools
import xlrd

def Sigmoid(x):
    return round(1/(1 + numpy.exp(-x)), 5)

def NN_Train(time_learn,learning_rate,io,weights):
    """
    This function will take in:
        time_learn: number of times to train on the data
        learning_rate: number between 0-1 that changes how intensly the weights shift towards minimizing point
        io: dictionary of the data to train on including output
        weights: dictionary of the weights with the same keys as io, if the weights list is empty, will gen random
    """
    class WeightListLenError(Exception):
        #rasied if length of weights list given does not match the number of inputs given
        pass

    #check if there is a proper number of weights for the number of inputs


    #enter main training loop
    for time in range(time_learn):
        #pick a random data row
        ran_index = int(numpy.random.randint(0,len(io["Output"])))
        #reset value for z
        z = 0
        #loop through all inputs, multiply and add to total, then add b
        for key in weights:
            if key == "b":
                z += weights[key]
            else:
                z+= (weights[key] * io[key][ran_index])
    
        #normalize
        prediction = Sigmoid(z)
        #cost determinization
        #cost = (prediction - io["Output"])**2

        #change weights math
        #Weight change = (learning_rate) * (dcost_dpred) * (dpred_dz) * (dz_dweights)
        for key in weights:
            if key == "b":
                weights[key] -= ( learning_rate ) * ( 2 * (prediction - io["Output"][ran_index]) ) * ( prediction * (1-prediction) ) * ( 1 )
            else: 
                weights[key] -= ( learning_rate ) * ( 2 * (prediction - io["Output"][ran_index]) ) * ( prediction * (1-prediction) ) * ( io[key][ran_index] )

        percent_complete=round(time*100/time_learn, 4)
        if (time%(time_learn/10) == 0):
            print("Im Running: " + str(percent_complete) + " Percent Complete")
    return weights

def NN_Test(time_test,io,weights):
    test_counter = 0
    wrong_count = 0
    while(test_counter < time_test):
        #pick a random data row
        ran_index = int(numpy.random.randint(0,len(io["Output"])))
        #reset value for z
        z = 0
        #loop through all inputs, multiply and add to total, then add b
        for key in io:
            if key != "Output":
                z+= (weights[key] * io[key][ran_index])
        z+=weights["b"]
    
        #normalize
        prediction = Sigmoid(z)
        if(prediction > .5):
            prediction = 1
        else:
            prediction = 0
        wrong_count += abs(prediction-io["Output"][ran_index])
        test_counter+=1
    print("Percent Wrong: " + str(wrong_count/test_counter))
    return wrong_count/test_counter

def NN_Data_Retrieve():
    #need to use this function to feed NN_Train
    # number of times to learn, learning rate, io (dictionary with keys storing lists of data), weights (empty list if not stored))
    """
    ENTER DATA ON BACK END
    """
    excel_df = pandas.read_excel("Stock_Training_Data.xlsx", sheet_name="AMZN")
    time_learn = 100000
    time_test = 100000
    learning_rate = 1
    """
    ENTER DATA IN TERMINAL
    """
    # filename = input("Please enter file name: ")
    # excel_df = pandas.read_excel(filename)
    # time_learn = input("Please enter learning time: ")
    # time_test = input("Please enter testing time: ")
    # learning_rate = input("Please enter learning rate: ")

    weights = {}
    io = {}
    percent_wrong = 100

    #store inputs/outputs in dictionary indexed by col name
    for col_name in excel_df.columns:
        if col_name == "Output":
            weights["b"] = numpy.random.randn()
        else:
            weights[col_name] = numpy.random.randn()
        io[col_name] = []
        for index in range(len(excel_df[col_name])):
            io[col_name].append(excel_df[col_name][index])


    weights = NN_Train(time_learn,learning_rate,io,weights)
    percent_wrong = NN_Test(time_test, io, weights)
    print(weights)
    
NN_Data_Retrieve()