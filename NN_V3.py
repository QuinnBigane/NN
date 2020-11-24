import numpy
import pandas

def Sigmoid(x):
    return round(1/(1 + numpy.exp(-x)), 5)

def NN_Train(time_learn,learning_rate,inputs,weights):
    counter = 0
    for i in range(time_learn):
        y = int(numpy.random.randint(1,len(inputs[0])))
        z = 0
        for i in range(len(inputs) - 1):
            z += inputs[i][y] * weights[i]
        z += weights[-1]
        prediction = Sigmoid(z)
        cost = (prediction - inputs[-1][y])**2

        dcost_dpred = 2 * (prediction - inputs[-1][y])
        
        dpred_dz = prediction * (1-prediction)

        dz_dweights = []
        for i in range(len(inputs) - 1):
            dz_dweights.append(inputs[i][y])
        dz_dweights.append(1)
        

        dcosts_dweights = []
        for i in range(len(inputs)):
            dcosts_dweights.append(dcost_dpred * dpred_dz * dz_dweights[i])
        

        for i in range(len(weights)):
            weights[i] -= learning_rate * dcosts_dweights[i]
        
        counter +=1
        percent_complete=round(counter*100/time_learn, 4)
        if (counter%(time_learn/10) == 0):
          print("Im Running: " + str(percent_complete) + " Percent Complete")
    return weights

def NN_Test(time_test,inputs,weights):
    test_counter = 0
    wrong_count = 0
    while(test_counter < time_test):
        y = int(numpy.random.randint(1,len(inputs[0])))
        z = 0
        for i in range(len(inputs)-1):
            z += inputs[i][y] * weights[i]
        z += weights[-1]
        prediction = Sigmoid(z)
        if(prediction > .5):
            prediction = 1
        else:
            prediction = 0
        wrong_count += abs(prediction-inputs[-1][y])
        test_counter+=1
    print("Percent Wrong: " + str(wrong_count/test_counter))
    return wrong_count/test_counter

def NN_Numbergen(excel_data, col_names = []):
    weights = []
    inputs = []
    for i in range(len(col_names)):
        weights.append(numpy.random.randn())
    for i in range(len(col_names)):
        inputs.append(excel_data[col_names[i]])
    return inputs, weights

def NN_Run(excel_data,col_names, time_learn, time_test, learning_rate):
    percent_wrong = 1
    lst1=[]
    lst2=[]
    
    run_counter = 0
    print("Reseting weights")
    inputs, weights = NN_Numbergen(excel_data, col_names)
    while(percent_wrong > .1):
        print("\n\n=====Training=====\n\n")
        weights = NN_Train(time_learn,learning_rate,inputs, weights)
        print("\n\n=====Testing=====\n\n")
        percent_wrong = NN_Test(time_test,inputs, weights)
        lst1.append(percent_wrong)
        run_counter+=1
        print(lst1)
        print(weights)
    return weights
