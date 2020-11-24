import numpy
import pandas

def Sigmoid(x):
    return round(1/(1 + numpy.exp(-x)), 5)

def NN_Train(time_learn,learning_rate,x1,x2,x3,x4,x5,x6,x7,x8,tar,w1,w2,w3,w4,w5,w6,w7,w8,b):
    counter = 0
    for i in range(time_learn):
        #pick random datapoint
        y = int(numpy.random.randint(1,len(x1)))
        #retrieve prediction of datapoint

        z=((x1[y] * w1) + (x2[y] * w2) + (x3[y] * w3) + (x4[y] * w4) + (x5[y] * w4) + (x6[y] * w4) + (x7[y] * w4) + (x8[y] * w4) + b)
        
        prediction = Sigmoid(z)
        #retrieve cost of datapoint
        cost = (prediction - tar[y])**2
        #=============================
        #calculate slope of datapoint
        #=============================
        #start by deriv of cost in terms of pred, pred in terms of z, and z in terms of the weights

        #derive of cost with respect to pred
        dcost_dpred = 2 * (prediction - tar[y])

        #derive of pred with respect to the z (which is a function of sigmoid)
        dpred_dz = prediction * (1-prediction)
        #derive of z in terms of the weights and bias
        dz_dw1 = x1[y]
        dz_dw2 = x2[y]
        dz_dw3 = x3[y]
        dz_dw4 = x4[y]
        dz_dw5 = x5[y]
        dz_dw6 = x6[y]
        dz_dw7 = x7[y]
        dz_dw8 = x8[y]
        dz_db = 1
        
        #how does cost change when prediction change times how does the cost change 
        #when the z changes times how does the cost change when w1 changes
        dcost_dw1 = dcost_dpred * dpred_dz * dz_dw1
        dcost_dw2 = dcost_dpred * dpred_dz * dz_dw2
        dcost_dw3 = dcost_dpred * dpred_dz * dz_dw3
        dcost_dw4 = dcost_dpred * dpred_dz * dz_dw4
        dcost_dw5 = dcost_dpred * dpred_dz * dz_dw5
        dcost_dw6 = dcost_dpred * dpred_dz * dz_dw6
        dcost_dw7 = dcost_dpred * dpred_dz * dz_dw7
        dcost_dw8 = dcost_dpred * dpred_dz * dz_dw8
        dcost_db = dcost_dpred * dpred_dz * dz_db

        #update perameters
        w1 -= learning_rate * dcost_dw1
        w2 -= learning_rate * dcost_dw2
        w3 -= learning_rate * dcost_dw3
        w4 -= learning_rate * dcost_dw4
        w5 -= learning_rate * dcost_dw5
        w6 -= learning_rate * dcost_dw6
        w7 -= learning_rate * dcost_dw7
        w8 -= learning_rate * dcost_dw8
        b -= learning_rate * dcost_db
        counter +=1
        percent_complete=round(counter*100/time_learn, 4)
        if (counter%(time_learn/100) == 0):
           print("Im Running: " + str(percent_complete) + " Percent Complete")
    return w1, w2, w3, w4, w5,w6,w7,w8, b

def NN_Test(time_test,x1,x2,x3,x4,x5,x6,x7,x8,tar,w1,w2,w3,w4,w5,w6,w7,w8,b):
    test_counter = 0
    wrong_count = 0
    while(test_counter < time_test):
        y = int(numpy.random.randint(1,len(x1)))
        prediction = Sigmoid((x1[y] * w1) + (x2[y] * w2) + (x3[y] * w3) + (x4[y] * w4) + (x5[y] * w4) + (x6[y] * w4) + (x7[y] * w4) + (x8[y] * w4) + b)
        if(prediction>.5):
            prediction = 1
        else:
            prediction = 0
        wrong_count += abs(prediction-tar[y])
        test_counter+=1
    print("Percent Wrong: " + str(wrong_count/test_counter))
    return wrong_count/test_counter
def NN_Numbergen(excel_data, col1, col2, col3, col4, col5,col6,col7,col8,col9):
    w1 = numpy.random.randn()
    w2 = numpy.random.randn()
    w3 = numpy.random.randn()
    w4 = numpy.random.randn()
    w5 = numpy.random.randn()
    w6 = numpy.random.randn()
    w7 = numpy.random.randn()
    w8 = numpy.random.randn()
    b = numpy.random.randn()
    x1 = excel_data[col1]
    x2 = excel_data[col2]
    x3 = excel_data[col3]
    x4 = excel_data[col4]
    x5 = excel_data[col5]
    x6 = excel_data[col6]
    x7 = excel_data[col7]
    x8 = excel_data[col8]
    tar = excel_data[col9]
    return x1, x2, x3, x4,x5,x6,x7,x8, tar, w1, w2, w3, w4,w5,w6,w7,w8,b
def NN_Run(excel_data,col1 ,col2 ,col3 ,col4 ,col5,col6,col7,col8,col9, time_learn, time_test, learning_rate):
    #if(excel_data == "%" & col1 == "%" & col2 == "%" & col3 == "%" & col4 == "%" & col5 == "%" & time_learn > 0 & time_test > 0 & learning_rate > 0):
    #    print("\nRunning\n")
    #else:
    #    print("excel_data as pandas variable, col1 ,col2 ,col3 ,col4 ,col5, time_learn, time_test, learning_rate")
    percent_wrong = .5
    while(percent_wrong > .25):
        run_counter = 0
        print("Reseting weights")
        x1, x2, x3, x4, x5,x6,x7,x8,tar, w1, w2, w3, w4,w5,w6,w7,w8, b = NN_Numbergen(excel_data, col1, col2, col3, col4, col5, col6, col7, col8, col9)
        while(run_counter < 100 ):
            print("\n\n=====Training=====\n\n")
            w1,w2,w3,w4,w5,w6,w7,w8,b = NN_Train(time_learn,learning_rate,x1,x2,x3,x4,x5,x6,x7,x8,tar,w1,w2,w3,w4,w5,w6,w7,w8,b)
            print("\n\n=====Testing=====\n\n")
            percent_wrong = NN_Test(time_test,x1,x2,x3,x4,x5,x6,x7,x8,tar,w1,w2,w3,w4,w5,w6,w7,w8,b)
            run_counter+=1
    print("w1: " + str(w1))
    print("w2: " + str(w2))
    print("w3: " + str(w3))
    print("w4: " + str(w4))
    print("b: " + str(b))
    return
