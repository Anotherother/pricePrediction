import numpy as np

# Прибыль моделек

def classifyALL(data):
    ## 2 - Buy Class, 1 - Sell Class, 0 - Hold Class
    
    label = []
    for i in range(len(data)):     
        if i ==0:
            label.append(2)
        else:
            price = data[i:i+2]
            label.append(2 * (price[-1] > (price[0])) + 1 * (price[-1] < (price[0])))
    label = np.array(label)
    label[len(data)-1] = 1
    return np.array(label)

def calcDOXOD(data, labels):
    s = 0
    buffer = 0
    for i in range (len(data)):
        if (i == 0):
            s = s - data[i]
            
        elif (labels[i] == 2 and labels[i-1] != 2 and buffer == 1):
            s= s - data[i]
            buffer = 0
            
        elif (labels[i] == 2 and labels[i-1] == 2):
            i+=1
        
            
        elif (labels[i] == 1 and buffer == 0):
            s= s + data[i]
            buffer = 1

        elif labels[i] == 0:
            pass
        
        if (i == len(data)):
            s= s + data[i]

    return s