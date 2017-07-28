import numpy as np
"""
def load_data(X, seq_len, train_size=1):

    amount_of_features = X.shape[1]
    X_mat = np.array(X)

    data = []

    for index in range(len(X_mat)  - seq_len):
        data.append(X_mat[index: index + seq_len +1])

    data = np.array(data)
    train_split = int(round(train_size * data.shape[0]))
    train_data = data[:train_split, :]

    x_train = train_data[:, :-1]
    y_train = train_data[:, -1][:,-1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))

    return x_train, y_train
"""
def load_data(data, seq_len, train_size = 1, TrainTest = False ):
    
    amount_of_features = data.shape[1] 
    X_mat = np.array(data)
    
    sequence_length = seq_len + 1 
    frames = []
    

    for index in range(len(X_mat) + seq_len):
        frames.append(X_mat[index: index + seq_len + 1])
    
    frames = np.array(frames)
    

    
    if TrainTest == False:
        x_train = frames
        y_train = frames[:, -1][:,-1]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
        
        return x_train, y_train
    
    if TrainTest == True:
        x_train = train_data[:, :-1]
        y_train = train_data[:, -1][:,-1]
        train_split = int(round(train_size * data.shape[0]))
        x_test = data[train_split:, :-1] 
        y_test = data[train_split:, -1][:,-1]
        
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features)) 
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))

        return x_train, y_train, x_test, y_test


def LoadData(data, train_size = 0.9, n_day = 1, window = 22, TestTrain = False):
    """
    data - source data
    train size - how much data u will be use for split on train and test batches
    n_day - numbers of day for prediction. default 0
    window - size of secuence for LSTM model
    TestTrain - trigger
    """
    amount_of_features = data.shape[1]
    data_mat = np.array(data)

    frames = []
    for index in range(len(data_mat) - window):
        frames.append(data_mat[index: index + window ])

    frames = np.array(frames)
    if TestTrain == False:
        x  = frames[:-n_day,:,:-1] # delete feature, which we want predict and add it to
                                    # labels - y_train

        temp_y  = frames[:, -1][:,-1]   # last index - index with value,
                                        # which we want predict. 4 - index for 'close'

        y = []
        for index in range(len(temp_y) - n_day):
            y.append(temp_y[index: index + n_day +1 ])
        y = np.array(y)

        return x,y

    train_split = int(round(train_size * data.shape[0]))

    if TestTrain == True:

        x_train = frames[:train_split,:,:-1]
        x_test = frames[train_split:-n_day,:,:-1]



        temp_y  = frames[:, -1][:,-1]   # lat index - index with value,
                                        # which we want predict. 4 - index for 'close'

        y = []
        for index in range(len(temp_y) - n_day):
            y.append(temp_y[index: index + n_day ])
        y = np.array(y)

        y_train = y[:train_split,:n_day,4]
        y_test = y[train_split:-n_day,:n_day,4]

        #print (x_train.shape)
        #print (x_test.shape)
        return x_train,x_test, y_train, y_test
    

