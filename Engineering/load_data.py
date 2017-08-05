import numpy as np

def load_data(X, seq_len, train_size=0.9, TrainTest = False):    
        
    # for x extract all data beside last 
    # (i.e. we must have last value for y in training)
    
    amount_of_features = X.shape[1] 
    
    X_mat = np.array(X)
    
    # make blocks of data
    
    data = []    
    for index in range(len(X_mat)  - seq_len):
        data.append(X_mat[index: index + seq_len +1])
    
    data = np.array(data)
    
    if TrainTest ==False:
        x_train = data[:, :-1] # delete feature, which we want predict and add it to
                                 # labels - y_train
            
        y_train = data[:, -1][:,-1] # last index - index with value,
                                    # which we want predict. 4 - index for 'close'

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))

        return x_train, y_train
        
    if TrainTest ==True:

        train_split = int(round(train_size * data.shape[0]))
        train_data = data[:train_split, :]

        x_train = train_data[:, :-1]
        y_train = train_data[:, -1][:,-1]

        x_test = data[train_split:, :-1] 
        y_test = data[train_split:, -1][:,-1]

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  


        return x_train[:,:,:-1], y_train, x_test[:,:,:-1], y_test