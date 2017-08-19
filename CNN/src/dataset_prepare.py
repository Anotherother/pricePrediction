import random
import matplotlib.pylab as plt
import datetime


WINDOW=22
FORECAST=2
EMB_SIZE=10
STEP=1
TRAIN_TEST_PERCENTAGE=0.9
SAVE_NAME = "classification_model.hdf5"
LOAD_NAME = "classification_model.hdf5"
ENABLE_CSV_OUTPUT = 1
NAME_CSV = "classification"
TRAINING = 1
TESTING = 0
NUMBER_EPOCHS = 10
TRADING_DAYS = 5

def df_sample(df):
    print( '\nINSPECTING DATABASE..\n')
    print ('DATABASE SIZE [',len(df),']')
    
    print ('SAMPLE VALUES..')
    
    r = int(random.random()*len(df))
    sample = df[r:r+WINDOW]
    target = df[r:r+WINDOW+FORECAST]
    
    plt.plot(sample.open)
    plt.plot(sample.high)
    plt.plot(sample.low)
    plt.plot(target.close)
    plt.legend()
    #plt.plot(sample.Volume)
    plt.show()    

def df_plot(df):
    print ('\nINSPECTING DATABASE..\n')
    print ('DATABASE SIZE [',len(df),']')
    
    for col in df.columns:
        if col!='date'and col!='volume_x' and col !='volume_y':
            plt.plot(df[col])
    plt.show()

def df_dead(df):
    
    _is_dead = 0    
    print ('\nINSPECTING DATABASE..\n')
    print ('DATABASE SIZE [',len(df),']')
        
    deads = []
    
    for i in range(len(df.close.values)):
        if(df.close.values[i]==df.open.values[i]==df.high.values[i]==df.low.values[i]):
            if(_is_dead==0):
                deads.append(i)
            _is_dead = 1
        if(df.close.values[i]!=df.open.values[i]):
            if(_is_dead==1):
                deads.append(i)
            _is_dead = 0
        
    return deads

def health_check(df):
    flag=1
    deads = df_dead(df)
    long_deads, i =[], 0
    
    while i<len(deads)-1:
        dead_len = deads[i+1]-deads[i]
        
        if(dead_len>100):
            long_deads.append(deads[i])
            long_deads.append(deads[i+1])
            print ('WARNING long dead period [',deads[i],'] to [',deads[i+1],']')
            flag=0
        i=i+2
    
    print('Data from '+str(df.date[df.index[0]])+' to '+str( df.date[df.index[len(df.index)-1]]))
    print('..check completed')
    if flag:
        print ("no DEAD periods")
    return long_deads



def dataset(df, START=0, END=0, SLICING=0):
    print ('dataset preparation.. (cleaning size ', END - START,')')
    
    df = df[['open', 'close', 'low', 'high', 'volume', 'date']]
    
    if(SLICING):
        df = df[START:END]



    def clean_string(s):
        ss = '0'
        for i in range(len(s)):
            if s[i]!='\\':
                ss = ss+s[i]
        return float(ss)
    
    for i in range(START,END):
        df.loc[i, ('volume')] = clean_string(df.loc[i, ('volume')])
        if i%1000==0:
            print ('cleaning dataset ',i,'/',len(df.Volume))
   
    def find_hour(df, hour='07:00:00'):
        vec = []
        for i in range(0,len(df.date)):
            if(str(df.date[i])[11:19] == hour):
                weekday = df.date[i].weekday()
                vec.append(i)
        return vec 
    
    if (SLICING==0):
        STARTS = find_hour(df,'07:00:00')
        ENDS = find_hour(df,'23:59:00')
        
        return df, STARTS, ENDS
    
    else:
        print('dataset ready to use')
        return df



def dataset_check(df):
    df, STARTS, ENDS = dataset(df)
    health_check(df)
    print('\nDATASET NOT CLEANED\n')
    print(df.head())
    print(df.tail())
    return STARTS, ENDS



def full_df(STARTS, END, ASK_FNAME=ASK_FNAME, BID_FNAME=BID_FNAME):
    ask_df = []
    bid_df = []
    for i in range(0,len(STARTS)):
        ask_df.append(dataset(ASK_FNAME,STARTS[i],ENDS[i],SLICING=1))
        bid_df.append(dataset(BID_FNAME,STARTS[i],ENDS[i],SLICING=1))
            
    ask_df= pd.concat(ask_df)
    bid_df= pd.concat(bid_df)

    deads = health_check(bid_df)
    
    df = pd.merge(ask_df,bid_df, on="Datetime")
    
    for i in xrange(len(df)):
        df.loc[i, ('Datetime')] = datetime.datetime.strptime(df.loc[i, ('Datetime')], '%d.%m.%Y %H:%M:%S.%f')
        if i%1000==0:
                print ('parsing time ',i,'/',len(df.Datetime))
    
    print('\nDATASET CLEANED\n')
    print(df.head())
    print(df.tail())
    #df_plot(df)
    
    return df