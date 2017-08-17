
import pandas as pd
import datetime as dt

def read_csv_dataframe(filename):

    data = pd.read_csv(filename)
    data = data.dropna()
    data = data.reset_index()

    data.time = pd.to_datetime(data['time'])
    data.title  = data.title.astype('str')
    data = data[['time', 'title', 'url']]
    return data

def join_SA_and_data(data, sent_data):
    result = data.join(sent_data)
    result.index = data.time
    result['dayOfWeek'] = result.index.weekday_name
    result = result[['dayOfWeek', 'title', 'url', \
                     'neg', 'neu', 'pos', 'compound']]
    return result


def averageSintimentEveryDay(dataframe):
    dataframe['time'] =  dataframe.index
    dataframe.index = dataframe.index.rename('index')
    d_range = dataframe.index.unique().floor('d')

    returned_df = {'date': [], 'neg': [], 'neu': [], 'pos': []}

    for i in d_range:

        df_new = dataframe[(dataframe['time'] >= i) \
                         & (dataframe['time'] < i + dt.timedelta(1))].mean()

        returned_df['date'].append(i)
        returned_df['neg'].append(df_new['neg'])
        returned_df['neu'].append(df_new['neu'])
        returned_df['pos'].append(df_new['pos'])

        # Если нужно будет заполнять метками по compuund = раскоментить

        #returned_df['neg'].append(\
        #    df_new[df_new["compound"] < -0.5].shape[0] / df_new.shape[0])
        #returned_df['neu'].append(\
        #    df_new[(df_new["compound"] > -0.5) & (df_new["compound"] < 0.5)].shape[0] / df_new.shape[0])
        #returned_df['pos'].append(\
        #    #df_new[df_new["compound"] > 0.5].shape[0] / df_new.shape[0])

    return pd.DataFrame(returned_df)