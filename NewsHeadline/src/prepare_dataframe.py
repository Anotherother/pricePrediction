
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
    dataframe['time'] = dataframe.index.copy()
    dataframe.index = dataframe.index.rename('index')
    d_range = dataframe.index.unique().floor('d')

    returned_df = {'date': [], 'neg': [], 'neu': [], 'pos': []}
    compound_df = {'date': [], 'neg': [], 'neu': [], 'pos': []}

    for i in d_range:

        df_new = dataframe[(dataframe['time'] >= i) \
                         & (dataframe['time'] < i + dt.timedelta(1))].mean()

        returned_df['date'].append(i)
        returned_df['neg'].append(df_new['negative'])
        returned_df['neu'].append(df_new['neutral'])
        returned_df['pos'].append(df_new['positive'])

        # Если нужно будет заполнять метками по compound = раскоментить
        c_df = dataframe[(dataframe['time'] >= i) \
                           & (dataframe['time'] < i + dt.timedelta(1))]

        compound_df['date'].append(i)

        compound_df['neg'].append(\
            c_df[c_df["compound"] < -0.5].shape[0] / c_df.shape[0])
        compound_df['neu'].append(\
            c_df[(c_df["compound"] > -0.5) & (c_df["compound"] < 0.5)].shape[0] / c_df.shape[0])
        compound_df['pos'].append(\
            c_df[c_df["compound"] > 0.5].shape[0] / c_df.shape[0])

    returned_df = pd.DataFrame(returned_df)
    compound_df = pd.DataFrame(compound_df)
    return returned_df, compound_df