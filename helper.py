import numpy as np
import pandas as pd
from tqdm import *
import time
import scipy.stats as st


def get_pmf(x):
    y = x.values / x.values.sum()
    pmf = st.rv_discrete(a=x.index.min(), b=x.index.max(),
                         values=(x.index.tolist(), y))
    return pmf


def get_weekofyear(x):
    '''
    the weekofyear of datetime return strange thing due
    to the ISO format
    '''
    return int(time.strftime("%U", x.timetuple()))


def dump_login_web():
    N = 2683344
    login_web = []
    for chunk in tqdm(web, total=int(N / 1e5)):
        chunk = chunk.dropna(subset=['user_id'])
        login_web.append(chunk)
    login_web = pd.concat(login_web)
    login_web.to_csv('login_web.csv', index=False)


def dump_notuser_web():
    N = 2683344
    notusers = []
    users = pd.read_csv('login_web.csv')
    network_userids = users.network_userid.unique()
    domain_userids = users.domain_userid.unique()
    for chunk in tqdm(web, total=int(N / 1e5)):
        bool_network = chunk.network_userid.isin(network_userids)
        bool_domain = chunk.domain_userid.isin(domain_userids)
        is_user = bool_domain | bool_network
        notuser = chunk[~is_user]
        notusers.append(notuser)
    notusers = pd.concat(notusers)
    notusers.to_csv('notusers.csv', index=False)


def find_first_date(x):
    x = x.sort_values('collector_tstamp')
    return x.collector_tstamp.iloc[0]


def find_period_of_activity(x):
    x = x.sort_values('collector_tstamp')
    date_in = pd.to_datetime(x.collector_tstamp.iloc[0])
    date_out = pd.to_datetime(x.collector_tstamp.iloc[-1])
    return (date_out - date_in).days


def find_info_conection(df):
    df.loc[:, 'collector_tstamp'] = pd.to_datetime(df.collector_tstamp)
    df = df.sort_values('collector_tstamp')
    df.loc[:, 'diff_con'] = df.collector_tstamp.diff()
    df.loc[:, 'new_connection'] = [f for f in map(
        lambda x:x.total_seconds() > 5 * 60 if type(x) == pd.tslib.Timedelta else True, df.diff_con)]
    df.loc[:, 'connection'] = np.cumsum(df.new_connection)
    return df


def find_mean_time_bet_connection(df):
    df = df.sort_values('collector_tstamp')
    mean_time_bet_connection = df[
        df.new_connection].collector_tstamp.diff().mean()
    try:
        mean_time_bet_connection = mean_time_bet_connection.days
    except:
        mean_time_bet_connection = -1
    return mean_time_bet_connection


def find_mean_connection_time(df):
    mean_connection_time = df.groupby('connection').apply(
        lambda tmp: tmp.collector_tstamp.max() - tmp.collector_tstamp.min()).mean().seconds
    return mean_connection_time


def find_nb_of_event_by_connection(df):
    return df.groupby('connection').apply(lambda tmp: len(tmp.page_urlpath.unique())).mean()


def find_nb_of_event_by_connection_associated_with_keywords(df):
    return df.groupby('connection').apply(lambda tmp: len(tmp.page_urlpath.unique())).mean()


def dump_aquisition():
    print('dump_aquisition')
    login_web = pd.read_csv('login_web.csv')
    gp = login_web.groupby('user_id')
    aquisition = gp.apply(lambda df: find_first_date(df))
    aquisition = pd.DataFrame(
        aquisition, columns=['date_in']).reset_index('user_id')
    aquisition.index = pd.to_datetime(aquisition.date_in)
    aquisition['week'] = [get_weekofyear(f) for f in aquisition.index]
    aquisition.to_csv('aquisition.csv', index=False)


def dump_aquisition_by_channel():
    '''
    Dump aquisition by channel based on the assumition that the website
    was launch on the first of october 2014.
    '''
    print('dump_aquisition_by_channel')
    login_web = pd.read_csv('login_web.csv')
    customers = pd.read_csv('users',
                            header=None,
                            names=['user_id', 'channel'])
    login_web = login_web.merge(customers, on='user_id')
    gp = login_web.groupby(['user_id', 'channel'])
    channel_aquisition = gp.apply(
        lambda df: find_first_date(df)).reset_index(name='date_in')
    channel_aquisition['week'] = [get_weekofyear(
        f) for f in pd.to_datetime(channel_aquisition.date_in)]
    df = channel_aquisition.groupby(['week', 'channel']).size(
    ).reset_index(name='NbAquisition').set_index('week')
    df['channel'] = [f for f in map(lambda x:int(x.split('_')[1]), df.channel)]
    df.to_csv('aquisition_by_channel.csv', index=True)


def dump_activity_period():
    print('dump activity period')
    login_web = pd.read_csv('login_web.csv')
    gp = login_web.groupby('user_id')
    activity_period = gp.apply(lambda df: find_period_of_activity(df))
    activity_period = pd.DataFrame(activity_period, columns=[
        'period_of_activity']).reset_index('user_id')
    activity_period.to_csv('activity_period.csv', index=False)


def dump_info_connection():
    print('dump_info_connection')
    connection = pd.read_csv('connection_by_user.csv',
                             parse_dates=[0, 1, 2, 3, 4])
    connection = connection.sort_values('collector_tstamp')
    connection.diff_con = connection.collector_tstamp.diff()
    g1 = connection.groupby('user_id').apply(lambda x: find_mean_time_bet_connection(
        x)).reset_index(name='mean_time_bet_connection')
    g2 = connection.groupby('user_id').apply(
        lambda x: find_mean_connection_time(x)).reset_index(name='mean_connection_time')
    g3 = connection.groupby('user_id').apply(lambda x: find_nb_of_event_by_connection(
        x)).reset_index(name='mean_nb_of_event_by_connection')
    g = g1.merge(g2, on='user_id').merge(g3, on='user_id')
    g.to_csv('info_connections.csv', index=False)


def dump_connection_by_user():
    print('dump connection_by_user')
    login_web = pd.read_csv('login_web.csv').sort_values('user_id')
    df = pd.concat([find_info_conection(x) for _, x in tqdm(
        login_web.groupby('user_id'), total=len(login_web.user_id.unique()))])
    df.to_csv('connection_by_user.csv', index=False)


def dump_neworder():
    orders = pd.read_csv('orders',
                         header=None,
                         names=['id', 'user_id', 'period_id', 'sku', 'num'])
    periods = (orders.groupby('period_id').size()
               > 2000).reset_index(name='is_ok')
    norders = orders[orders.period_id.isin(
        periods[periods.is_ok].period_id.tolist())]
    norders.to_csv('good_orders.csv', index=False)


def detect_cookbook(df):
    df = df.dropna(subset=['page_urlpath'])
    df['is_in'] = ['cookbook' in f for f in df.page_urlpath]
    if len(df.is_in) == 0:
        return 0
    else:
        return np.sum(df.is_in) / float(len(df.is_in))


def dump_use_has_cookbook():
    orders = pd.read_csv('orders',
                         header=None,
                         names=['id', 'user_id', 'period_id', 'sku', 'num'])
    login_web = pd.read_csv('login_web.csv')
    d = login_web[~login_web.user_id.isin(orders.user_id)]
    cook = d.groupby('user_id').apply(
        lambda df: detect_cookbook(df)).reset_index(name='proportion')
    cook.to_csv('use_has_cookbook.csv')


def dump_nconnection():

    orders = pd.read_csv('orders',
                         header=None,
                         names=['id', 'user_id', 'period_id', 'sku', 'num'])
    connection = pd.read_csv('connection_by_user.csv',
                             parse_dates=[0, 1, 2, 3, 4])
    connection['week'] = [get_weekofyear(f)
                          for f in connection.collector_tstamp]

    nconnectionbyweek = connection.groupby(
        ['user_id', 'week']).new_connection.sum().reset_index()
    nconnectionbyweek['has_order'] = nconnectionbyweek.user_id.isin(
        orders.user_id)
    nconnection = connection.groupby(
        'user_id').new_connection.sum().reset_index(name='N')
    nconnection['has_order'] = nconnection.user_id.isin(orders.user_id)
    nconnection.to_csv('nconnection.csv', index=False)
    nconnectionbyweek.to_csv('nconnectionbyweek.csv', index=False)

if __name__ == '__main__':

    orders = pd.read_csv('orders',
                         header=None,
                         names=['id', 'user_id', 'period_id', 'sku', 'num'])
    customers = pd.read_csv('users',
                            header=None,
                            names=['user_id', 'channel'])
    web = pd.read_csv('web', header=None, error_bad_lines=False, iterator=True, chunksize=int(1e5),
                      names=['event_id', 'collector_tstamp', 'domain_userid', 'network_userid',
                             'user_id', 'page_urlpath', 'refr_medium', 'refr_source', 'mkt_medium',
                             'mkt_source', 'useragent'])
    # Get only user that register
    dump_notuser_web()
    dump_use_has_cookbook()
    dump_login_web()
    dump_aquisition()
    dump_aquisition_by_channel()
    dump_activity_period()
    dump_connection_by_user()
    dump_info_connection()
    dump_nconnection()
