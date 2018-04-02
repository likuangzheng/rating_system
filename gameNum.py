import pandas as pd
import numpy as np
import os
import json
from pandas import DataFrame, Series
import random

def get_user_comparsion(res, u1, u2):
    """u1:standard player
    compare u2 with u1"""
    u1_res, u2_res = res[u1], res[u2]
    u2_win = (u2_res > u1_res)
    draw = (u2_res == u1_res)
    u2_lose = (u2_res < u1_res)
    u2_to_u1 = u2_win + draw*0 - u2_lose
    return u2_to_u1

def get_all_players_comparsion(res, u1):
    """u1:standard player
    compare all players with u1
    """
    other_p = [p for p in res.columns if p != u1]
    all_res = pd.concat([get_user_comparsion(res, u1, p) for p in other_p], axis=1)
    #all_res = all_res.sample(frac=1)
    all_res.index = range(len(all_res))
    return all_res

def get_corr_trend(res):
    u1 = random.choice(res.columns)
    all_com = get_all_players_comparsion(res, u1)
    cum = all_com.cumsum()
    games = (all_com>-10).cumsum()
    average_score = cum / games
    #average_score.plot()
    game_process = average_score.T
    corr_ls = []
    for i in range(len(game_process.columns)-5):
        l = game_process[i]
        time_corr = []
        for k in range(1,5):
            n = game_process[i+k]
            c = l.corr(n,method='kendall')
            time_corr.append(c)
        min_corr = sum(time_corr)/4
        corr_ls.append(min_corr)
    return np.array(corr_ls)

def get_cut_num(corr_ls, q):
    for i , _ in enumerate(corr_ls):
        if (corr_ls[i:]>q).all():
            return(i)
        else:
            continue

def get_ample_cut_nums(res,q):
    cut_num = []
    for i in range(1000):
        corr_ls = get_corr_trend(res)
        cut_num.append(get_cut_num(corr_ls,q))
    return np.array(cut_num)

if __name__ == '__main__':
    seasonLog = pd.read_json("/Users/kuangzheng/Programming/GameAI/Data/log_data/SpringCup.json", lines=True)
    seasonLog['time'] = seasonLog['time'].str.rsplit(' ',expand=True)[0]
    seasonLog = seasonLog[(seasonLog['gameStatus'] == 1) | (seasonLog['gameStatus']== 3)]
    seasonLog_1 = seasonLog[seasonLog.seasonId == '5885e0c6684a332c58913c05']
    seasonLog_2 = seasonLog[seasonLog.seasonId == '5885e1000d2b3a7658b3e4b4']
    def get_full_game_player(sl):
        full_game_player = sl.groupby(['user']).time.count()
        games = sl[['seasonId','sessionId','gameId']].drop_duplicates()
        full_game_player = full_game_player[full_game_player == len(games)].index
        sl = sl[sl.user.isin(full_game_player)]
        return sl
    seasonLog_1 = get_full_game_player(seasonLog_1)
    seasonLog_2 = get_full_game_player(seasonLog_2)
    seasonLog_all = get_full_game_player(pd.concat([seasonLog_1,seasonLog_2]))
    res = seasonLog_all.set_index(['seasonId', 'sessionId', 'gameId', 'user']).gameResult
    res = res.unstack(level='user')
    cut_nums_75 = get_ample_cut_nums(res, 0.975)


