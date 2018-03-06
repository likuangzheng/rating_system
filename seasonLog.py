import pandas as pd
import numpy as np
import os
import json
from pandas import DataFrame, Series
from collections import defaultdict
from math import log
from sklearn.cluster import DBSCAN,AgglomerativeClustering

class SeasonLogManagement():
    """处理SeasonLog处理，转换分数等
    """
    def __init__(self, seasonLog):
        self.log = seasonLog

    def _get_game_score_distribution(self):
        """计算每个分数的分布"""
        grouped = self.log.groupby(['seasonId', 'sessionId', 'gameId', 'gameResult'])
        game_user_num = grouped.user.count().rename('userNum')
        game_user_num = game_user_num.unstack().fillna(0)
        game_score_dict = {}
        for _, l in game_user_num.iterrows():
            game_score_dict[_] = l.rename('userNum')
        return game_score_dict

    def _get_player_game_score(self, seasonId, sessionId, gameId, game_score_dict, game_result):
        """
        Params:
        ------
        seasonId:赛季编号
        sessionId:赛事编号
        gameId:game编号
        game_score_dist:每个game各个分数玩家人数的分布, dict

        Returns:
        ------
        score:某实际分在该game上评分是什么
        """
        game_score = game_score_dict[(seasonId, sessionId, gameId)]
        num = game_score.sum() - 1
        val_num_1 = game_score[game_score.index < game_result].sum()
        if val_num_1 is np.nan:
            val_num_1 = 0
        val_num_2 = (game_score[game_score.index == game_result].sum() - 1) / 2
        if num == 0:
            return 50
        else:
            score = (val_num_1 + val_num_2) / num
            score *= 100
            return score
    
    def get_all_player_game_scores(self):
        """给出每个玩家在每个game上的得分"""
        _game_score_dict = self._get_game_score_distribution()
        _game_player_scores = {}
        for _, l in self.log.iterrows():
            info = tuple([l.user, l.seasonId, l.sessionId, l.gameId])
            try:
                score = self._get_player_game_score(*info[1:], _game_score_dict, l.gameResult)
                _game_player_scores[info] = score
            except:
                pass
        player_scores = Series(_game_player_scores,name='game_score')
        player_scores.index.names = ['user', 'seasonId', 'sessionId', 'gameId']
        player_scores = player_scores.unstack(level='user')
        return player_scores

class RankModel():
    """根据已有分数计算
    Params:
    ------
    game_player_scores:每个game上各玩家的得分,DataFrame, 这样记录会慢很多呀可是。
    更好的话应该是传进来一个matrix,gameNum x playerNum
    """

    def __init__(self, game_player_scores, group_filter=False):
        
        self.game_player_scores = game_player_scores
        self.group_filter = group_filter
        self.useless_game = None

    def get_useful_game(self):
        """返回无效game的名称
        """
        useless_user = (self.game_player_scores == 50).all(axis=1)
        self.useless_game = [i for i in useless_user.index]
        filtered_game_scores = self.game_player_scores[~useless_user]
        return filtered_game_scores
        
    def get_ken_corr_of_games(self,filtered_game_scores):
        """compute_corr
        """
        if filtered_game_scores.index.name != 'user':
            s = filtered_game_scores.T
        corr = s.corr(method='kendall')
        return corr

    def get_player_std_score(self, game_scores):
        
        std_score = game_scores.std()
        upper = std_score.quantile(0.85)
        lower = std_score.quantile(0.15)

        def liner_transformation(s):
            
            if s> upper:
                return 0
            elif s < lower:
                return 100
            else:
                return (upper-s)/(upper-lower) * 100
        std_score = std_score.apply(liner_transformation)
        return std_score

    def classify_by_ken_corr(self, game_scores):
        
        model = AgglomerativeClustering(affinity='precomputed', linkage='average', n_clusters=5)
        corr = self.get_ken_corr_of_games(game_scores)
        dist = 1 - corr
        clf = model.fit_predict(dist)
        clf = Series(clf, index=corr.index, name='cluster_index')
        return clf

    def fit(self):
        """最后返回"""
        if  self.group_filter: 
            filtered_game_scores = self.get_useful_game()
            clf = self.classify_by_ken_corr(filtered_game_scores)
            cluster_count = clf.value_counts()
            max_cluster = cluster_count.idxmax()
            left_game = (clf == max_cluster)
            left_game = filtered_game_scores[left_game]
            player_score_std = filtered_game_scores.std()
            mean_score = left_game.mean()
            std_score = self.get_player_std_score(left_game)
            self.player_scores = 0.95 * mean_score + 0.05 * std_score
        else:
            self.player_scores = self.game_player_scores.mean()
        return self.player_scores

class AlgorithmTesting():
    
    def __init__(self, train_log, test_log, fil=False):
        self.train_log = train_log
        self.test_log = test_log
        self._train_p_scores = Series()
        self._test_gp_result_dict = {}
        self._test_g_users = {}
        self._filter = fil
    def _get_train_player_score(self):
        
        slm = SeasonLogManagement(self.train_log)
        game_player_scores = slm.get_all_player_game_scores()
        model = RankModel(game_player_scores, self._filter)
        self._train_p_scores = model.fit()

    def _get_test_game_player_result_dict(self):
        _gp_result = self.test_log.set_index(['seasonId', 'sessionId', 'gameId', 'user'])
        _gp_result = _gp_result.gameResult
        self._test_gp_result_dict = _gp_result.to_dict()
    
    def _get_game_player_dict(self):
        #生成每局game_player查询表,格式'seasonId', 'sessionId','gameId'
        game_users = self.test_log.set_index(['seasonId', 'sessionId', 'gameId']).user
        for ind in game_users.index:
            self._test_g_users[ind] = list(game_users.loc[ind])

    def _get_pairise_player_compare(self, seasonId, sessionId, gameId, user_1, user_2):
        #比较查询，返回两个值，是否合法，是否存在
        #本赛季result_dict
        #之前的p_season_scores
        game_user_1 = (seasonId, sessionId, gameId, user_1)
        game_user_2 = (seasonId, sessionId, gameId, user_2)
        try:
            s1 = self._train_p_scores[user_1]
            s2 = self._train_p_scores[user_2]
            score_diff = s1 - s2
            r1 = self._test_gp_result_dict[game_user_1]
            r2 = self._test_gp_result_dict[game_user_2]
            result_diff = r1 - r2
            if score_diff * result_diff > 0:
                return 1, 1
            elif result_diff == 0:
                return 0, 0
            else:
                return 1, 0
        except KeyError:
            return 0, 0

    def get_test_result(self):
        ##season log
        self._get_train_player_score()
        self._get_test_game_player_result_dict()
        self._get_game_player_dict()
        games = self.test_log[['seasonId', 'sessionId', 'gameId']].drop_duplicates()
        total_val, total_hit = 0, 0
        for g in games.itertuples(index=False, name=None):
            users = self._test_g_users[g]
            for i, u1 in enumerate(users):
                for j, u2 in enumerate(users[i+1:]):
                    val, hit = self._get_pairise_player_compare(*g, u1, u2)
                    total_val += val
                    total_hit += hit
        return total_val, total_hit, total_hit/total_val
        
