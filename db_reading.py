from bson.objectid import ObjectId
import pandas as pd
from pandas import DataFrame, Series
from pymongo import MongoClient
import json
import datetime

def generate_ana_table(jj, seasonId):
    season = jj['seasons']#[ObjectId('58297abea723c1077adf5d8b')]
    seasonId = ObjectId(seasonId)
    season_info = season.find({"_id":seasonId})#, 'status':'FINISHED'})
    sessions = season_info[0]['sessions']
    #有没有问题啊
    history_db = jj['gamehistories']
    game_db = jj['games']
    table = []
    for ses in sessions:
        games = ses['games']
        for g in games:
            g_info = list(game_db.find({'_id':g}))[0]
            peoples = g_info['people']
            g_config = g_info['gameConfig']
            startbid = list(jj['gameconfigs'].find({'_id':g_config}))[0]
            startbid = startbid['bidStartPosition']

            for p in peoples:
                #if p['status'] == 'FINISHED':
                hist_id = p['history']
                h = history_db.find({"_id" : hist_id})
                end = json.loads(list(h)[0]['rawInput'])['endAt']
                end = datetime.datetime.fromtimestamp(end/1000)
                res = {'seasonId':str(seasonId),
                       'sessonId':str(ses['_id']),
                       'gameId':str(g),
                       'user':p['uid'],
                       'bid':p['bid'],
                       'bidStartPosition':startbid,
                       'bombs':p['bombs'],
                       'spring':p['spring'],
                       'gameResult':p['score'],
                       'hist_id':str(p['history']),
                       'role':p['role'],
                       'endTime':end,
                       'status':p['status']
                        }
                table.append(res)
    return DataFrame(table)

if __name__ == '__main__':
    mc = MongoClient()
    jj = mc['jj-production']
    seasonId = ObjectId('58297abea723c1077adf5d8b')
    db_res = generate_ana_table(jj, seasonId)
    db_res.to_csv("/Users/kuangzheng/Programming/GameAi/Data/20180402/seasonResult.csv")
