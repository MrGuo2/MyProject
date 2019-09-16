from __future__ import division

import itertools
import pickle
import numpy as np 
import scipy.io as sio
import scipy.sparse as ss 
import scipy.spatial.distance as ssd

from collections import defaultdict
from sklearn.preprocessing import normalize

dpath = "../Data/"
class ProgramEntities:
    '''
    User和Song的字典  
    打分矩阵
    '''
    def __init__(self):
        # 创建user和song的set
        uniqueUsers = set()
        uniqueSongs = set()
        # 创建songsForUser和userForSongs的字典
        songsForUser = defaultdict(set)
        usersForSong = defaultdict(set)
        # 读取文件
        for filename in [dpath+"TrainMembers.csv", dpath+"TestMembers.csv"]:
            f = open(dpath+filename, 'r')
            f.readline().strip().split(",")
            for line in f:
                cols = line.strip().split(",")
                uniqueUsers.add(cols[0])
                uniqueSongs.add(cols[1])
                songsForUser[cols[0]].add(cols[1])
                usersForSong[cols[1]].add(cols[0])
        f.close()
        # 创建打分矩阵
        self.userSongScores = ss.dok_matrix((len(uniqueUsers), len(uniqueSongs)))
        # 创建以User/song为索引的字典
        self.userIndex = dict()
        self.songIndex = dict()
        for i, u in enumerate(uniqueUsers):
            self.userIndex[u] = i
        for i, s in enumerate(uniqueSongs):
            self.songIndex[s] = i

        ftrain = open(dpath+"TrainMembers.csv", "r")
        ftrain.readlines()
        for line in ftrain:
            cols = line.strip().split(",")
            i = self.userIndex[cols[0]]
            j = self.songIndex[cols[1]]
            # user对应song的得分
            self.userSongScores[i, j] = int(cols[5])
        ftrain.close()
        sio.mmwrite(dpath+"PE_userSongScores", self.userSongScores)
        
        # 找出有关联的用户或者有关联的歌曲
        self.uniqueUserPairs = set()
        self.uniqueSongPairs = set()
        # 如果这首歌的用户大于2那么
        for song in uniqueSongs:
            users = usersForSong[song]
            if len(users) > 2:
                # combinations()排列组合 update()字典的添加
                self.uniqueUserPairs.update(itertools.combinations(users, 2))
        # 如果这个用户听的歌大于2那么
        for user in uniqueUsers:
            songs = songsForUser[user]
            if len(songs) > 2:
                self.uniqueSongPairs.update(itertools.combinations(songs, 2))
        pickle.dump(self.userIndex, open(dpath+"PE_userIndex.pkl", "wb"))
        pickle.dump(self.songIndex, open(dpath+"PE_songIndex.pkl", "wb"))

class Users:
    '''
    user/user相似度矩阵
    '''
    def __init__(self, ProgramEntities, sim=ssd.correlation):
        # cleaner = DataCleaner()
        nusers = len(ProgramEntities.userIndex.keys())
        fin = open(dpath+"TrainMembers.csv", "r")
        colnames = fin.readline().strip().split(",")
        self.userMatrix = ss.dok_matrix((nusers, len(colnames)-1))
        for line in fin:
            cols = line.strip().split(",")
            i = ProgramEntities.userIndex[cols[0]]
            self.userMatrix[i, 0] = cols[2] # source_system_tab
            self.userMatrix[i, 1] = cols[3] # source_screen_name
            self.userMatrix[i, 2] = cols[4] # source_type
            self.userMatrix[i, 3] = cols[6] # city
            self.userMatrix[i, 4] = cols[7] # bd
            self.userMatrix[i, 5] = cols[8] # gender
            self.userMatrix[i, 6] = cols[9] # registered_via
        fin.close()
        sio.mmwrite(dpath+"US_userMatrix", self.userMatrix)

        # 计算用户相似度矩阵，之后会用到
        self.userSimMatrix = ss.dok_matrix((nusers, nusers))
        for i in range(nusers):
            self.userSimMatrix[i, i] = 1.0
        for u1, u2 in ProgramEntities.uniqueUserPairs:
            i = ProgramEntities.userIndex[u1]
            j = ProgramEntities.userIndex[u2]
        if not self.userSimMatrix.has_key((i, j)):
            usim = sim(self.userMatrix.getrow(i).todense(),
            self.userMatrix.getrow(j).todense())
            self.userSimMatrix[i, j] = usim
            self.userSimMatrix[j, i] = usim
        sio.mmwrite(dpath+"US_userSimMatrix", self.userSimMatrix)
class Songs:
  """
  构建song-song相似度，注意这里有2种相似度：  
  1）由用户-songs行为，类似协同过滤算出的相似度  
  2）由song本身的内容(song信息)计算出的song-song相似度
  """
  def __init__(self, programEntities, psim=ssd.correlation, csim=ssd.cosine):
    fin = open(dpath+"TrainSongs.csv", 'r')
    fin.readline()
    nsongs = len(programEntities.songIndex.keys())
    self.songProgMatrix = ss.dok_matrix((nsongs, 7))
    self.songContMatrix = ss.dok_matrix((nsongs, nsongs))
    # song-song相似度
    for line in fin.readlines():
        cols = line.strip().split(",")
        songId = cols[1]
        i = programEntities.songIndex[songId]
        self.songProgMatrix[i, 0] = cols[2] # source_system_tab
        self.songProgMatrix[i, 1] = cols[3] # source_screen_name
        self.songProgMatrix[i, 2] = cols[4] # source_type
        self.songProgMatrix[i, 3] = cols[6] # song_length
        self.songProgMatrix[i, 4] = cols[7] # artist_name
        self.songProgMatrix[i, 5] = cols[8] # language
        self.songProgMatrix[i, 6] = cols[9] # genre_ids
        for j in range(nsongs):
          self.songContMatrix[i, j] = cols[5]
    fin.close()
    sio.mmwrite(dpath+"SO_songProgMatrix", self.songProgMatrix)
    sio.mmwrite(dpath+"SO_songContMatrix", self.songContMatrix)

    # calculate similarity between song pairs based on the two matrices    
    self.songProgSim = ss.dok_matrix((nsongs, nsongs))
    self.songContSim = ss.dok_matrix((nsongs, nsongs))
    for s1, s2 in programEntities.uniquesongPairs:
        i = programEntities.songIndex[s1]
        j = programEntities.songIndex[s2]
        if not self.songProgSim.has_key((i,j)):
            spsim = psim(self.songProgMatrix.getrow(i).todense(), 
            self.songProgMatrix.getrow(j).todense())
        self.songProgSim[i, j] = spsim
        self.songProgSim[j, i] = spsim
        if not self.songContSim.has_key((i,j)):
            scsim = csim(self.songContMatrix.getrow(i).todense(),
            self.songContMatrix.getrow(j).todense())
        self.songContSim[i, j] = spsim
        self.songContSim[j, i] = scsim
    sio.mmwrite(dpath+"SO_songProgSim", self.songProgSim)
    sio.mmwrite(dpath+"SO_songContSim", self.songContSim)

def data_prepare():
    """
    计算生成所有的数据，用矩阵或者其他形式存储方便后提取特征和建模
    """
    print ("第一步：统计user和song的相关信息")
    pe = ProgramEntities()
    print ("第一步完成...\n")
    print ("第二步: 计算用户相似度信息，并用矩阵形式存储...")
    Users(pe)
    print ("第二步完成...\n")
    print ("第三步: 计算歌曲相似度信息，并用矩阵形式存储...")
    Songs(pe)
    print ("第三步完成...\n")
# 运行进行数据准备
data_prepare()