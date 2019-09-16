from __future__ import division
import pickle
import numpy as np 
import scipy.io as sio

dpath = "../Data/"
class DataRewriter:
  def __init__(self):
    #读入数据做初始化
    self.userIndex = pickle.load(open(dpath+"PE_userIndex.pkl", 'rb'))
    self.songIndex = pickle.load(open(dpath+"PE_songIndex.pkl", 'rb'))
    self.userSongScores = sio.mmread(dpath+"PE_userSongScores").todense()
    self.userSimMatrix = sio.mmread(dpath+"US_userSimMatrix").todense()
    self.songProgSim = sio.mmread(dpath+"SO_songProgSim").todense()
    self.songContSim = sio.mmread(dpath+"SO_songContSim").todense()

  def userReco(self, userId, songId):
    """
    根据User-based协同过滤，得到song的推荐度
    """
    i = self.userIndex[userId]
    j = self.songIndex[songId]
    vs = self.userSongScores[:, j]
    sim = self.userSimMatrix[i, :]
    prod = sim * vs
    try:
      return prod[0, 0] - self.userSongScores[i, j]
    except IndexError:
      return 0

  def songReco(self, userId,songId):
    """
    构建song的协同过滤，注意这里有2种方法求出相似度：  
  1）由用户-songs行为，计算出song推荐度    
  2）由song本身的内容(song信息)计算出的song推荐度
    """
    i = self.userIndex[userId]
    j = self.songIndex[songId]
    js = self.userSongScores[i, :]
    psim = self.songProgSim[:, j]
    csim = self.songContSim[:, j]
    pprod = js * psim
    cprod = js * csim
    pscore = 0
    cscore = 0
    try:
      pscore = pprod[0, 0] - self.userSongScores[i, j]
    except IndexError:
      pass
    try:
      cscore = cprod[0, 0] - self.userSongScores[i, j]
    except IndexError:
      pass
    return pscore, cscore

  def rewriteData(self, start=1, train=True, header=True):
    """
    把前面user-based协同过滤和item-based协同过滤结合到一起,
    生成新的训练数据
    """
    fn = dpath+"TrainMember.csv" if train else dpath+"TestMember.csv"
    fin = open(fn, 'r')
    fout = open("CF_"+fn, "w")
    # write output header
    if header:
      ocolnames = ["user_reco", "song_p_reco", "song_c_reco"]
      if train:
        ocolnames.append(["target"])
      fout.write(",".join(ocolnames) + "\n")
      ln = 0
    for line in fin:
      ln += 1
      if ln < start:
        continue
      cols = line.strip().split(",")
      userId = cols[0]
      songId = cols[1]
      if ln%500000 == 0:
        print("%s:%d (userId, songId)=(%s, %s)" %(fn, ln, userId, songId))
      user_reco = self.userReco(userId, songId)
      song_p_reco, song_c_reco = self.songReco(userId, songId)
      ocols = [user_reco, song_p_reco, song_c_reco]
      if train:
        ocols.append(cols[5])
      fout.write(",".join(map(lambda x: str(x), ocols)) + "\n")
    fin.close()
    fout.close()

    
dr = DataRewriter()
print("生成训练数据...\n")
dr.rewriteData(train=True, start=2, header=True)
print("生成预测数据...\n")
dr.rewriteData(train=False, start=2, header=True)