"""
Data handler for EEG competition at Kaggle
Written by Aasta Lin
BSD License
"""
import scipy.io as sio   
import numpy as np
import pickle

target = 1

def readInput(filename):
  ptr = sio.loadmat('data/'+filename)
  raw = ptr['dataStruct']
  raw = raw[0,0]
  rate = raw['iEEGsamplingRate'][0][0]
  num  = raw['nSamplesSegment'][0][0]
  idex = raw['channelIndices']
  data = raw['data']
  return data


fp = open('data/test%d.list' % target, 'r')
dp = fp.readlines()
fp.close()

print 'num: %d' % len(dp)
for d in dp:
  name = d.split('\n')[0]
  data = readInput(name)
  print name
