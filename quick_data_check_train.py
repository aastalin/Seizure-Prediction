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
  seq  = raw['sequence'][0][0]
  data = raw['data']
  return data


fp = open('data/train%d_pos.list' % target, 'r')
fn = open('data/train%d_neg.list' % target, 'r')
dp = fp.readlines()
dn = fn.readlines()
fp.close()
fn.close()

print 'positive: %d' % len(dp)
for d in dp:
  #positive pass
  name = d.split('\n')[0]
  data = readInput(name)
  print name

print 'positive: %d' % len(dn)
for d in dn:
  #negative pass
  name = d.split('\n')[0]
  data = readInput(name)
  print name
