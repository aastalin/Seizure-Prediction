"""
Data handler for EEG competition at Kaggle
Written by Aasta Lin
BSD License
"""
import scipy.io as sio   
import numpy as np
import pickle
import csv

wptr = open('test_response.csv',"w")
writer = csv.writer(wptr)
writer.writerow(["File","Class"])

# hyperparameters
feature_size = 16
output_size = 2
hidden_size = 100
seq_size = 240000
res_size = 400

def predictFun(data):
  hs, dt = {}, {}
  hs[-1] = np.zeros((hidden_size,1))
  res = hs[-1]

  # forward pass
  for t in xrange(seq_size):
    dt[t] = np.reshape(data[t], (feature_size,1))
    hs[t] = np.tanh(np.dot(Wxh, dt[t]) + np.dot(Whh, hs[t-1]) + bh)
    if (t+1) % res_size == 0:
      rid = (t+1) / res_size
      hs[t] += res
      res = hs[t]
  ys = np.dot(Why, hs[t]) + by
  ps = np.exp(ys) / np.sum(np.exp(ys))
  return np.argmax(ps)

def readInput(filename):
  ptr = sio.loadmat('data/'+filename)
  raw = ptr['dataStruct']
  raw = raw[0,0]
  data = raw['data']
  return data

# test-1
with open('res_1_148.pickle') as f:
  Wxh, Whh, Why, bh, by = pickle.load(f)

fp = open('data/test1.list', 'r')
dp = fp.readlines()
fp.close()

for d in dp:
  name = d.split('\n')[0]
  data = readInput(name)
  value = predictFun(data)
  writer.writerow([name.split('/')[1],value])

# test-2
with open('res_2_149.pickle') as f:
  Wxh, Whh, Why, bh, by = pickle.load(f)

fp = open('data/test2.list', 'r')
dp = fp.readlines()
fp.close()

for d in dp:
  name = d.split('\n')[0]
  data = readInput(name)
  value = predictFun(data)
  writer.writerow([name.split('/')[1],value])

# test-3
with open('res_3_149.pickle') as f:
  Wxh, Whh, Why, bh, by = pickle.load(f)

fp = open('data/test3.list', 'r')
dp = fp.readlines()
fp.close()

for d in dp:
  name = d.split('\n')[0]
  data = readInput(name)
  value = predictFun(data)
  writer.writerow([name.split('/')[1],value])
