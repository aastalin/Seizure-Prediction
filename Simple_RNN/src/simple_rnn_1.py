"""
Data handler for EEG competition at Kaggle
Written by Aasta Lin
BSD License
"""
import scipy.io as sio   
import numpy as np
import pickle

target = 1

# hyperparameters
feature_size = 16
output_size = 2
hidden_size = 100
seq_size = 240000
learning_rate = 0.01

# model parameters
Wxh = np.random.randn(hidden_size, feature_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(output_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((output_size, 1)) # output bias


def lossFun(data, label):
  hs, dt = {}, {}
  hs[-1] = np.zeros((hidden_size,1))

  # forward pass
  ts = np.zeros((output_size,1))
  ts[label] = 1
  for t in xrange(seq_size):
    dt[t] = np.reshape(data[t], (feature_size,1))
    hs[t] = np.tanh(np.dot(Wxh, dt[t]) + np.dot(Whh, hs[t-1]) + bh)
  ys = np.dot(Why, hs[t]) + by
  ps = np.exp(ys) / np.sum(np.exp(ys))
  loss = -np.log(np.dot(ts.T,ps))

  # backward pass
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  dy = np.copy(ps)
  dy[label] -= 1
  dWhy = np.dot(dy, hs[seq_size-1].T)
  dby = dy
  dhnext = np.dot(Why.T, dy)
  for t in reversed(xrange(seq_size)):
    dh = dhnext
    dhraw = (1 - hs[t] * hs[t]) * dh
    dbh += dhraw
    dWxh += np.dot(dhraw, dt[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam)
  return loss, dWxh, dWhh, dWhy, dbh, dby


def readInput(filename):
  ptr = sio.loadmat('data/'+filename)
  raw = ptr['dataStruct']
  raw = raw[0,0]
  data = raw['data']
  return data


fp = open('data/train%d_pos.list' % target, 'r')
fn = open('data/train%d_neg.list' % target, 'r')
dp = fp.readlines()
dn = fn.readlines()
fp.close()
fn.close()

nump = len(dp)
numn = len(dn)

n = 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)
smooth_loss = -np.log(1.0/feature_size)

while True:
  #positive pass
  name = dp[n%nump].split('\n')[0]
  data = readInput(name)
  loss, dWxh, dWhh, dWhy, dbh, dby = lossFun(data, 1)
  print "[Train %d]%s: %f" % (target, name, loss)

  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

  #negative pass
  name = dn[n%numn].split('\n')[0]
  data = readInput(name)
  loss, dWxh, dWhh, dWhy, dbh, dby = lossFun(data, 0)
  print "[Train %d]%s: %f" % (target, name, loss)

  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

  # dump weight
  if (n+1) % nump == 0:
    with open('weight_%d_%d.pickle' % (target,n), 'w') as f:
      pickle.dump([Wxh, Whh, Why, bh, by], f)

  n += 1
