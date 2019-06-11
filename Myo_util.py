# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import RadioButtons
import collections
#import myo
import threading
import pandas as pd
from sklearn.model_selection import train_test_split
#myo.init()

#X_train, X_val = train_test_split(X_train, y_train, test_size = 0.2)
#%%

def val_(data):
    total_len = len(data)
    train = int(total_len*0.7)
    test = total_len - train
    train_data = data[:train]
    test_data = data[train:]
    return train_data, test_data

def preprocessing(data,test=False):
    #act_list = ['circle','rectange','triangle']
    idx = 0
    data_len = len(data)
    window_size = 10
    if test is True:
        window_size = 50
    CHUNK_SIZE = 50
    overlab_data = list()
    while True:
        
        last = min(idx + CHUNK_SIZE, data_len)        
        last2 = last +50
        data_x = data
        #data_y = label
        #print(data_len)
        #print('last2',last, 'idx:',idx)
        if last >= data_len:
            break
        data_batch_x = data_x[idx:last]
        #data_batch_y = data_y[idx:last]
        overlab_data.append(data_batch_x)    
        idx += window_size
        
    d = np.asarray(overlab_data)
    #print(d.shape())
    return d

def fit_data_label(data,label):
    pass


#class MyListener(myo.DeviceListener):
#
#  def __init__(self, queue_size=3000):
#    self.lock = threading.Lock()
#    self.emg_data_queue = collections.deque(maxlen=queue_size)
#    self.orientation_data_queue = collections.deque(maxlen=queue_size)
#    self.orientation = 0
#    self.acceleration = 0
#  def on_connect(self, device, timestamp, firmware_version):
#    device.set_stream_emg(myo.StreamEmg.enabled)
#
#  def on_emg_data(self, device, timestamp, emg_data):
#    with self.lock:
#      self.emg_data_queue.append((timestamp, emg_data))
#
#  def on_orientation_data(self, myo, timestamp, orientation):
#      roll, pitch, yaw = orientation.roll, orientation.pitch, orientation.yaw
#      orien_x, orien_y , orien_z, orien_w = orientation.x, orientation.y, orientation.z, orientation.w
#      b.append([orien_x, orien_y, orien_z,orien_w, roll, pitch, yaw])
#        #self.acceleration = acceleration
#        #acc.append(self.acceleration)
#        #self.output()
#  def on_accelerometor_data(self, myo, timestamp, acceleration):
#      x = acceleration.x
#      y = acceleration.y
#      z = acceleration.z
#      a.append(acceleration)
#      acc.append([x,y,z])
#      
#  def get_emg_data(self):
#    with self.lock:
#      return list(self.emg_data_queue)

#
#print("R: ", avg_r)
#print("Correct:", avg_corr)
#print("Length", avg_len)
#
#np.save("E:\정리\코드\Demo\data/overlab_data",d)
#Cir = np.load("E:\정리\코드\Demo\data/60_Circle.npy")
#rec = np.load("E:\정리\코드\Demo\data/60_Rec.npy")
#Tri = np.load("E:\정리\코드\Demo\data/60_Triangle.npy")
#
#Cir = Cir[:,-3:]
#rec = rec[:,-3:]
#Tri = Tri[:,-3:]
#
#Cir = preprocessing(Cir)
#rec = preprocessing(rec)
#Tri = preprocessing(Tri)
#
#data = np.vstack([Cir, rec])
#data = np.vstack([data,Tri])
##data = data[:,-3:]
#label = np.zeros_like(data)[:,0:1]+2
#label[0:len(Cir)] = 0
#label[len(Cir):len(rec)+len(Cir)] = 1
#X_train, X_test, Y_train, Y_test = train_test_split(data,label, test_size=0.3)
#np.save("E:\정리\코드\classification-with-costly-features-master\data/spc_train", X_train)
#np.save("E:\정리\코드\classification-with-costly-features-master\data/spc_test", X_test)
#np.save("E:\정리\코드\classification-with-costly-features-master\data/spc_ytrain", Y_train)
#np.save("E:\정리\코드\classification-with-costly-features-master\data/spc_ytest", Y_test)
