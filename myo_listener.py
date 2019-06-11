# The MIT License (MIT)
#
# Copyright (c) 2017 Niklas Rosenstein
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
"""
This example displays the orientation, pose and RSSI as well as EMG data
if it is enabled and whether the device is locked or unlocked in the
terminal.
Enable EMG streaming with double tap and disable it with finger spread.
"""

from __future__ import print_function
from myo.utils import TimeInterval
from matplotlib import pyplot as plt
import myo
import sys
import numpy as np
import threading
import collections
import time
from collections import deque
from threading import Lock, Thread
#acc = list()
#ori = list()
#gyr = list()

class Listener(myo.DeviceListener):

  def __init__(self, queue_size = 8):
      
    self.lock = Lock()
    self.emg_data_queue = collections.deque(maxlen = queue_size)
    self.imu_data_queue = collections.deque(maxlen = queue_size)
    self.acc_data_queue = collections.deque(maxlen = queue_size)
    self.interval = TimeInterval(None, 0.05)
    self.orientation = list()
    self.acc = list()
    self.gyro = list()
    self.pose = myo.Pose.rest
    self.emg_enabled = False
    self.locked = False
    self.rssi = None
    self.emg = list()
    
  
  def on_connected(self, event):
    event.device.stream_emg(True)
    event.device.request_rssi()

  def on_rssi(self, event):
    self.rssi = event.rssi
    #self.output()
    #self.output()
  def on_orientation(self, event):
      with self.lock:
          self.imu_data_queue.append((event.timestamp, [i for i in event.orientation]))
          self.acc_data_queue.append((event.timestamp, [i for i in event.acceleration]))
          self.emg_data_queue.append((event.timestamp, [i for i in event.emg]))
          self.acc.append([i for i in event.acceleration])
          self.orientation.append([i for i in event.orientation])
          #self.emg.append([i for i in event.emg])
          self.gyro = event.gyroscope
          
          #self.orientation = event.orientation
      #acc.append([self.acc[0],self.acc[1],self.acc[2]])
      #ori.append(self.orientation)
      #gyr.append(self.gyro)
      #self.output()
  def on_emg(self, event):
      with self.lock:
          self.emg_data_queue.append((event.timestamp, event.emg))
  def get_emg_data(self):
      with self.lock:
          return list(self.emg_data_queue)
  def get_acc_data(self):
      with self.lock:
          return list(self.acc_data_queue)
  def get_ori_data(self):
    with self.lock:
        return list(self.imu_data_queue)
    

# ======================================================================        
def main():
    queue_size = 512
    interval = 5
    myo.init()
    hub = myo.Hub()
    listener = Listener(queue_size)
    fig = plt.figure()
    axes = [fig.add_subplot('31'+str(i)) for i in range(1,4)]
    [(ax.set_ylim([-2,2]))for ax in axes]
    graphs = [ax.plot(np.arange(queue_size), np.zeros(queue_size))[0] for ax in axes]
    plt.ion()
    start = time.time()
    
    def update_plot():
        emg_data = listener.get_emg_data()
        emg_data = np.array([x[1] for x in emg_data]).T
        for g, data in zip(graphs, emg_data):
            if len(data) < queue_size:
                data = np.concatenate([np.zeros(queue_size - len(data)), data])
            g.set_ydata(data)
        plt.draw()
        
    def plot_acc():
        imus = np.array([x[1] for x in listener.get_acc_data()]).T    
        for g, data in zip(graphs, imus):
            #temp = [data[i] for i in range(4)]
            #data = np.array(data)
            if len(data) < queue_size:
                data = np.concatenate([np.zeros(queue_size - len(data)), data])
            g.set_ydata(data)
        plt.draw()
    try:
        threading.Thread(target = lambda: hub.run_forever(listener.on_event)).start()
        while True:
            plot_acc()
            plt.pause(1.0/10)
            cur_time = time.time()
            print(cur_time - start)
            if cur_time - start > interval:
                break
            #print(len(listener.imu_data_queue))
    finally:
        np.save('acc_data',listener.acc)
        np.save('ori_data',listener.orientation)
        #np.save('emg_data',listener.emg)
        hub.stop()
        
        
if __name__=='__main__':
    main()        
          
#%%
