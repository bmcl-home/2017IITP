# -*- coding: utf-8 -*-

import sys
import collections
import threading
import myo
import pandas as pd
import time
import datetime
import load_data
from myo.utils import TimeInterval
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import RadioButtons
import matplotlib.image as mpimg
from main_process import main_
import numpy as np
import myo_listener
import eval_
from Myo_util import preprocessing
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets

 #%%

def print_progress(i, total, step=100):
	if i % step == 0:
		sys.stdout.write("\r{}/{}".format(i, total))
		
		if i >= total - step:	# new line
			print()
            
#============   MYO Listener ==================================================
#class MyListener(myo.DeviceListener):
#
#  def __init__(self, queue_size=8):
#    self.lock = threading.Lock()
#    self.emg_data_queue = collections.deque(maxlen=queue_size)
#    self.orientation_data_queue = collections.deque(maxlen=queue_size)
#    self.acc_data_queue = collections.deque(maxlen=queue_size)
#    self.orientation = list()
#    self.acceleration = list()
#    
#  def on_imu(self, event):
#      with self.lock:
#          self.orientation_data_queue.append((event.timestamp, [i for i in event.orientation]))
#          self.acc_data_queue.append((event.timestamp, [i for i in event.accleration]))
#          self.acceleration.append(self.acc_data_queue[-1])
#          self.orientation.append(self.orientation_data_queue[-1])
#          
#  def get_acc_data(self):
#    with self.lock:
#        return list(self.acc_data_queue)
#
#  def get_ori_data(self):
#    with self.lock:
#        return list(self.orientation_data_queue)            

# ====================  GUI Class ==========================
class Index(object):
    def __init__(self):
        #self.myo = 
        self.label_flag = 0
        self.load_flag = 0
        
        self.acc = None
        self.orientation = None
        #self.load_flag = True
        self.label = 0
        self.data_name = ''
        self.myo_data = 0
        self.all_train = 0
        self.all_label = 0
        self.class1 = 0
        self.class2 = 0
        self.class3 = 0
        self.class1_label=0
        self.class2_label=0
        self.class3_label=0
        #self.acc = 0
        #self.hub = myo.Hub()
        #self.listener = MyListener()
        self.fig, self.ax = plt.subplots(2)
        self.fig.subplots_adjust(left=0.3)
        self.fig.subplots_adjust(bottom=0.2)

    def Training(self, event):
        #data = self.all_train
        #label = self.all_label
        pass
        

    def Train2(self, event):
        
        main_(self.load_flag)
        
    def vis(self, event):
        self.acc = np.load('acc_data.npy')
        
        #self.orientation = np.load('ori_data.npy')
#        min_val = len(self.acc)
#        if len(self.acc) > len(self.orientation):
#            min_val = len(self.orientation)
#        self.acc = self.acc[:min_val,:]
#        self.orientation = self.orientation[:min_val,:]
        # == 앞부분 버림 ====
        #myo_data = myo_data[40:,:]
        self.myo_data = self.acc
        #self.myo_data = np.hstack([self.acc, self.orientation])
        dt = datetime.datetime.now()
        date = dt.strftime("%H_%M_%S")
        np.save("./data/temp%s.npy" %date, self.myo_data)
        np.save("./test_data/temp.npy", self.myo_data)
        self.data_name = date
        #==================== plot graph =======================
        
        #Orientation_data = pd.DataFrame(np.asarray(self.orientation),columns= ['Ox','Oy','Oz','Ow'])
        Accelerometer_data = pd.DataFrame( np.asarray(self.acc), columns=['Ax','Ay','Az'])
        self.ax[0].plot(Accelerometer_data['Ax'],'r', Accelerometer_data['Ay'],'g',Accelerometer_data['Az'],'b')
        #self.ax[1].plot(Orientation_data['Ox'], 'r', Orientation_data['Oy'], 'g', Orientation_data['Oz'], 'b',
        #      Orientation_data['Ow'],'c')  
        #self.ax[1].legend(['x','y','z','w'])
        self.ax[0].legend(['x','y','z'])
        self.ax[0].set_title('Myo data')
        self.ax[1].set_xlabel('Myo data Samples')
        self.ax[0].set_ylabel('Amplitude')
        self.ax[0].set_xlabel('Acceleration samples')
        #print("Shape:", self.myo_data.shape())
        
    
    def radio_func(self,label):
        if label is 'Triangle':
            self.label_flag=0
            self.label = 0
            print('Tri', 'label:', self.label)
            self.class1 = self.myo_data
            if len(self.myo_data) >=500:
                self.class1 = self.myo_data[:500,:]
                
            data_num = self.class1.shape[0]
            data_label = np.zeros(data_num)
            self.class1_label = data_label
            
            print(self.class1_label)
        elif label is 'Square':
            self.label_flag=1
            self.label = 1
            self.class2 = self.myo_data[:500,:]
            if len(self.myo_data) >=500:
                self.class2 = self.myo_data[:500,:]
            print('sq', 'label:', self.label)
            data_num = self.class2.shape[0]
            data_label = np.zeros(data_num) +1 
            self.class2_label = data_label
            print(self.class2_label)
            #
        elif label is 'Circle':
            self.label_flag=2
            self.label = 2
            self.class3 = self.myo_data[:500,:]
            if len(self.myo_data) >=500:
                self.class3 = self.myo_data[:500,:]
            data_num = self.class3.shape[0]
            data_label = np.zeros(data_num) +2 
            self.class3_label = data_label
            print(self.class3_label.shape)
            print('Circle')
            
        elif label is 'ok':
            
            if self.load_flag is 1:
                a = np.load('data/temp11_18_57.npy')
                c = np.load('data/temp11_20_39.npy')
                b = np.load('data/temp11_19_44.npy')
                
                a = preprocessing(a).reshape([-1,150])
                b = preprocessing(b).reshape([-1,150])
                c = preprocessing(c).reshape([-1,150])
                
                l1 = np.zeros(a.shape[0])
                l2 = np.zeros(b.shape[0])+1
                l3 = np.zeros(c.shape[0])+2
                
                self.class1 = a
                self.class2 = b
                self.class3 = c
                
                self.class1_label = l1
                self.class2_label = l2
                self.class3_label = l3
            else:
                self.class1 = preprocessing(self.class1).reshape([-1,150])
                self.class2 = preprocessing(self.class2).reshape([-1,150])
                self.class3 = preprocessing(self.class3).reshape([-1,150])
                
                self.class1_label = np.zeros(self.class1.shape[0])
                self.class2_label  = np.zeros(self.class2.shape[0])+1
                self.class3_label  = np.zeros(self.class3.shape[0])+2
                
            self.all_train = self.class1
            self.all_train = np.vstack([self.all_train, self.class2])
            self.all_train = np.vstack([self.all_train,self.class3])
            
            
            all_label = self.class1_label
            #print(all_label.shape)
            all_label = np.hstack([all_label, self.class2_label])
            all_label = np.hstack([all_label, self.class3_label])
            self.all_label = all_label
            np.save('test_data/all_data',self.all_train)
            np.save('test_data/all_label',self.all_label)
            
            #  데이터 생성 
            
            load_data.gen_data(self.all_label, self.all_train)
            #print(self.all_label.shape, self.all_trian.shape)
            print('rec', 'label:', self.label) 
            
            
    def clear(self, event):
        self.ax[0].cla()
        self.ax[1].cla()    
        
        
    def Start(self, event):
        myo_listener.main()

    def load(self, event):
        self.load_flag = 1
        print("Load --model")
        
    def Test_Btn(self, event):
        
        #test_data = self.myo_data[:500,:]
#        if len(self.myo_data) >=500:
#                self.test_data = self.myo_data[:500,:]
        #test_data = test_data[50:,:]
        test_data = self.myo_data
        if self.label_flag ==0:
            label = self.class1_label
        elif self.label_flag==1:
            label = self.class2_label
        else:
            label = self.class3_label
        load_data.gen_test_data(label, test_data)
        #print(label)
        eval_.test_action()
        print("test")
        
    def close(self, event):
        plt.close()
        
    def test(self,event):
        print("test")
        
    def sub_plot(self):
        fig, ax = plt.subplots()
        return fig, ax
    
    def show(self):
        plt.show()
        
        
class CustomAccDataSet(Dataset):
    def __init__(self, data,label):
        self.to_tensor = transforms.ToTensor()
        #self.data_info = pd.read_csv(csv_path, sep ='\t')
        #self.data_info.columns =['x','y','z','label']
        self.data_arr = data
        self.label_arr = label.reshape([-1,1])
        #labels = keras.utils.to_categorical(labels)
        #self.data_len = len(self.data_info.index)
        self.data_len = len(self.data_arr)
        
    def __getitem__(self, index):
        single_data = self.data_arr[index]
        single_label = self.label_arr[index]
        
        return (single_data, single_label)
    
    def __len__(self):
        return self.data_len
#%%
def feature(x,y,z):
    mean = np.mean()
    
    



    