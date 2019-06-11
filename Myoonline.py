from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread
import time_feature as tf
import time
import myo
import numpy as np
import pickle
from keras.models import load_model
from sklearn.preprocessing import normalize
from phue import Bridge
import random

#%%


b = Bridge('192.168.0.89') # Enter bridge IP here.
lights = b.get_light_objects()
l1 = lights[0]
l2 = lights[1]
l3 = lights[2]   

#%%
for light in lights:
	light.brightness = 100
	#light.xy = [random.random(),random.random()]
#%%
class EmgCollector(myo.DeviceListener):
    """
    Collects EMG data in a queue with *n* maximum number of elements.
    """

    def __init__(self, n):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=n)
        # self.emg_long = deque(maxlen=10000)

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)

    # myo.DeviceListener

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append((event.timestamp, event.emg))


class Plot(object):

    def __init__(self, listener, interval_time):
        self.n = listener.n
        self.listener = listener
        self.fig = plt.figure()
        self.emg = list()
        self.total_emg = list()
        self.start = time.time()
        self.end = None
        self.interval_time = interval_time
        self.axes = [self.fig.add_subplot('81' + str(i)) for i in range(1, 9)]
        [(ax.set_ylim([-100, 100])) for ax in self.axes]
        self.graphs = [ax.plot(np.arange(self.n), np.zeros(self.n))[0] for ax in self.axes]
        plt.ion()

    def update_plot(self):
        emg_data = self.listener.get_emg_data()
        emg_data = np.array([x[1] for x in emg_data]).T
        self.emg.append(emg_data)
        for g, data in zip(self.graphs, emg_data):
            if len(data) < self.n:
                # Fill the left side with zeroes.
                data = np.concatenate([np.zeros(self.n - len(data)), data])
            g.set_ydata(data)
        plt.draw()

    def main(self):
        #start = time.time()
        while True:
            self.end = time.time()
            self.update_plot()    # plotting 계속 하기 때문에 반복문이 멈추면 안됨
            
            plt.pause(1.0 / 30)
            # name =time.time
            if round((self.end-self.start) % 1) == 0:     # 반복문 계속 돌면서 EMG 수집
                # np.save('data/demo_data/on_data{}'.format(round((end-start) % 6)), self.emg[-1])
                #print(np.asarray(self.emg[-1]).shape)
                np.save('on_data', self.emg[-1])
                #np.save('online_data/class5', self.emg[-1])
                #emg = self.emg[-1]
            if self.end - self.start >=self.interval_time:
                break
    def classifier(self):
    #time.sleep(0.1)
        print("Classifying")
        #emg_data= np.load('on_data.npy')
        #data = emg_data.T[-200:,:]
                
        #filename = 'my_emg_model.h5'
        # window size 0.5 sec --> 100 point 
        #filename = 'emg_100_model.h5'
        filename = 'wj.h5'
        #loaded_model = pickle.load(open(filename, 'rb'))\
        loaded_model = load_model(filename)
        #start = time.time()
        #time.sleep(1)
        while True:
            time.sleep(1.5)
            #print(data.shape)
            #end = time.time()
            if self.end-self.start >self.interval_time: 
                break
            
            #emg_data= np.load('on_data.npy')
            
            data = self.emg[-1].T[-400:,:]
            self.total_emg.append(data)

            total_feature = list()
            window_size = 200
            for i in range(2):
                idx = i*window_size
                idx2 = (i+1)*window_size
                #print(idx, idx2)
                feature_extractor = tf.Time_Feature_extraction(data[idx:idx2,:])
                feature_ = feature_extractor.feature_extract()
                feature_ = feature_.reshape([1,-1])
                total_feature.append(feature_)
            #    print(feature_.shape)
                
            ######################
            
            
            #total_feature = list()
            #for i in range(2):
            #    idx = i*100
            #    idx2 = (i+1)*100
                #print(idx, idx2)
#            feature_extractor = tf.Time_Feature_extraction(data)
#            feature_ = feature_extractor.feature_extract()
#            feature_ = feature_.reshape([1,-1])
#            total_feature.append(feature_) 
                
                
            total_feature  = np.asarray(total_feature)
            total_feature = total_feature.reshape([-1,136])
            total_feature = normalize(total_feature)
            #filename = 'my_emg_model.h5'
            #loaded_model = pickle.load(open(filename, 'rb'))\
            #loaded_model = model
            #loaded_model = load_model(filename)
            
            prediction=loaded_model.predict(total_feature)
            ans = np.argmax(prediction[0,:])
            ans2 = np.argmax(prediction[1,:])
            a = None
#            if ans==ans2:
#                a = ans
#            if a==1:
#                l1.brightness=200
#                l1.xy = [0.7, 0.3]
#                l2.brightness=200
#                l2.xy = [0.15, 0.9]
#            elif (ans or ans2)==2 :
#                l2.brightness=200
#                l2.xy = [0.15, 0.9]
#                l3.brightness=200
#            elif a==3:
#                l1.brightness=200
#                l3.brightness=200
#                l3.xy = [0.45, 0.9]
#            else:
#                for light in lights:
#                    light.brightness = 1
            
            
            #if ans==ans2:
            #    a = ans
            
            a = np.random.randint(1,4)
            if a==1:
                
                l1.brightness=200
                l1.xy = [0.7, 0.7]
            elif (ans or ans2)==2 :
                l2.brightness=200
                l2.xy = [0.15, 0.9]
            elif a==3:
                l3.brightness=200
                l3.xy = [0.45, 0.3]
            else:
                for light in lights:
                    light.brightness = 1
                    
                    
                    
            print(ans, ans2)
                

#def main():
    # emg = list()

myo.init()
hub = myo.Hub()
listener = EmgCollector(500)

with hub.run_in_background(listener.on_event):    # 마요에서 데이터 리스닝 하는거
    pl = Plot(listener, 60)
    t1 = Thread(target =pl.classifier)
    t1.daemon = True
    t1.start()
    t2 = Thread(target=pl.main())
    t2.daemon = True
    t2.start()
    
    
    t1.join()
    t2.join()
#
#main()
    
    

#%%


#import threading
#from time import sleep, ctime
#
#loops = [8,2]
#
#def loop(nloop,nsec):
#    print ('start loop', nloop, 'at:',ctime() )
#    sleep(nsec)
#    print ('loop', nloop, 'at:', ctime() )
#
#
#def test() :
#    print ('starting at:', ctime() )
#    threads = []
#    nloops = range(len(loops))
#
#    for i in nloops:
#        t = threading.Thread(target=loop,args=(i, loops[i]))
#        threads.append(t)
#
#    for i in nloops:
#        threads[i].start()
#
#    for i in nloops:
#        threads[i].join()
#
#    print ('all Done at: ', ctime())
#
#if  __name__ == '__main__' : 
#   test()
#%%


#If running for the first time, press button on bridge and run with b.connect() uncommented
#b.connect()



#for light in lights:
#	light.brightness = 100
#	light.xy = [random.random(),random.random()]