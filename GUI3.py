# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:38:01 2018

@author: seokwoojoon
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import RadioButtons
import matplotlib.image as mpimg

import collections
import threading
import time
import datetime
import load_data
import torch.nn.functional as F
from torch.autograd import Variable
#import eval
#import utils as utils
import pandas as pd
import eval_
# Myo Application import 
import myo
from myo.utils import TimeInterval
import matplotlib.font_manager as fm
from main_process import main_
import utils
import myo_listener
from matplotlib.widgets import TextBox
from matplotlib import rc, font_manager
from matplotlib.image import imread
import Myoonline, time_feature
#%%
#freqs = np.arange(2, 20, 3)

fig=plt.figure(figsize=(8,8))
#fig.images()
#img = imread('room.jpg')
#plt.imshow(img)

#plt.subplots_adjust(bottom=0.2)
#t = np.arange(0.0, 1.0, 0.001)
#s = np.sin(2*np.pi*freqs[0]*t)
#l, = plt.plot(t, s, lw=2)


    
class Index(object):
    ind = 0

    def EMG(self, event):
        
        #Myoonline.main()
        fig.text(0.7, 0.85, 'fff ')
       
    
    def SSVEP(self, event):
        # 원래 있었던 Text 제거 
        for txt in fig.texts:
            txt.set_visible(False)
        fig.text(0.7, 0.88, '인식 결과 : 1')
        #text_box = TextBox(axSSVEP_Textbox, 'SSVEP_1', initial='1')
        #text_box.on_submit()
        
    def close(self, event):
        plt.close()
        
    def text_box(self, event):
        pass
    def RL_training(self,event):
        pass


callback = Index()
axSSVEP = plt.axes([0.1, 0.85, 0.5, 0.075])
#axSSVEP_Textbox = plt.axes([0.7, 0.85, 0.2, 0.075])
axEMG = plt.axes([0.1, 0.75, 0.5, 0.075])
axRL = plt.axes([0.1, 0.65, 0.5, 0.075])

axClose = plt.axes([0.81, 0.05, 0.1, 0.075])


bEMG = Button(axEMG, 'EMG')
bEMG.on_clicked(callback.EMG)
bSSVEP = Button(axSSVEP, 'SSVEP')
bSSVEP.on_clicked(callback.SSVEP)
bClose = Button(axClose, 'Close')
bClose.on_clicked(callback.close)
bRL = Button(axRL, 'Reinforcement Learning Training')
bRL.on_clicked(callback.RL_training)
#bText_box = TextBox(axSSVEP_Textbox,'  ',initial=' ')
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
plt.show()