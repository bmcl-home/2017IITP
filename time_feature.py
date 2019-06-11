import numpy as np
from statsmodels.tsa.ar_model import AR


class Time_Feature_extraction():
    def __init__(self, EMG_data):
        self.EMG_data = np.array(EMG_data)     # 2차원 row-data 매트릭스 들어와야 함
        self.fs = 200
        self.length = len(self.EMG_data)
        self.threshold = 0.01     # threshold
        self.no_feature = 8      # no. of feature

    def feature_extract(self):
        # time_domain
        rms = self.rms()
        iav = self.iav()
        mav = self.mav()
        mav1 = self.mav1()
        ssi = self.ssi()
        var = self.var()
        wl = self.wl()
        aac = self.aac()
        dasdv = self.dasdv()
        zc = self.zc()
        ssc = self.ssc()
        wamp = self.wamp()
        myop = self.myop()
        arc1, arc2, arc3, arc4 = self.fourth_AR()
        result = np.array([rms, iav, mav, mav1, ssi, var, wl, aac, dasdv, zc,
                           ssc, wamp, myop, arc1, arc2, arc3, arc4])
        # return np.array([rms, iav, mav, mav1, ssi, var, wl, aac, dasdv, zc, ssc, wamp, myop, arc1, arc2, arc3, arc4])
        result = result.reshape([-1])
        return result

    # time-domain
    def rms(self):
        rms = np.square(self.EMG_data)
        rms = np.sqrt(np.sum(rms, axis=0) / self.length)
        return rms

    def iav(self):
        iav = np.sum(np.abs(self.EMG_data), 0)
        return iav

    def mav(self):
        mav = np.mean(self.EMG_data, 0)
        return mav

    def mav1(self):
        mav1_start = int(len(self.EMG_data * 0.25))
        mav1_end = int(len(self.EMG_data * 0.75))
        mav1 = self.EMG_data * 0.5
        mav1[mav1_start:mav1_end] = self.EMG_data[mav1_start:mav1_end]
        mav1 = np.mean(np.abs(mav1), 0)
        return mav1

    def ssi(self):
        ssi = np.square(self.EMG_data)
        ssi = np.sum(ssi, 0)
        return ssi

    def var(self):
        var = np.square(self.EMG_data)
        var = np.sum(var, 0) / (self.length - 1)
        return var

    def wl(self):
        wl = np.zeros((self.length, self.no_feature))
        for i in range(self.length-1):
            wl[i] = self.EMG_data[i+1] - self.EMG_data[i]
        wl = np.sum(wl, 0)
        return wl

    def aac(self):
        aac = self.wl() / self.length
        return aac

    def dasdv(self):
        dasdv = np.zeros((self.length, self.no_feature))
        for i in range(self.length - 1):
            dasdv[i] = self.EMG_data[i+1] - self.EMG_data[i]
        dasdv = np.square(dasdv)
        dasdv = np.sqrt(np.sum(dasdv, 0) / (self.length - 1))
        return dasdv

    def zc(self):
        zc_bool = np.zeros((self.length, self.no_feature))
        for i in range(self.length - 1):
            zc_bool[i] = ((self.EMG_data[i] > 0) & (self.EMG_data[i+1] < 0)) | ((self.EMG_data[i] < 0) & (self.EMG_data[i+1] > 0))
        zc_bool = np.sum(zc_bool, 0)
        return zc_bool

    def ssc(self):
        ssc_bool = np.zeros((self.length, self.no_feature))
        for i in range(1, self.length - 1):
            ssc_bool[i] = ((self.EMG_data[i] > self.EMG_data[i-1]) & (self.EMG_data[i] > self.EMG_data[i+1])) | ((self.EMG_data[i] < self.EMG_data[i-1]) & (self.EMG_data[i] < self.EMG_data[i+1]))
        ssc_bool = np.sum(ssc_bool, 0)
        return ssc_bool

    def wamp(self):
        wa_num = np.zeros((self.length, self.no_feature))
        for i in range(self.length - 1):
            wa_num[i] = self.EMG_data[i] - self.EMG_data[i+1]
            wa_num[i] = self.threshold_checker(wa_num[i], self.threshold)
        wa_num = np.sum(wa_num, 0)
        return wa_num

    def myop(self):
        myop = np.zeros((self.length, self.no_feature))
        for i in range(self.length - 1):
            myop[i] = self.threshold_checker(self.EMG_data[i], self.threshold)
        myop = np.sum(myop, 0) / self.length
        return myop

    def threshold_checker(self, x, t):    # wamp()와 myop()를 위한 함수
        th_over = x > t
        return th_over

    def fourth_AR(self):
        self.EMG_data
        mat = np.zeros((4, 8))
        for i in range(8):
            model = AR(self.EMG_data[:,i])
            model_fit = model.fit(3)
            mat[:,i]=model_fit.params
        a1, a2, a3, a4 = mat[0], mat[1], mat[2], mat[3]
        return a1, a2, a3, a4
