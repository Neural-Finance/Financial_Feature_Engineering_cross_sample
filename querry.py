#Author: AlexFang, alex.holla@foxmail.com.
import h5py
import numpy as np
import pandas as pd
import time

class querry(object):

    def __init__(self):
        pass

    def softmax(self,res):

        return np.exp(res)/np.sum(np.exp(res))

    def generate_data(self):

        def load(name):
            tar={}
            f=h5py.File("./cifar10_data/finance/%s.h5"%str(name))
            for i in f.keys():
                tar[i]=f[i][:100,:]
            f.close()
            return tar

        self.x=load("x_test")
        self.y=load("y_test")
        self.z=load("z_test")

        #generate factor, querry set
        factor1={}
        for i in self.x.keys():
            tmp=self.x[i]
            #tmp2=tmp[:,5]-tmp[:,1]+tmp[:,-1]-tmp[:,-5]+0.01
            tmp2=tmp[:,19]
            tmp2 = pd.Series(tmp2.reshape(-1,))
            factor1[i] = np.array(tmp2.rank())/len(tmp2)
            tmp3=np.zeros((tmp.shape[0],4+tmp.shape[1]))
            tmp3[:,2:-2]=tmp
            self.x[i]=tmp3.reshape(1,-1)

        self.factor1=factor1