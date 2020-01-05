import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import numpy as np
from scipy.stats import norm
import math

def Gaussian1D(x,mu,sigma):

    A = 1/(sigma*np.sqrt(2*math.pi))
    Z = A*np.exp(-(x-mu)**2/(2.*sigma**2))

    return Z
    
np.random.seed(0)

points = np.linspace(-5,5,num=20)
points0 = points*np.random.rand(len(points))+16 
points1 = points*np.random.rand(len(points))-16 
points2 = points*np.random.rand(len(points)) 
points_flattened = np.stack((points0,points1,points2)).flatten() 

class GaussianMixture1D:
    def __init__(self,X,iterations):
        self.iterations = iterations
        self.variance = None
        self.X = X
        self.pi = None
        self.mean = None
        
    def prog(self):

        self.mean = [-8,8,5]
        self.pi = [1/3,1/3,1/3]
        self.variance = [5,3,1]
      
###########################################EXPECTATION STEP#######################################      
        for iter in range(self.iterations):

            probab = np.zeros((len(points_flattened),3))
 
            for cluster,gaussian,height in zip(range(3),[norm(loc=self.mean[0],scale=self.variance[0]),
                                       norm(loc=self.mean[1],scale=self.variance[1]),
                                       norm(loc=self.mean[2],scale=self.variance[2])],self.pi):
                probab[:,cluster] = height*gaussian.pdf(points_flattened) 
            
            for i in range(len(probab)):
                probab[i] = probab[i]/(np.sum(self.pi)*np.sum(probab,axis=1)[i])

            if iter == 0:
                fig = plt.figure(figsize=(10,10))
                axis0 = fig.add_subplot(111)
                for i in range(len(probab)):
                    axis0.scatter(self.X[i],0,c=np.array([probab[i][0],probab[i][1],probab[i][2]]),s=100)

                for g,c in zip([norm(loc=self.mean[0],scale=self.variance[0]).pdf(np.linspace(-20,20,num=60)),
                                norm(loc=self.mean[1],scale=self.variance[1]).pdf(np.linspace(-20,20,num=60)),
                                norm(loc=self.mean[2],scale=self.variance[2]).pdf(np.linspace(-20,20,num=60))],['r','g','b']):
                    axis0.plot(np.linspace(-20,20,num=60),g,c=c)
              
########################################### MAXIMIZATION STEP################################################################
          
            mean_of_cluster = []
            for c in range(len(probab[0])):
                m = np.sum(probab[:,c])
                mean_of_cluster.append(m) 

            for k in range(len(mean_of_cluster)):
                self.pi[k] = (mean_of_cluster[k]/np.sum(mean_of_cluster))

            self.mean = np.sum(self.X.reshape(len(self.X),1)*probab,axis=0)/mean_of_cluster
            print('mean',self.mean)

            variance_of_cluster = []
            for c in range(len(probab[0])):
                variance_of_cluster.append((1/mean_of_cluster[c])*np.dot(((np.array(probab[:,c]).reshape(60,1))*(self.X.reshape(len(self.X),1)-self.mean[c])).transpose(),(self.X.reshape(len(self.X),1)-self.mean[c])))
          
            flattened_ls = []
            for sublist in variance_of_cluster:
                for item in sublist:
                    for sub_item in item:
                        flattened_ls.append(sub_item)
          
            a = np.linspace(-20,20,num=60)
            a = a.reshape((1,a.shape[0]))
            if iter == 9:
                fig = plt.figure(figsize=(10,10))
                axis0 = fig.add_subplot(111)
                for i in range(len(probab)):
                    axis0.scatter(self.X[i],0,c=np.array([probab[i][0],probab[i][1],probab[i][2]]),s=100)

                for g,c in zip([norm(loc=self.mean[0],scale=flattened_ls[0]).pdf(np.linspace(-30,30,num=60)),
                                norm(loc=self.mean[1],scale=flattened_ls[1]).pdf(np.linspace(-30,30,num=60)),
                                norm(loc=self.mean[2],scale=flattened_ls[2]).pdf(np.linspace(-30,30,num=60))],['r','g','b']):
                    axis0.plot(np.linspace(-30,30,num=60),g,c=c)
            plt.show()
  
GM1D = GaussianMixture1D(points_flattened,10)
GM1D.prog()
