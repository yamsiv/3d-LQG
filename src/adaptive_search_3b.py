"""Small GP surrogate for exploratory acquisition, not inferential confidence.

Inputs share fields and feature estimates, so diagonal observation noise is a
working acquisition approximation. Reported predictive SD is not a confidence
band for the correspondence or a controlled sampler-error bound.
"""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize


class ScoreSurrogate:
    def __init__(self,inputs,values,*,noise_floor=.04):
        self.x=np.asarray(inputs,float)/np.array([1.,1.,1.,.2])
        y=np.asarray(values,float)
        if self.x.ndim!=2 or self.x.shape[1]!=4 or len(y)!=len(self.x):raise ValueError('Need C12,C13,C23,xi inputs')
        self.mean=float(y.mean());self.amplitude=max(float(y.std()),.05);self.y=(y-self.mean)/self.amplitude
        self.noise=max(noise_floor/self.amplitude,.05)
        distance=(self.x[:,None,:]-self.x[None,:,:])**2
        def kernel(theta):
            ell=np.exp(theta)
            return np.exp(-.5*(distance[:,:,:3].sum(axis=2)/ell[0]**2+distance[:,:,3]/ell[1]**2))
        def objective(theta):
            try:
                factor=cho_factor(kernel(theta)+np.eye(len(y))*self.noise**2,lower=True)
                alpha=cho_solve(factor,self.y)
                return .5*self.y@alpha+np.log(np.diag(factor[0])).sum()
            except np.linalg.LinAlgError:return 1e30
        fit=minimize(objective,np.log([.35,.4]),method='L-BFGS-B',bounds=[(np.log(.12),np.log(1.5)),(np.log(.1),np.log(2.))])
        self.length_scales=np.exp(fit.x)
        self.factor=cho_factor(kernel(fit.x)+np.eye(len(y))*self.noise**2,lower=True)
        self.alpha=cho_solve(self.factor,self.y)

    def predict(self,inputs):
        z=np.atleast_2d(inputs)/np.array([1.,1.,1.,.2])
        difference=(z[:,None,:]-self.x[None,:,:])**2
        k=np.exp(-.5*(difference[:,:,:3].sum(axis=2)/self.length_scales[0]**2+difference[:,:,3]/self.length_scales[1]**2))
        mean=self.mean+self.amplitude*(k@self.alpha)
        var=np.maximum(0,1-np.sum(k*cho_solve(self.factor,k.T).T,axis=1))
        return mean,self.amplitude*np.sqrt(var)
