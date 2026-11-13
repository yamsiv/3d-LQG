"""Positive corner factors with polynomial Rayleigh--Ritz angular modes.

An independent alternative to P1 FEM and particular-solution collocation.
The basis enforces exact face zeros and leading corner exponents. A Bernstein
coefficient certificate can prove positivity of the fitted polynomial factor;
it does not certify eigenfunction/harmonic error or the resulting path law.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import factorial
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh
from .covariance import validate_covariance


def _exponents(degree):
    return np.array([(a,b,degree-a-b) for a in range(degree+1)
                     for b in range(degree-a+1)],dtype=int)


class CornerFactor:
    def __init__(self,covariance):
        self.covariance=validate_covariance(covariance)
        if self.covariance.shape!=(3,3):raise ValueError("Only spherical triangles are supported")
        self.R=np.linalg.cholesky(self.covariance)
        self.charts=[]
        self.face_pairs=[]
        for vertex in range(3):
            j,k=[i for i in range(3) if i!=vertex]
            rho=self.covariance[j,k]
            opening=float(np.arccos(-rho));nu=np.pi/opening
            e0=(self.R[j]-rho*self.R[k])/np.sqrt(1-rho*rho)
            e1=self.R[k]
            self.charts.append((opening,nu,e0,e1))
            self.face_pairs.append((j,k,rho))
        self.homogeneity=sum(chart[1] for chart in self.charts)-3
        reference=np.linalg.solve(self.R,np.ones(3))
        reference/=np.linalg.norm(reference)
        self.log_normalization=0.
        self.log_normalization=float(self.log_value_and_gradient(reference)[0])

    def log_value_and_gradient(self,points):
        x=np.asarray(points,float);scalar=x.ndim==1;x=np.atleast_2d(x)
        d=x@self.R.T
        if np.any(d<=0):raise ValueError("Corner-factor evaluation requires strictly interior points")
        logf=np.log(d).sum(axis=1)-self.log_normalization
        gradient=(1/d)@self.R
        for (opening,nu,e0,e1),(j,k,rho) in zip(self.charts,self.face_pairs):
            if abs(nu-2)<1e-12:
                # sin(2theta)/(sin theta cos theta)=2 exactly.
                logf+=np.log(2.)
                continue
            dj,dk=d[:,j],d[:,k]
            a,b=(dj-rho*dk)/np.sqrt(1-rho*rho),dk
            r2=a*a+b*b
            # Measure the angle from the closer face directly. Subtracting
            # opening-atan2(...) destroys relative accuracy near that face.
            near,far=np.minimum(dj,dk),np.maximum(dj,dk)
            delta=np.arctan2(near,(far-rho*near)/np.sqrt(1-rho*rho))
            if np.any(delta<=0):raise ValueError("Unresolved angular boundary in corner chart")
            logf+=(nu-2)*np.log(r2)/2+np.log(nu)
            logf+=np.log(np.sinc(nu*delta/np.pi))-np.log(np.sinc(delta/np.pi))
            logf-=np.log(np.sin(opening-delta))
            small=delta<1e-4
            difference=np.empty_like(delta)
            t=delta[small]
            difference[small]=-(nu*nu-1)*t/3-(nu**4-1)*t**3/45-2*(nu**6-1)*t**5/945
            t=delta[~small]
            difference[~small]=nu/np.tan(nu*t)-1/np.tan(t)
            angular_derivative=difference+1/np.tan(opening-delta)
            angular_derivative*=np.where(dk<=dj,1.,-1.)
            gradient+=(nu-2)*(a[:,None]*e0+b[:,None]*e1)/r2[:,None]
            gradient+=angular_derivative[:,None]*(-b[:,None]*e0+a[:,None]*e1)/r2[:,None]
        return (float(logf[0]),gradient[0]) if scalar else (logf,gradient)

    def value(self,points):return np.exp(self.log_value_and_gradient(points)[0])


def spherical_quadrature(covariance,order=48):
    """Duffy/Gauss simplex quadrature followed by radial projection to S²."""
    C=validate_covariance(covariance);A=np.linalg.inv(np.linalg.cholesky(C))
    nodes,weights=leggauss(order);nodes=(nodes+1)/2;weights=weights/2
    u,v=np.meshgrid(nodes,nodes,indexing="ij")
    wu,wv=np.meshgrid(weights,weights,indexing="ij")
    t=np.column_stack((u.ravel(),((1-u)*v).ravel(),((1-u)*(1-v)).ravel()))
    z=t@A.T;r=np.linalg.norm(z,axis=1)
    w=(wu*wv*(1-u)).ravel()*abs(np.linalg.det(A))/r**3
    return z/r[:,None],w


@dataclass
class CornerRitzMode:
    covariance: np.ndarray
    factor: CornerFactor
    degree: int
    exponents: np.ndarray
    coefficients: np.ndarray
    eigenvalue: float
    quadrature_order: int
    mass_condition_number: float

    @property
    def alpha1(self):return float(np.sqrt(self.eigenvalue+.25))

    @property
    def p(self):return self.alpha1-.5

    def angular_mode(self,points):
        x=np.asarray(points,float);scalar=x.ndim==1;x=np.atleast_2d(x)
        radius=np.linalg.norm(x,axis=1)
        if np.any(radius<=0):raise ValueError("Nonzero directions required")
        z=x/radius[:,None];d=z@self.factor.R.T
        interior=np.all(d>0,axis=1)
        output=np.zeros(len(x))
        if np.any(interior):
            monomials=np.prod(d[interior,None,:]**self.exponents[None,:,:],axis=2)
            output[interior]=self.factor.value(z[interior])*(monomials@self.coefficients)
        return float(output[0]) if scalar else output

    def log_h(self,points):
        points=np.asarray(points,float);radius=np.linalg.norm(points,axis=-1)
        if np.any(radius<=0):raise ValueError("h is singular at zero")
        with np.errstate(divide="ignore",invalid="ignore"):
            return (-self.alpha1-.5)*np.log(radius)+np.log(self.angular_mode(points))

    def positivity_certificate(self,max_elevation=16):
        """Sufficient global sign bound via homogeneous Bernstein elevation.

        q(d)=sum(d)^D q(t). Elevated Bernstein coefficients bound q(t) on
        the whole closed simplex. Mixed coefficients are inconclusive, not
        proof of a negative mode. Never clip an eigenmode to enforce positivity.
        """
        coefficients={tuple(e):Fraction.from_float(float(c)) for e,c in zip(self.exponents,self.coefficients)}
        for elevation in range(max_elevation+1):
            degree=self.degree+elevation
            bernstein=[c*Fraction(factorial(e[0])*factorial(e[1])*factorial(e[2]),factorial(degree))
                       for e,c in coefficients.items()]
            lower,upper=min(bernstein),max(bernstein)
            minimum,maximum=float(lower),float(upper)
            if lower>0:
                return dict(certified_positive=True,elevation=elevation,
                    polynomial_simplex_lower=minimum,polynomial_simplex_upper=maximum,
                    qualification="exact rational Bernstein certificate for the stored binary-float polynomial; not an eigenfunction-error enclosure")
            next_coefficients={}
            for e,c in coefficients.items():
                for k in range(3):
                    target=list(e);target[k]+=1;target=tuple(target)
                    next_coefficients[target]=next_coefficients.get(target,Fraction(0))+c
            coefficients=next_coefficients
        return dict(certified_positive=False,elevation=max_elevation,
                    polynomial_simplex_lower=minimum,polynomial_simplex_upper=maximum,
                    qualification="inconclusive; no clipping or sign repair applied")


def fit_corner_ritz(covariance,degree=4,quadrature_order=48):
    C=validate_covariance(covariance)
    if not isinstance(degree,int) or degree<0:raise ValueError("Nonnegative polynomial degree required")
    factor=CornerFactor(C);z,w=spherical_quadrature(C,quadrature_order)
    d=z@factor.R.T;exponents=_exponents(degree)
    logf,loggrad=factor.log_value_and_gradient(z)
    B=np.exp(logf[:,None])*np.prod(d[:,None,:]**exponents[None,:,:],axis=2)
    monomial_gradient=np.einsum("bi,nij->nbj",exponents,factor.R[None,:,:]/d[:,:,None])
    gradient=B[:,:,None]*(loggrad[:,None,:]+monomial_gradient-(factor.homogeneity+degree)*z[:,None,:])
    M=B.T@(w[:,None]*B)
    K=sum(gradient[:,:,k].T@(w[:,None]*gradient[:,:,k]) for k in range(3))
    scale=np.sqrt(np.diag(M));Mscaled=M/np.outer(scale,scale);Kscaled=K/np.outer(scale,scale)
    condition=float(np.linalg.cond(Mscaled))
    if condition>1e15:raise ValueError("Polynomial mass matrix unresolved in double precision")
    eigenvalues,eigenvectors=eigh(Kscaled,Mscaled,subset_by_index=[0,0],driver="gvx")
    coefficients=eigenvectors[:,0]/scale
    reference=np.linalg.solve(factor.R,np.ones(3));reference/=np.linalg.norm(reference)
    dref=factor.R@reference
    if np.prod(dref[None,:]**exponents,axis=1)@coefficients<0:coefficients=-coefficients
    return CornerRitzMode(C,factor,degree,exponents,coefficients,float(eigenvalues[0]),
                          quadrature_order,condition)
