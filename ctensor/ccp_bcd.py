#*************************************************************************
#***        Author: N. Benjamin Erichson <nbe@st-andrews.ac.uk>        ***
#***                              <2016>                               ***
#************************************************************************* 

import numpy as np
import scipy as sci
import timeit

from .ktensor import ktensor
from .dtensor import dtensor, unfolded_dtensor
from .ctools import kr, _rT, _cT, _compress, _sign_flip, _eiginit, _normalization, _arrange


def ccp_bcd(X, r=None, c=True, nu=0, p=10, q=1, tol=1E-5, maxiter=500, trace=True):
    """
    Randomized CP Decomposition using the Block Coordinate Descent Method.
    
    Given a tensor X, the best rank-R CP model is estimated using the 
    block coordinate descent method.
    If `c=True` the input tensor is compressed using the randomized 
    QB-decomposition.
    
    Parameters
    ----------
    X : array_like or dtensor
        Real tensor `X` with dimensions `(I, J, K)`.
    
    r : int
        `r` denotes the number of components to compute.

    c : bool `{'True', 'False'}`, optional (default `c=True`)
        Whether or not to compress the tensor.         

    p : int, optional (default `p=10`)
        `p` sets the oversampling parameter.

    q : int, optional (default `q=2`)
        `q` sets the number of power iterations.        
        
    tol : float, optional (default `tol=1E-5`)
        Stopping tolerance for reconstruction error.
        
    maxiter : int, optional (default `maxiter=500`)
        Maximum number of iterations to perform before exiting.

    trace : bool `{'True', 'False'}`, optional (default `trace=True`)
        Display progress.


    Returns
    -------
    P : ktensor
        Tensor stored in decomposed form as a Kruskal operator.

    
    Notes
    -----  
    
    
    References
    ----------
    Kolda, T. G. & Bader, B. W.
    "Tensor Decompositions and Applications." 
    SIAM Rev. 51 (2009): 455-500
    http://epubs.siam.org/doi/pdf/10.1137/07070111X

    Comon, Pierre & Xavier Luciani & Andre De Almeida. 
    "Tensor decompositions, alternating least squares and other tales."
    Journal of chemometrics 23 (2009): 393-405.
    http://onlinelibrary.wiley.com/doi/10.1002/cem.1236/abstract

    """
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Error catching
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~       
    if r is None:
        raise ValueError("Number of components 'r' not given.")

    if len(X.shape) < 3:
        raise ValueError("Array with ndim == 3 expected.")

    #if len(X.shape) > 3:
    #    raise ValueError("Array with ndim >= 3 not supported.")

    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Init
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    t0 = timeit.default_timer() # start timer
    fit_out = []
    fit = 0
    rdiff = 1
    N = X.ndim
    normX = sci.linalg.norm(X) # Fro norm


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compress Tensor
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    if trace==True: print('Shape of Tensor: ', X.shape ) 
        
    if c==True:    
        Q , X = _compress(X, r=r, p=p, q=q)

        # Compare fro norm between the compressed and full tensor  
        normXc = sci.linalg.norm(X) # Fro norm
        rdiff =  normXc/normX   
        if trace==True: print('Shape of cTensor: ', X.shape ) 
        if trace==True: print('Fro. norm of Tensor: %s,  cTensor: %s' %(normX, normXc) )
        if trace==True: print('Rel. difference of the Fro. norm: %s' %( round(1-rdiff,2) ))
        normX = normXc 
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize [A,B,C] using the higher order SVD
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    U =   [_eiginit(X.unfold(n), r, n) for n in xrange(N)]    
    #U = [np.random.standard_normal(U[n].shape) for n in xrange(N)]    

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Block coordinate descent 
    # i)  Compute Khatri-Rao Pseudoinverse
    # ii) Update component A, B, C
    # iii) Normalize columns of A, B, C to length 1
    # repeat until convergences
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    R = X 
    lamb = np.zeros(r)
    itertol = 0    
    for l in xrange(r):

            normR = sci.linalg.norm(R)
            
            for itr in xrange(maxiter):
                fitold = fit
                
                for n in xrange(N):                  
                    
                    components = [ U[j][:,l].reshape(U[j][:,l].shape[0],1)  for j in xrange(N) if j != n]
                
                    grams = [ arr.T.dot(arr) for arr in components ]             

                    # ii)  Compute Khatri-Rao Pseudoinverse
                    p1 = reduce(kr, components[:-1][::-1], components[-1])
                    p2 = sci.linalg.pinv(reduce(sci.multiply, grams, 1.))
        
                    # iii) Update component U_n            
                    U[n][:,l] = R.unfold(n).dot(p1.dot(p2) ).reshape(-1)          
        
                    # iv) normalize U_n to prevent singularities
                    lamb_temp = _normalization(U[n][:,l].reshape(U[n][:,l].shape[0],1), itr)
                    U[n][:,l] /=  lamb_temp
           
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Compute fit of the approximation,
                # defined as 1 - norm(X - full(P)) / norm(X) 
                # This is loosely the proportion of the data described by the CP model, 
                # i.e., fit of 1 is perfect
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
                components = [U[j][:,l].reshape(U[j][:,l].shape[0],1) for j in xrange(N)]
                P = ktensor( components , ([lamb_temp]) )
                normresidual = normR**2 + P.norm()**2 - 2 * P.innerprod(dtensor(R))
                fit = (1 - (normresidual / normR ** 2)) * rdiff**2
                fitchange = np.abs(fitold - fit)
        	
		itertol += 1
                if trace==True:
                    print('Iteration: %s fit: %s, fitchange: %s' %(itertol, fit, fitchange))
        
                if (fitchange < tol):
                    break
         
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update regularization parameter
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
            lamb[l] = lamb_temp
            R = R - P.toarray()

            # Global fit
            #components = [U[j][:,xrange(l+1)] for j in xrange(N)]
            #P = ktensor(components, lamb[xrange(l+1)])
            #normresidual = normX**2 + P.norm()**2 - 2 * P.innerprod(dtensor(X))
            #fit = (1 - (normresidual / normX ** 2)) * rdiff**2
            #fit_out.append( fit )
            #if trace==True:
            #        print('Global: %s fit: %s, fitchange: %s' %(len(fit_out), fit, fitchange))      
                

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Recover full-state components 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    P = ktensor(U, lamb)
    
    if c==True:
        for n in xrange(len(P.U)):
            if Q[n] is not None:
              P.U[n] = np.array(Q[n]).dot(P.U[n])
              
        P.shape = tuple([arr.shape[0] for arr in P.U])
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Normalize and sort components and store as ktensor
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    P = _arrange(P, None)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    if trace==True: print('Compute time: %s seconds' %(timeit.default_timer()  - t0))
    return ( P )        


