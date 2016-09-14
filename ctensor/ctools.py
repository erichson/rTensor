import numpy as np
import scipy as sci
import timeit

from .ktensor import ktensor
from .dtensor import dtensor, unfolded_dtensor


def kr(A, B):
    """
    Khatri-Rao product.

    Computes the Khatri-Rao product of two matrices using
    einstein summation.

    
    Parameters
    ----------
    A : array_like
        Real/complex input matrix  `A` with dimensions `(n, p)`.
        
    B : array_like
        Real/complex input matrix  `B` with dimensions `(m, p)`.
        
    Returns
    -------
    C : array_like
        Real/complex matrix  `C` with dimensions `(mn, p)`.

    Notes
    -----  
    Assumes blocks to be the columns of both matrices.
    
    References
    ----------
    See https://en.wikipedia.org/wiki/Kronecker_product#Khatri.E2.80.93Rao_product
    Code adapted from https://github.com/tensorlib/tensorlib

    Examples
    --------

    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must have 2 dimensions")

    n, pA = A.shape
    m, pB = B.shape

    if pA != pB:
        raise ValueError("A and B must have the same number of columns")

    return np.einsum('ij, kj -> ikj', A, B).reshape(m * n, pA)


#matrix transpose for real matricies
def _rT(A): 
    return A.T
    
#matrix transpose for complex matricies
def _cT(A): 
    return A.conj().T   


def _compress(X, r, p, q):
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Shape of input matrix 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~         
    dat_type =  X.dtype   
    if  dat_type == np.float32: 
        isreal = True
        fT = _rT
    elif dat_type == np.float64: 
        isreal = True
        fT = _rT
    elif dat_type == np.complex64:
        isreal = False 
        fT = _cT
    elif dat_type == np.complex128:
        isreal = False 
        fT = _cT
    else:
        raise ValueError("A.dtype is not supported")    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compress each principal flattened version of tensor
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    Qlist = [] # init empty list to store Q
    
    if isinstance(p, list) == False: 
        p = np.repeat(p,X.ndim)
        
    for mode in xrange(X.ndim):
        
        A = X.unfold(mode)
        m , n = A.shape 
        
	if p[mode] == None:  
            Qlist.append(None)

        elif (p[mode]+r) >= m:   
            Qlist.append(None)
        
        else:
        
            # Set number of samples
            k = min(p[mode] + r, m)
            if k < 0:
            	raise ValueError("The number of samples must be positive.")  
        
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #Generate a random sampling matrix O
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            O = np.array( np.random.uniform( -1 , 1 , size=( n, k ) ) , dtype = dat_type ) 
            
            if isreal==False: 
                    O += 1j * np.array( np.random.uniform(-1 , 1 , size=( n, k  ) ) , dtype = dat_type )
                
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #Build sample matrix Y : Y = A * O
            #Note: Y should approximate the range of A
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
            Y = A.dot(O)
            del(O)
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #Orthogonalize Y using economic QR decomposition: Y=QR
            #If q > 0 perfrom q subspace iterations
            #Note: check_finite=False may give a performance gain
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      
#            s=1 #control parameter for number of orthogonalizations
#            if q > 0:
#                for i in np.arange( 1, q+1 ):
#                    if( (2*i-2) % s == 0 ):
#                        Y , _ = sci.linalg.qr( Y , mode='economic', check_finite=False, overwrite_a=True )
#                    
#                    
#                    if( (2*i-1) % s == 0 ):
#                        Z , _ = sci.linalg.qr( np.dot( fT(A) , Y ) , mode='economic', check_finite=False, overwrite_a=True)
#               
#                    Y = np.dot( A , Z )
#                    #print('subiter', i)
#                #End for
#             #End if       
#                
#            Q , _ = sci.linalg.qr( Y ,  mode='economic' , check_finite=False, overwrite_a=True ) 
#            del(Y)
            
            
            if q > 0:
                for i in np.arange( 1, q+1 ):
                    Y , _ = sci.linalg.lu( Y , permute_l=True, check_finite=False, overwrite_a=True )
                    Z , _ = sci.linalg.lu( np.dot( fT(A) , Y ) , permute_l=True, check_finite=False, overwrite_a=True)
                    Y = np.dot( A , Z )
                #End for
             #End if       
                
            Q , _ = sci.linalg.qr( Y ,  mode='economic' , check_finite=False, overwrite_a=True ) 
            del(Y)
            
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Project the data matrix a into a lower dimensional subspace
            # B = Q.T * A 
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
            B = fT(Q).dot(A)   
        
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # fold matrix
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            shape = np.array( X.shape )
            shape[mode] = k
            X = unfolded_dtensor(B, mode=mode, ten_shape=tuple(shape))
            X = X.fold()
            Qlist.append( Q )
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Return Q and B
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   
    return ( Qlist, X )

def _sign_flip(A):
    """
    Flip the signs of A so that largest absolute value is positive.
    """
    signs = np.sign(A[np.argmax(np.abs(A), axis=0), list(range(A.shape[1]))])
    return signs * A


def _eiginit(X, r, mode):
    if mode==0: return( np.zeros((X.shape[0],r)) )
    XXt = X.dot(X.T)
    XXt = 0.5*(XXt+XXt.T) # ensure symmetry
    m = XXt.shape[0]
    _, U = sci.linalg.eigh(XXt, eigvals=(m-r, m-1))
    U = U[:, ::-1]  # reverse order 
    U = _sign_flip(U)
    return U

def _normalization(X, itr):
        if itr == 0:
            normalization = np.sqrt((X ** 2).sum(axis=0))
        else:
            normalization = np.abs(X).max(axis=0)
            normalization[normalization < 1] = 1
        return normalization


def _arrange(P, mode=None):
    # Normalize components   
    for n in xrange(len(P.U)):
        norm = np.sqrt((P.U[n] ** 2).sum(axis=0))
        P.U[n] /= norm
        P.lmbda *= norm
    
    # Sort
    sort_idx = np.argsort(P.lmbda)[::-1]
    P.lmbda = P.lmbda[sort_idx]
    P.U = [arr[ : , sort_idx] for arr in P.U]
    
    # Flip signs
    # X = [_sign_flip(arr) for arr in X]

    return( P )

