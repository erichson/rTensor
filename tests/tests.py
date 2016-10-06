from __future__ import division
import numpy as np
import scipy as sci
import scipy.sparse.linalg as scislin

from ctensor import *
from ctensor.pyutils import *
from ctensor.ctools import *


from unittest import main, makeSuite, TestCase, TestSuite
from numpy.testing import assert_raises, assert_equal

atol_float32 = 1e-4
atol_float64 = 1e-8

#
#******************************************************************************
#
def randTensor(R, lamb, I, J, K, L=None):
    if L is None:
            tensor = np.zeros((I,J,K))
    else:    
            tensor = np.zeros((I,J,K,L))
    
    
    for i in range(R):    
        a = np.random.standard_normal((I,1))
        a /= np.linalg.norm(a)
        b = np.random.standard_normal((J,1))
        b /= np.linalg.norm(b)
        c = np.random.standard_normal((K,1))
        c /= np.linalg.norm(c)
        if L is None:
            tensor += ktensor( [a, b, c], lamb[i]).toarray()
        else:
            d = np.random.standard_normal((L,1))
            d /= np.linalg.norm(d)        
        
            tensor += ktensor( [a, b, c, d], lamb[i]).toarray()            
        
    return tensor


#
#******************************************************************************
#
class test_base(TestCase):
    def setUp(self):
        np.random.seed(123)

    def test_khatrirao(self):
        A = array([ 
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
        ])
	    
        B = array([
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9]
        ])

        C = array([
        [1, 8, 21],
        [2, 10, 24],
        [3, 12, 27],
        [4, 20, 42],
        [8, 25, 48],
        [12, 30, 54],
        [7, 32, 63],
        [14, 40, 72],
        [21, 48, 81]
        ])

        assert np.allclose(khatrirao((A, B)), C, atol_float64)   

    def test_khatrirao2(self):
        A = array([ 
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
        ])
	    
        B = array([
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9]
        ])

        C = array([
        [1, 8, 21],
        [2, 10, 24],
        [3, 12, 27],
        [4, 20, 42],
        [8, 25, 48],
        [12, 30, 54],
        [7, 32, 63],
        [14, 40, 72],
        [21, 48, 81]
        ])

        assert np.allclose(kr(A, B), C, atol_float64)   

    def test_dense_fold(self):
        T = np.array([[[ 1,  4,  7, 10],
                       [ 2,  5,  8, 11],
                       [ 3,  6,  9, 12]],

                       [[13, 16, 19, 22],
                       [14, 17, 20, 23],
                       [15, 18, 21, 24]]])

        X = dtensor(T)
        # Unfold 
        U = X.unfold(0)
        assert np.allclose(U.fold(), T, atol_float64)   
        U = X.unfold(1)
        assert np.allclose(U.fold(), T, atol_float64)   
        U = X.unfold(2)
        assert np.allclose(U.fold(), T, atol_float64)   

    def test_from_to_without(self):
        frm, to, without = 2, 88, 47
        lst = list(range(frm, without)) + list(range(without + 1, to))
        assert_equal(lst, from_to_without(frm, to, without))

        rlst = list(range(to - 1, without, -1)) + list(range(without - 1, frm - 1,-1))
        assert_equal(rlst, from_to_without(frm, to, without, reverse=True))
        assert_equal(lst[::-1], from_to_without(frm, to, without, reverse=True))      

#
#******************************************************************************
#
class test_cp_als(TestCase):
    def setUp(self):
        np.random.seed(123)        
        
    def test_cp_deterministic(self):
        I,J,K,R = 15,15,15,3
        lamb = np.float64(np.floor(np.linspace(10000,1000,num=R)))
        X = randTensor(R,lamb, I,J,K)         
        
        P = ccp_als(dtensor(X), r=R, c=False, tol=atol_float64, maxiter=150)   
        percent_error = sci.linalg.norm(P.toarray() - X) / np.linalg.norm(X)
        assert percent_error < atol_float32   

    def test_cp_randomized(self):
        I,J,K,R = 15,15,15,3
        lamb = np.float64(np.floor(np.linspace(10000,1000,num=R)))
        X = randTensor(R,lamb, I,J,K)         
        
        P = ccp_als(dtensor(X), r=R, c=True, p=10, q=2, tol=atol_float64, maxiter=150)   
        percent_error = sci.linalg.norm(P.toarray() - X) / np.linalg.norm(X)
        assert percent_error < atol_float32   

#
#******************************************************************************
#
class test_cp_bcd(TestCase):
    def setUp(self):
        np.random.seed(123)        
        
    def test_cp_deterministic(self):
        I,J,K,R = 15,15,15,3
        lamb = np.float64(np.floor(np.linspace(10000,1000,num=R)))
        X = randTensor(R,lamb, I,J,K)   
        
        P = ccp_bcd(dtensor(X), r=R+5, c=False, tol=atol_float64, maxiter=150)   
        percent_error = sci.linalg.norm(P.toarray() - X) / np.linalg.norm(X)
        assert percent_error <  0.01  

    def test_cp_randomized(self):
        I,J,K,R = 15,15,15,3
        lamb = np.float64(np.floor(np.linspace(10000,1000,num=R)))
        X = randTensor(R,lamb, I,J,K) 
        
        P = ccp_bcd(dtensor(X), r=R, c=True, p=10, q=2, tol=atol_float64, maxiter=150)   
        percent_error = sci.linalg.norm(P.toarray() - X) / np.linalg.norm(X)
        assert percent_error < 0.01


#
#******************************************************************************
#
        
def suite():
    s = TestSuite()
    s.addTest(test_base('test_khatrirao'))
    s.addTest(test_base('test_dense_fold'))
    s.addTest(test_base('test_from_to_without'))

    s.addTest(test_cp_als('test_cp_deterministic'))
    s.addTest(test_cp_als('test_cp_randomized'))
    s.addTest(test_cp_bcd('test_cp_deterministic'))
    s.addTest(test_cp_bcd('test_cp_randomized'))
    
    return s

if __name__ == '__main__':
    main(defaultTest = 'suite')
