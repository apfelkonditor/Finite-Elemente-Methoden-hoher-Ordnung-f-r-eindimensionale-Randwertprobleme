import math as m
from time import perf_counter
import numpy as np

def time_it(func):
    
    def wrapper(*args, **kwargs):
        
        start = perf_counter()
        res = func(*args, *kwargs)
        end = perf_counter()
        delta = end - start
        print(f"function '{func.__name__}' with inputs ({args},{kwargs}) took {delta} seconds to complete")
        return res
    
    return wrapper


#@time_it
def my_factorial(n, n_start):
    
    assert n >= 0 and n_start >= 0, "n und n_start have to be bigger than 0"
    assert n >= n_start, "n has to be bigger than n_start"
    
    res = 1
    if n_start < 2:
        n_start == 2
    for i in range(n_start,n+1):
        res = res*i
    
    return res


def polynom_eval(x, coeffs):
    """
    evaulates a given polynomial at a given point x, using horner's method
    
    
    Parameters
    ======
    x ... point at which the polynomial shall be evaluated, float or int
    coeffs ... list of the coefficients of the polynomial
    
    
    Examples
    ======
    >>> polynom_eval(2, [2,3,4])
    18
    
    """
    
    assert isinstance(x, float) or isinstance(x, int), "x has to be flaot or int"
    assert isinstance(coeffs, list), "coefficients have to be passed as list"
    
    res = coeffs[0]
    for coeff in coeffs[1:]:
        res = res * x + coeff
    
    return res


def diagonal_form_universal(a, upper = 1, lower= 1):
    """
    a is a numpy square matrix
    this function converts a square matrix to diagonal ordered form
    returned matrix in ab shape which can be used directly for scipy.linalg.solve_banded
    
    slightly changed from https://github.com/scipy/scipy/issues/8362
    """
    n = a.shape[1]
    assert(np.all(a.shape ==(n,n)))
    
    ab = np.zeros((2*n-1, n))
    
    for i in range(n):
        ab[i,(n-1)-i:] = a.diagonal((n-1)-i)
        
    for i in range(n-1): 
        ab[(2*n-2)-i,:i+1] = a.diagonal(i-(n-1))

    mid_row_inx = int(ab.shape[0]/2)
    upper_rows = [mid_row_inx - i for i in range(1, upper+1)]
    upper_rows.reverse()
    upper_rows.append(mid_row_inx)
    lower_rows = [mid_row_inx + i for i in range(1, lower+1)]
    keep_rows = upper_rows+lower_rows
    ab = ab[keep_rows,:]


    return ab


def diagonal_form_tridiag(a):
    """
    takes in a numpy matrix a (or sparse matrix) and convertrs it to diagonal form (band storage)
    in order to uses it in scipy.linalg.solve_banded
    """
    
    n = a.shape[0]
    ab = np.zeros((3,n))
    
    ab[0,1:] = a.diagonal(1)
    ab[1] = a.diagonal(0)
    ab[2,:(n-1)] = a.diagonal(-1)
    
    return ab
