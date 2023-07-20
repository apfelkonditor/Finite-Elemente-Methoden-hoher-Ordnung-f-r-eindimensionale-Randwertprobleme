import math as m
import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

from Hilfsfunktionen import *
from scipy.linalg import eigh_tridiagonal, solve_banded


class funktion:
    
    def __init__(self, expr, *args):
        """
        constructor for the funktion class
        a funktion instance has the following instance variables:
                self.funk       ... the sympy expression itself
                self.variables  ... list of variables of the funktion
                self.isconstant ... True if there are no variables, False else
        
        
        Parameters
        ======
        
        expr ... the function which one wants to instantiate, input has to be either a sympy expression or a string
        args ... the variables on which the instantiated function depends on, have to be either sympy Symbols or strings
        
        
        Examples
        ======
        
        >>>funktion("2*x + 5", "x")
        funktion('2*x + 5','x')
        
        >>>expr = sympy.sympify("4 * cos(y)*x")
        >>>x, y = sympy.symbols("x y")
        >>>funktion(expr, x, y)
        funktion('4*x*cos(y)','x','y')
        
        """
        
        #assert right type of input
        for var in args:
            assert isinstance(var, sym.Basic) or isinstance(var, str), f"'{var}' is not an sympy expression or string"
            
        assert isinstance(expr, sym.Basic) or isinstance(expr, str), f" the first argument '{expr}' is not an sympy expression or string"
        
        #convert any input strings into sympy expressions
        if isinstance(expr, str):
            expr = sym.sympify(expr)
        
        variables = list(args)
        for var in variables:
            if isinstance(var, str):
                var = sym.symbols(var)
        variables = tuple(variables)
                
        #assign instance variables
        self.funk = expr
        self.variables = variables
        self.isconstant = False
        if len(args) == 0:
            self.isconstant = True
    
    
    def __repr__(self):
        rep = "funktion" + "('" + str(self.funk) + "'"
        for var in self.variables:
            rep = rep + ",'" + str(var) + "'"
        rep = rep + ")"
        
        return rep
    
    
    def funk_eval(self, x):
        """
        returns the funktion evaluation at the point x
        in case of x being a numpy array, the output is elementwise
        
        
        Parameters
        ======
        x ... point of the evaluation, can be int, float or numpy array
        
        
        Examples
        ======
        >>> f = funktion("2*x+1","x")
        >>> f.funk_eval(np.array([3,4,5]))
        array([ 7,  9, 11])
        
        >>> f.funk_eval(4)
        9
        
        >>> g = funktion("5")
        >>> g.funk_eval(np.array([3,4,5]))
        array([5., 5., 5.])
        
        """
        
        assert isinstance(x, np.ndarray) or isinstance(x, float) or isinstance(x, int), "x is not of the right type"
        
        if self.isconstant:
            
            if isinstance(x, np.ndarray):
                return np.ones(len(x)) * float(self.funk)
            else:
                return float(self.funk)
        
        f_lambda = sym.lambdify(self.variables, self.funk)
        
        return f_lambda(x)
    
    
    def __mul__(self, other):
        """"
        Class multiplication for funktion class.
        Togheter with __rmul__ it's possible to multiply a funktion instance with other funktion instances, ints, floats, and sympy Expressions
        
        Example
        ======
        >>>f = funktion("2*x","x")
        >>>f * 2
        funktion("4*x","x")
        
        >>>f * funktion("cos(y)","y")
        funktion(2*x*cos(y),"x","y")
        
        >>>f * sympy.sympify("x**2")
        funktion("2*x**3", "x")
        """
        
        try:
            res_funk = self.funk * other.funk
            res_vars = set(self.variables).union(set(other.variables))
            
            return(funktion(res_funk,*res_vars))
        except:
            pass
        
        try:
            res_funk = self.funk * other
            
            return(funktion(res_funk,*self.variables))
        except:
            print(f"Something went wrong multiplying {self} with {other}... check if {other} is of type 'funktion', 'int', or 'sympy.Expression'")
        
    
    
    def __rmul__(self, other):
        """"
        Class multiplication for funktion class.
        Togheter with __mul__ it's possible to multiply a funktion instance with other funktion instances, ints, floats, and sympy Expressions
        
        Example
        ======
        >>>f = funktion("2*x","x")
        >>>f * 2
        funktion("4*x","x")
        
        >>>f * funktion("cos(y)","y")
        funktion(2*x*cos(y),"x","y")
        
        >>>f * sympy.sympify("x**2")
        funktion("2*x**3", "x")
        """
        
        try:
            res_funk = self.funk * other.funk
            res_vars = set(self.variables).union(set(other.variables))
            
            return(funktion(res_funk,*res_vars))
        except:
            pass
        
        try:
            res_funk = self.funk * other
            
            return(funktion(res_funk,*self.variables))
        except:
            print(f"Something went wrong multiplying {self} with {other}... check if {other} is of type 'funktion', 'int', or 'sympy.Expression'")
    
    
    def plot(self, a,b, resolution = 10**3, show = True):
        
        x_axis = np.linspace(a,b,resolution)
        y_axis = [self.funk_eval(x) for x in x_axis]
        
        plt.plot(x_axis, y_axis)
        
        if show:
            plt.show()
        
    
    #!!! static_n == False ist vermutlich kaputt
    def quadrature(self, a, b, n = 16, tol = 10**-6, static_n = True):
        """
        approximates the integral of a given fuunktion object on the interval [a,b]
        
        
        Parameters
        ======
        a,b ... boudaries of the integration interval
        n   ... degree of the quadrature, if n is passed static_n must be True
        tol ... tolerance up until which the approximation gets refined
        static_n ... wether or not the quadrature is computed for one n only, or it is computed for increasing n until tol is reached
        
        Examples
        ======
        >>> f = funktion("1/x","x")
        >>> f.quadrature(2,10)
        1.609437843051694
        
        """
        
        #switching a and b if b>a and setting the sign accordingly
        sign = 1
        if a>b:
            a,b = b,a
            sign = -1
        
        #preferred option due to faster calculation
        if static_n:
            
            points, weights = funktion.construct_gauß_quadrature(n)
            
            #scaling points and weights to the right interval
            weights = (b-a)/2 * weights
            points = 1/2 * (points*(b-a) + a + b)
            
            estimate = np.inner(self.funk_eval(points), weights)
            
            return estimate * sign
        
        #alternatve option for more precise calculation
        #getting the first estimates of the quadrature
        n = 4
        points, weights = funktion.construct_gauß_quadrature(n)
        
        n= n + 2
        points_2, weights_2 = funktion.construct_gauß_quadrature(n)
        
        estimate = np.inner(self.funk_eval(points), weights)
        estimate_2 = np.inner(self.funk_eval(points_2), weights_2)
        delta = float(abs(estimate - estimate_2))
        
        #getting new estimates until further estimates won't significantly impact the result
        while delta > tol and n<32:
            estimate = estimate_2
            
            n = n + 2
            points_2, weights_2 = funktion.construct_gauß_quadrature(n)
            estimate_2 = np.inner(self.funk_eval(points_2), weights_2)
            
            delta = float(abs(estimate - estimate_2))
        
        if n == 32:
            print("integral probably doesn't converge")
        
        return estimate_2 * sign
    
    
    def differentiate(self, variable: str):
        assert variable in self.variables
        
        differentiation_var = sym.Symbol(variable)
        diff_funk = sym.diff(self.funk, differentiation_var)
        return funktion(diff_funk, variable) 

    def l2_projection(self, a, b, n=10, partition_type = "equidistant",partition = None, plot = False):
        """
        Calculates the L2 projection of a function f on the finite dimensional vectorspace span{b_i}, where the b_i are Hatfunctions on the interval [a,b] with basis points x_i in partition.
        
        Parameters
        ======
        
        a, b ... interval boundaries
                 type: int or float
                 
        n    ... resolution of the partition (=amount of Hatfunctions b_i)
                 type: int
                 
        partition_type ... type of partition, (eg. equidistant)
                           type: string
                           
        partition ... optionally a custom partition can be passed, overwrites partition_type aswell as start and endpoint a,b and resolution n
                      type: list or tuple
        
        plot ... wether or not you want to plot the projection (and underlying function f)
                 type: bool
        """
        
        #creating a partition of the interval [a,b] on which the l2_projection is based upon
        if partition is not None:
            #assert isinstance(partition, list) or isinstance(partition, tuple), f"passed partition {partition} has to be of type list or tuple"
            assert n == len(partition), f"passed partition {partition} is not the same lenght as the n that is passed"
            assert partition[0] == a and partition[n-1] == b, f"passed partition {partition} has not starting point a and endpoint b"
            n = len(partition)
            a = partition[0]
            b = partition[n-1]
            
        elif partition_type  == "equidistant":
            partition = np.linspace(a,b,n)
        
        #constructing the matrix A used for calculating "projection coefficients" of the l2 projection, careful it is in diagonal form (band storage)
        A = np.zeros((3,n))
        A[0,1:] = [1/6 * (partition[i+1] - partition[i]) for i in range(n-1)]
        A[1,1:(n-1)] = [1/3 * (partition[i+1] - partition[i-1]) for i in range(1,n-1)]
        A[2,:(n-1)] = [1/6 * (partition[i+1] - partition[i]) for i in range(n-1)]
        A[1,0] = 1/3 * (partition[1] - partition[0])
        A[1,n-1] = 1/3 * (partition[n-1] - partition[n-2])
        #print(A)
        #constructing the right hand side vector y
        y = np.zeros((n,))
        dif1 = partition[1] - partition[0]
        dif2 = partition[n-1] - partition[n-2]
        
        assert "x" in self.variables
        f1 = self * sym.sympify(f"({partition[1]} - x) / {dif1}")
        f2 = self * sym.sympify(f"(x - {partition[n-2]}) / {dif2}")
        y[0] = f1.quadrature(partition[0], partition[1])
        y[n-1] = f2.quadrature(partition[n-2], partition[n-1])
        
        for i in range(1, n-1):
            
            dif1 = partition[i] - partition[i-1]
            dif2 = partition[i+1] - partition[i]
            f1 = self * sym.sympify(f"(x - {partition[i-1]}) / {dif1}")
            f2 = self * sym.sympify(f"({partition[i+1]} - x) / {dif2}")
            
            y[i] = f1.quadrature(partition[i-1], partition[i]) + f2.quadrature(partition[i], partition[i+1])
        #print(y)
        #calculation the projection coefficients Ax = y
        projection_coeffs = solve_banded((1,1), A, y)
        
        if plot:
            plt.figure()
            self.plot(a, b, show=False)
            plt.plot(partition,projection_coeffs, color = "r", lw = 0, marker = "x")
            plt.show()
        
        return projection_coeffs
    
    
    def transform_to_interval(self, a: float, b: float):
        """
        Transforms the input fuction f, such that the integral of f on the interval [a,b] is the same as the integral of the transformed f on the interval [-1,1]
        returns the transformed function as a funktion object
        useless in quadrature, but may be usefull in some other setting, who knows
        
        Parameters
        ======
        
        a,b ... interval boundries
        
        
        Example
        ======
        >>> f = funktion('2*x', 'x')
        >>> f_hut = f.transform_to_interval(2,5)
        >>> f_hut
        funktion('4.5*x + 10.5','x')
        
        
        """
        
        #assert right input type
        assert (isinstance(a,float) or isinstance(a,int)) and (isinstance(b,float) or isinstance(b,int)), f"passed interval boundaries have to be float or int, {type(a)} and {type(b)} were passed"
        
        #switching interval boundaries if a>b
        if a>b:
            print("!!Warning: a>b  -> switching boundary variables!!")
            a, b = b, a
        
        #constant functions just get multiplied with (b-a)/2 to conserve the integral
        if self.isconstant:
            F_hut = funktion(self.funk * (b-a)/2)
            return F_hut
        
        #non constant functions additionally get composed with an linear function
        y = 1/2 * (self.variables[0] * (b - a) + b + a)
        F_hut_funk = self.funk.subs(self.variables[0], y) * (b-a)/2
        F_hut = funktion(F_hut_funk, *self.variables)
        
        return F_hut
    
    
    @staticmethod
    def construct_gauß_quadrature(n):
        """
        returns gauß quadrature points and the corresponding weights as a numpy array
        uses the corresponding tridiagonal matrix of the legendre polynomials 
        
        preferred method to funktion.construct_gauß_quadrature_points in combination with funktion.construct_quadrature_weights
        note: scipy.sparse is not used directly since it's slower than scipy.linalg.eigh_tridiagonal, additional it does'nt return all eigenvalues
        
        
        Parameters
        ======
        n ... amount of quadrature points and weights
        
        
        Examples
        ======
        >>> funktion.construct_gauß_quadrature(5)
        (array([-9.06179846e-01, -5.38469310e-01,  5.55111512e-16,  5.38469310e-01,
                 9.06179846e-01]),
         array([0.23692689, 0.47862867, 0.56888889, 0.47862867, 0.23692689]))
        """
        
        assert n>=1, "n has to be bigger than 0"
        
        #initializing main an off diagonals 
        diag = np.zeros(n)
        off_diag = np.fromfunction(np.vectorize(lambda i: (i+1) / m.sqrt(4 * (i+1)**2 - 1)), (n-1,))
        
        ##calculating eigenvalues and vectors of the corresponding tridiagonal matrix, the eigenvalues are the quadrature points
        points, eigvecs = eigh_tridiagonal(diag, off_diag)
        
        #calculating the quadrature weights
        weights = np.array([ev ** 2 * 2 for ev in eigvecs[0,:]])
        
        return points, weights
    
    
    
    @staticmethod
    def construct_legendre_polynomial(n: int, norm = False):
        """
        constructs the legendre polynomial to a given integer n.
        returns the legende polynomial of degree n in form of a sympy polynomial
        
        
        Parameters
        =======
        n    ... degree of the polynomial
        norm ... if True the legendre polynomial is normalized
        
        
        Examples
        =======
        >>> funktion.construct_legendre_polynomial(4)
        Poly(4.375*x**4 - 3.75*x**2 + 0.375, x, domain='RR')
        
        
        Bugs: when n gets to large, the algorithm is unstable causing large errors in the coefficients
        
        """
        
        assert isinstance(n, int) and n >= 0, "input has to be an integer bigger than 0"
        
        #function to compute the coefficients of the legendre polynomial
        def coefficients(n, k):
            
            if k%2 == 1:
                return 0
            
            #divide k by 2, since every other coefficient is zero
            k = int(k/2)
            #using the my_factorial function for a bit of a speed-up (even if it's very miniscule)
            res = my_factorial(2*n - 2*k, n - k + 1) / (m.factorial(n - 2*k) * m.factorial(k) * 2**n)
            
            if k%2 == 0:
                return res
            else:
                return -res
                
        
        coeff = [coefficients(n, i) for i in range(0,n+1)]
        legendre_polynomial = sym.Poly(coeff, sym.Symbol("x"))
        
        #normalize the polynomial if norm == True
        if norm:
            legendre_polynomial = sym.monic(legendre_polynomial)
            
        return legendre_polynomial
    
    
    @staticmethod
    def construct_gauß_quadrature_points(n, method = "numpy"):
        """
        constructs the quadrature point for the Gauß-quadrature by calculating the eigenvalues of the companion-matrix of the legendre polynomila
        returns a numpy array with the roots of the legendre polynomial
        using "construct_gauß_quadrature" is preferred
        
        Parameters
        ======
        n      ... amount of quadrature points, integer >= 0
        method ... method to be used. (for now only 'numpy' and 'sympy' are available, whereby numpy is the preferred method)
        
        
        Examples
        ======
        >>> construct_gauß_quadrature_points(5)
        array([-0.90617985, -0.53846931,  0.90617985,  0.53846931,  0.        ])
        
        >>>construct_gauß_quadrature_points(3, method = "sympy")
        array([-0.77459667,  0.        ,  0.77459667])
        
        """
        
        assert isinstance(n, int) and n >= 0, "input has to be an integer bigger than 0"
        
        #construct legendre polynomial
        legendre_polynomial = funktion.construct_legendre_polynomial(n, True)
        
        #calculate roots
        if method == "numpy":
            coeffs = legendre_polynomial.all_coeffs()
            roots = np.roots(coeffs)
        
        
        return np.sort(roots)
    
    
    @staticmethod
    def construct_quadrature_weights(quadrature_points, a, b):
        """
        returns quadrature weights in form of a numpy array for a given interval and quadrature points
        calculates the integral of lagrange polynomials
        "costruct_gauß_quadrature" is preferred
        
        Parameters
        ======
        quadrature_points ... numpy array of points
        a, b              ... boundaries of the interval at which the quadrature is conducted
        
        
        Examples
        ======
        >>> points = funktion.construct_gauß_quadrature_points(5)
        >>> funktion.construct_quadrature_weights(points, -1,1)
        [0.2369268850561885, 0.4786286704993668, 0.5688888888888889, 0.4786286704993658, 0.2369268850561895]
        
        """
        
        assert isinstance(quadrature_points, np.ndarray), "quadrature_points have to be passed as a numpy array"
        
        #switching interval boundaries if a>b
        if a>b:
            print("!!Warning: a>b  -> switching boundary variables!!")
            a, b = b, a
        
        weights = []
        
        #for loop to construct j-th lagrange polynomials
        for j in range(len(quadrature_points)):
            
            L_j = funktion.construct_lagrange_polynomial(quadrature_points, j)
            coeffs = L_j.all_coeffs()
            #print(coeffs)
            n = len(coeffs)
            
            #for loop to integrate the j-th lagrange polynomiall
            for i in range(n):
                coeffs[i] = coeffs[i] * 1/(n-i)
            coeffs.append(0)
            
            #final part of integration
            weights.append(polynom_eval(b, coeffs) - polynom_eval(a, coeffs))
        
        return np.array(weights)
    
    
    @staticmethod
    def construct_lagrange_polynomial(interpol_points, j):
        """
        constructs the j-th lagrange polynomial for a given array of interpolation points
        returns the j-th lagrange polynomial in form of a sympy Polynomial
        
        
        Parameters
        ======
        interpol_points ... interpolation points for the lagrange polynomial, in form of a numpy array
        j               ... index of the lagrange polynomial which shall be constructed, 0 <= j < len(interpol_points) 
        
        
        Examples
        ======
        >>> array = np.array([2,4,6])
        >>> funktion.construct_lagrange_polynomial(array, 1)
        Poly(-1/4*x**2 + 2*x - 3, x, domain='QQ')
        
        """
        
        assert isinstance(interpol_points, np.ndarray), "interpolation points have to be an numpy array"
        assert j >= 0 and j < len(interpol_points), "j is not bigger or equal than zero, or j is larger than the lenght of the array"
        
        #calculate denominator of lagrangepolynomial, j-th entry gets set to 1 since it's neglected in the multiplication
        denominator = interpol_points[j] - interpol_points
        denominator[j] = 1
        denominator = denominator.prod()
        
        #calculate nominator of lagrangepolynomial, j-th entry gets skipped again
        
        expr = f"(x - {interpol_points[0]})"
        if j == 0:
            expr = f"(x - {interpol_points[1]})"
        for i in range(1, len(interpol_points)):
            
            if j == 0 and i == 1:
                continue
            
            if i == j:
                continue
            
            expr = expr + "*" + f"(x - {interpol_points[i]})"
        
        
        lagrange_polynomial = sym.sympify(expr)
        lagrange_polynomial = sym.Poly(lagrange_polynomial / denominator)
        
        return lagrange_polynomial



def plot_l2_error(f,a,b, tol = 10**-4):
    
    errors = []
    interval_length = b - a 
    n = 10
    
    step_size = [interval_length/(n-1)]
    projection_points = f.l2_projection(a,b, n)
    f_points = f.funk_eval(np.linspace(a,b,n))
    errors.append(m.sqrt(interval_length/(n-1) * np.sum(np.abs(projection_points - f_points)**2)))
    
    for i in range(100):
        
        if errors[i] <= tol:
            break
        
        n = n + 1
        step_size.append(interval_length/(n-1))
        projection_points = f.l2_projection(a,b, n)
        f_points = f.funk_eval(np.linspace(a,b,n))
        errors.append(m.sqrt(interval_length/(n-1) * np.sum(np.abs(projection_points - f_points)**2)))
    
    plt.loglog(step_size,errors, step_size, np.array(step_size) ** 2)
    plt.title("L2_error of L2_projection")
    plt.xlabel("step_size 'h'")
    plt.ylabel("l2_errors")
    plt.grid(visible=True, which="both")
    plt.show()