import matplotlib.pyplot as plt
import numpy as np
from scipy.special import binom
import sympy as sym


class Simple_function():
    """A class that represents so called hat-functions, 
    careful: allways the whole function is created;
    (important when assembling bilinearform only half the function is nedeed from 1st and last basis-function
    
    Instance Variables
    ======
    self.a ... starting point of the interval the function is defined on
    self.m ... middle point of the interval
    self.b ... endpoint of the function
    self.max ... maximum height of the hat function
    """

    def __init__(self, basis_points: list, max :float = 1, sidedness: str = "both") -> None:
        """
        Parameters
        ======
        basis_points: list ... list of length 3, with the starting, middle and endpoint of the interval on which the function is defined
        max ... maximum height of the hat function
        sidedness ... the side of the basis function that is relevant, other side is considered to be 0
        """
        self.sidedness = sidedness
        if sidedness == "both":
            assert len(basis_points) == 3, "3 basis_points must be provided"
            assert basis_points[0] < basis_points[1] and basis_points[1] < basis_points[2], "basis_points must be provided in ascending order" 

            self.a = basis_points[0]
            self.m = basis_points[1]
            self.b = basis_points[2]
            self.max = max
        
        elif sidedness == "left":
            assert len(basis_points) == 3 and not basis_points[2], "3 basis_points must be provided, third one should be 'None'."
            self.a = basis_points[0]
            self.m = basis_points[1]
            self.b = None
            self.max = max
        
        else:
            assert len(basis_points) == 3 and not basis_points[0], "3 basis_points must be provided, first one should be 'None'."
            self.a = None
            self.m = basis_points[1]
            self.b = basis_points[2]
            self.max = max
            
        

    def __repr__(self) -> str:
        return f"Simple_function([{self.a},{self.m},{self.b}],max = {self.max}, sidedness = {self.sidedness})"
    

    def __str__(self) -> str:
        if self.sidedness == "both":
            rep = f"piecewise(  {self.max} * (x - {self.a}) / {self.m - self.a}, if x in [{self.a},{self.m}] \n {self.max} * ({self.b} - x) / {self.b - self.m}, if x in [{self.m},{self.b}] \n 0, else)"
        elif self.sidedness == "left":
            rep = f"piecewise( {self.max} * (x - {self.a}) / {self.m - self.a}, if x in [{self.a},{self.m}] \n 0, else)"
        elif self.sidedness == "right":
            rep = f"piecewise( {self.max} * ({self.b} - x) / {self.b - self.m}, if x in [{self.m},{self.b}] \n 0, else)"

        return rep
    

    def __mul__(self, other):

        #multiplication with integer or float
        if type(other) == int or type(other) == float:
            return Simple_function([self.a, self.m, self.b], self.max * other, sidedness=self.sidedness)
        
        #multiplication with another instance of Simple_function, the result is going to be a Simple_quadratic
        try:
            #two left sided hat functions are multiplied
            if self.sidedness == "left" and other.sidedness == "left":
                assert self.a == other.a and self.m == other.m, "Basispoints have to match"
                return Simple_quadratic([self.a, self.m], coeff = self.max * other.max, type = "asc")

            #two right sided hat functions are multiplied
            elif self.sidedness == "right" and other.sidedness == "right":
                assert self.m == other.m and self.b == other.b, "Basispoints have to match"
                return Simple_quadratic([self.m, self.b], coeff = self.max * other.max, type = "desc")

            #a left sided function is multiplied with a right sided one
            elif (self.sidedness == "left" and other.sidedness == "right") or (self.sidedness == "right" and other.sidedness == "left"):
                if self.a == other.m and self.m == other.b:
                    return Simple_quadratic([self.a, self.m], coeff = self.max * other.max, type = "both")
                elif self.m == other.a and self.b == other.m:
                    return Simple_quadratic([self.m, self.b], coeff = self.max * other.max, type = "both")
                else:
                    raise ValueError
            
            else:
                raise ValueError
        
        except(ValueError):
            raise Exception("ValueError: check basispoints, they have to match. Or maybe you tried multiplying function with sidedness 'both', that's not possible either.")

        except:
            return NotImplemented


    def __rmul__(self, other):

        #multiply with integer or float; raises Error if left side of multiplication is not one of those types
        try:
            if type(other) == int or type(other) == float:
                return Simple_function([self.a, self.m, self.b], self.max * other, sidedness=self.sidedness)
            else:
                raise TypeError
        
        except(TypeError):
            return NotImplemented
    

    @staticmethod
    def float_mul(object_1 ,object_2):
        return object_1 * object_2


    def eval(self,x) -> float:
        if self.sidedness == "both":
            if x > self.a and x <= self.m:
                return self.max * (x - self.a) / (self.m - self.a)
            if x > self.m and x <= self.b:
                return self.max * (self.b - x) / (self.b - self.b)
            else:
                return 0
            
        elif self.sidedness == "left":
            if x > self.a and x <= self.m:
                return self.max * (x - self.a) / (self.m - self.a)
            else:
                return 0
            
        else:
            if x > self.m and x <= self.b:
                return self.max * (self.b - x) / (self.b - self.b)
            else:
               return 0
    

    def plot(self) -> None:
        if self.sidedness == "both":
            l = (self.b - self.a) / 3
            x1 = [self.a - l, self.a]
            x2 = [self.a, self.m]
            x3 = [self.m, self.b]
            x4 = [self.b, self.b + l]


            plt.plot(x1,[0, 0], "b",
                    x2, [0, self.max], "b",
                    x3, [self.max, 0], "b", 
                    x4, [0,0], "b"
                    )
        elif self.sidedness == "left":
            l = (self.m - self.a) / 2
            x1 = [self.a - l, self.a]
            x2 = [self.a, self.m]
            x3 = [self.m, self.m + l]

            plt.plot(x1,[0, 0], "b",
                    x2, [0, self.max], "b",
                    x3, [0,0], "b"
                    )
        else:
            l = (self.b - self.m) / 2
            x1 = [self.m - l, self.m]
            x2 = [self.m, self.b]
            x3 = [self.b, self.b + l]

            plt.plot(x1,[0, 0], "b",
                    x2, [self.max, 0], "b",
                    x3, [0,0], "b"
                    )    
        
        plt.title(f"{self}")
        plt.grid(visible=True, which="both")
        plt.show()

    
    def differentiate(self):
        """differentiates the hat function, result is a Simple_derivative"""
        if self.sidedness == "left":
            value_0 = self.max / (self.m - self.a)
            return Simple_derivative([self.a, self.m, self.m + 1], [value_0, 0])
        elif self.sidedness == "right":
            value_1 = - self.max / (self.b - self.m)
            return Simple_derivative([self.m - 1, self.m, self.b], [0, value_1])
        
        value_0 = self.max / (self.m - self.a)
        value_1 = - self.max / (self.b - self.m)
        return Simple_derivative([self.a, self.m, self.b], [value_0, value_1])


class Simple_derivative():
    """
    class that represents derivatives of hat-functions
    careful: allways the whole function is created;
    (important when assembling bilinearform only half the function is nedeed from 1st and last basis-function

    Instance Variables
    ======
    self.a ... starting point of the interval the function is defined on
    self.m ... middle point of the interval
    self.b ... endpoint of the function
    self.value_0 ... value of the function on the interval [self.a,self.m]
    self.value_1 ... value of the function on the interval [self.m, self.b]
    """

    def __init__(self, basis_points: list, values: list) -> None:
        """
        Parameters
        ======
        basis_points: list ... list of length 3, with the starting, middle and endpoint of the interval on which the function is defined
        values: list ... list of length 2 of the values the funciton has on the interval [a,m] and [m,b], where a,m,b are start, middle and endpoint of the interval defined in basis_points
        """

        assert len(basis_points) == 3, "3 basis_points must be provided"
        assert basis_points[0] < basis_points[1] and basis_points[1] < basis_points[2], "basis_points must be provided in ascending order" 

        self.a = basis_points[0]
        self.m = basis_points[1]
        self.b = basis_points[2]
        self.value_0 = values[0]
        self.value_1 = values[1]


    def __repr__(self) -> str:
        
        return f"Simple_derivative([{self.a},{self.m},{self.b},[{self.value_0},{self.value_1}]])"
    


    def __str__(self) -> str:

        if self.value_0 == 0:
            return f"Piecewise( {self.value_1} for x in [{self.m},{self.b}] \n 0 else)"
        
        if self.value_1 == 0:
            return f"Piecewise( {self.value_0} for x in [{self.a},{self.m}] \n 0 else)"
        
        val = f"Piecewise( {self.value_0}, if x in [{self.a},{self.m}], \n {self.value_1}, if x in [{self.m},{self.b}])"
        return val
    

    def __mul__(self,other):

        #multiplication with integer or float
        if type(other) == float or type(other) == int:
            return Simple_derivative([self.a, self.m, self.b], [self.value_0 * other, self.value_1 * other])
        
        #try to multiply with other Simple_derivative, if that fails call multiplication method of "other"
        try:
            if [self.a, self.m, self.b] == [other.a, other.m, other.b]:
                res = [self.value_0 * other.value_0, self.value_1 * other.value_1]
                return Simple_derivative([self.a,self.m,self.b],res)

            if [self.a, self.m] == [other.m, other.b]:
                res = self.value_0 * other.value_1
                return Simple_derivative([self.a,self.m,self.b],[res, 0])
        
            if [self.m, self.b] == [other.a, other.m]:
                res = self.value_1 * other.value_0
                return Simple_derivative([self.a,self.m,self.b],[0,res])
            else:
                raise ValueError
        
        except(ValueError):
            raise Exception("ValueError: in order to multiply two instances of Simple_derivative their basis points have to match in at least 2 Positions; Anyhow the result is 0")

        except:
            return NotImplemented


    def __rmul__(self,other):
        
        #multiply with integer or float; raises Error if left side of multiplication is not one of those types
        try:
            if type(other) == int or type(other) == float:
                return Simple_derivative([self.a, self.m, self.b], [self.value_0 * other, self.value_1 * other])
            else:
                raise TypeError
        
        except(TypeError):
            return NotImplemented
    

    @staticmethod
    def float_mul(object_1 ,object_2):
        return object_1 * object_2


    def eval(self,x : float) -> float:
        """evaluates the instance at a point x"""

        assert type(x) == int or type(x) == float, "Input has to be int or float"

        if x > self.a and x <= self.m:
            return self.value_0
        
        if x > self.m and x <= self.b:
            return self.value_1
        
        else:
            return 0

    
    def plot(self) -> None:
        """plot the given derivative function"""

        l = (self.b - self.a) / 3
        x1 = [self.a - l, self.a]
        x2 = [self.a, self.m]
        x3 = [self.m, self.b]
        x4 = [self.b, self.b + l]


        plt.plot(x1,[0,0], "b",
                x2, [self.value_0,self.value_0], "b",
                x3, [self.value_1,self.value_1], "b", 
                x4, [0,0], "b"
                )
        
        plt.title(f"{self}")
        plt.grid(visible=True, which="both")
        plt.show()

    
    def integrate(self) -> float:
        """integrate the given derivative function"""

        res = (self.m - self.a) * self.value_0 + (self.b - self.m) * self.value_1

        return res

    
class Simple_quadratic():
    """
    class that represents products of hat-functions

    Instance Variables
    ======
    self.a ... starting point of the interval the function is defined on
    self.m ... middle point of the interval
    self.b ... endpoint of the function
    self.coeff ...  Coefficient of the quadratic function. (eg.: if represented as coeff * (x-a)(m-x) / (m-a)²)
    self.side ...   which side of the function is looked at (eg.: if left, then only the interval [a,m] is relevant)
                    (stems from the fact, that a quadratic funtion is only obtained if 2 Simple_functions are multiplied)
    """

    def __init__(self, basis_points: list, coeff: float, type: str) -> None:
        """
        Parameters
        ======
        basis_points: list ... list of length 3, with the starting, middle and endpoint of the interval on which the function is defined
        coeff ... Coefficient of the quadratic function. (eg.: if represented as coeff * (x-a)(m-x) / (m-a)²)
        type ... which type of simple function are multiplied, if two left sided simple functions are multiplied type is asc, desc if two right sided are multiplied,
                        both if mixed simple function are multiplied
        """
        assert len(basis_points) == 2, "2 basis_points must be provided"
        assert basis_points[0] < basis_points[1], "basis_points must be provided in ascending order"
        assert type in ["asc", "desc", "both"], "sidedness has to be either 'left', 'right', or 'both'."
        
        self.a = basis_points[0]
        self.b = basis_points[1]
        self.coeff = coeff
        self.type = type
        

    def __repr__(self) -> str:
        return f"Simple_quadratic([{self.a}, {self.b}], coeff = {self.coeff}, type = '{self.type}')"

    
    def __str__(self) -> str:
        
        if self.type == "asc":
            rep = f"Piecewise( {self.coeff} * ((x - {self.a}) / {(self.b - self.a)}**2, if x in [{self.a},{self.b}] \n 0, else])"
            return rep
        
        if self.type == "desc":
            rep = f"Piecewise( {self.coeff} * (({self.b} - x) / {(self.b - self.a)})**2, if x in [{self.a},{self.b}] \n 0, else])"
            return rep

        if self.type == "both":
            rep = f"Piecwise( {self.coeff} * (x - {self.a}) * ({self.b} - x) / {(self.b - self.a)**2}, if x in [{self.a}, {self.b}] \n 0, else)"
            return rep
    

    def __mul__(self, other):

        if type(other) == float or type(other) == int:
            return Simple_quadratic([self.a, self.b], other * self.coeff, self.type)
        
        return NotImplemented
    

    def __rmul__(self, other):

        if type(other) == float or type(other) == int:
            return Simple_quadratic([self.a, self.b], other * self.coeff, self.type)
        
        return NotImplemented

    
    def eval(self, x: float | int) -> float:
        """evaluate the given function at point x"""

        if self.type == "both":
            if x >= self.a and x <= self.b:
                return self.coeff * (x - self.a)*(self.b - x) / (self.b - self.a)**2
 
            return 0
        
        if self.type == "asc":
            if x >= self.a and x <= self.b:
                return self.coeff * ((x - self.a) / (self.b- self.a)) ** 2
            
            return 0

        if self.type == "desc":
            if x>= self.a and x<= self.b:
                return self.coeff * ((self.b - x) / (self.b - self.a))**2

            return 0
    

    def plot(self) -> None:
        """plots the given function on the relevant interval"""

        
        x = np.linspace(self.a, self.b, 100)
        
        y = [self.eval(elem) for elem in x]
        l = (self.b - self.a) / 3
        
        plt.plot([x[0] - l, x[0]], [0,0], "b",
            x, y, "b",
            [x[-1], x[-1] + l], [0,0], "b",
            )
            
        plt.title(f"{self}")
        plt.grid(visible=True, which="both")
        plt.show()

    
    def integrate(self) -> float:
        """integrates the given quadratic function on the whole real Axis (in reality only the part on [a,b] different from 0)"""
        
        if self.type in ["asc", "desc"]:
            return self.coeff * (self.b - self.a) / 3
        
        elif self.type == "both":
            return self.coeff * (self.b - self.a) / 6


class Polynomial:

    def __init__(self, order: int, type: str = "standard", coeffs: np.array = None):
        assert order>= 0, "n has to be bigger or equal to 0"
        self.order = order
        self.coeffs: np.array = None
        self.type = type
        if coeffs is not None:
            self.coeffs = coeffs
        if type == "Legendre":
            self.coeffs = self.initialize_legendre_coeffs(order)
        if type == "Integrated_Legendre":
            assert order>= 1, "n has to be bigger or equal to 1"
            self.coeffs = self.initialize_integrated_legendre_coeffs(order)
        assert self.coeffs.shape[0] == self.order + 1, "length of coefficient vektor has to match order of polynomial."


    def __mul__(self, other, **kwargs):
        if type(other) == float or type(other) == int:
            coeffs = self.coeffs * other
            return Polynomial(order=self.order, type=self.type, coeffs=coeffs)
        
        try:
            order = self.order + other.order
            res_coeffs = np.polymul(self.coeffs[::-1], other.coeffs[::-1])
            return Polynomial(order, coeffs=res_coeffs[::-1])

        except:
            return NotImplemented
        

    def __rmul__(self, other, **kwargs):
        if type(self) == float or type(self) == int:
            coeffs = other.coeffs * self
            return Polynomial(order=self.order, type=self.type, coeff=coeffs)
        
        return NotImplemented
    

    @staticmethod
    def float_mul(object_1 ,object_2, **kwargs):
        return object_2 * object_1


    def transform_to_interval(self, a, b):
        a_hat = 2/(b-a)
        b_hat = -(b+a)/(b-a)
        coeffs = np.zeros((self.order + 1,))
        for i in range(self.order+1):
            for j in range(i, self.order+1):
                coeffs[i] += binom(j,i) * self.coeffs[j] * a_hat**i * b_hat**(j-i)
        return Polynomial(order=self.order, coeffs=coeffs)


    def differentiate(self, **kwargs):
        if self.type == "Integrated_Legendre":
            return Polynomial(order=self.order-1, type="Legendre")
        
        coeffs = []
        for i in range(self.order):
            coeffs.append(self.coeffs[i+1] * (i+1))
        coeffs = np.array(coeffs)
        return Polynomial(order=self.order-1, coeffs=coeffs)


    def integrate(self, a, b):
        res = 0
        for i in range(self.order+1):
            res += self.coeffs[i] / (i+1) * (b**(i+1)-a**(i+1))
        return res

    def sympify(self):
        func = sym.Poly(self.coeffs[::-1], sym.Symbol("x"))
        return func

    @staticmethod
    def initialize_legendre_coeffs(n):
        if n == 0:
            return np.array([1])
        if n == 1:
            return np.array([0,1])
        
        coeffs_n_1 = Polynomial.initialize_legendre_coeffs(n-1)
        coeffs_n_1 = coeffs_n_1 * ((2*n - 1) / n)
        coeffs_n_1 = np.pad(coeffs_n_1, (1,0), "constant", constant_values=(0,0))
        coeffs_n_2 = Polynomial.initialize_legendre_coeffs(n-2)
        coeffs_n_2 = coeffs_n_2 * ((n-1) / n)
        temp = coeffs_n_1.shape[0] - coeffs_n_2.shape[0]
        coeffs_n_2 = np.pad(coeffs_n_2, (0,temp),"constant", constant_values=(0,0))

        return coeffs_n_1 - coeffs_n_2


    @staticmethod
    def initialize_integrated_legendre_coeffs(n):
        if n == 1:
            return np.array([0,1])
        if n == 2:
            return np.array([-0.5, 0, 0.5])
        

        coeffs_n_1 = Polynomial.initialize_integrated_legendre_coeffs(n-1)
        coeffs_n_1 = coeffs_n_1 * ((2*n - 3) / n)
        coeffs_n_1 = np.pad(coeffs_n_1, (1,0), "constant", constant_values=(0,0))
        coeffs_n_2 = Polynomial.initialize_integrated_legendre_coeffs(n-2)
        coeffs_n_2 = coeffs_n_2 * ((n-3) / n)
        temp = coeffs_n_1.shape[0] - coeffs_n_2.shape[0]
        coeffs_n_2 = np.pad(coeffs_n_2, (0,temp),"constant", constant_values=(0,0))

        return coeffs_n_1 - coeffs_n_2
    

    def eval(self, x: float) -> float:
        return np.polyval(self.coeffs[::-1], x)


    def plot(self, a: float = -1, b: float = 1):
        x = np.linspace(a,b, 1000)
        eval_vectorized = np.vectorize(self.eval)
        y = eval_vectorized(x)
        plt.plot(x,y)
        plt.grid(color="k", linewidth=0.1)
        plt.show()


    @staticmethod
    def plot_multiple_legendre(orders: list, a: float=-1, b: float=1, safe=False):
        x = np.linspace(a,b, 1000)

        for order in orders:
            p = Polynomial(order=order, type="Legendre")
            eval_vectorized = np.vectorize(p.eval)
            y = eval_vectorized(x)
            plt.plot(x,y)

        plt.grid(color="k", linewidth=0.05)

        if safe:
            plt.savefig("Legendre_Polynome.eps", format="eps")
        else:
            plt.show()
    

    @staticmethod
    def plot_multiple_integrated_legendre(orders: list, a: float=-1, b: float=1, safe=False):
        x = np.linspace(a,b, 1000)

        for order in orders:
            p = Polynomial(order=order, type="Integrated_Legendre")
            eval_vectorized = np.vectorize(p.eval)
            y = eval_vectorized(x)
            plt.plot(x,y)

        plt.grid(color="k", linewidth=0.1)

        if safe:
            plt.savefig("Legendre_Polynome.eps", format="eps")
        else:
            plt.show()

