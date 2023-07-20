import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import math as m
from pprint import pprint

from scipy.linalg import solve_banded, solve

from Simple_Functions import *
from operation_tree import *
from Funktionen_Klasse import *


class Fem:
    
    def __init__(
            self,
            bilinearform,
            linearform,
            type: str = "Neumann",
            grid: list | None = None,
            boundary_values: list | None = None
            ):
        self.bilinearform = bilinearform
        self.assembled_bilineaform = None
        self.lineaform = linearform
        self.assembled_lineaform = None
        self.type = type
        self.grid = grid
        self.boundary_values = boundary_values
        self._debug_solution = None
    

    def assemble(self):
        n = len(self.grid)
        assert n>=2, "must provide at least 2 points"
        operations = Operations_tree(operation_string=self.bilinearform)
        
        t_start = perf_counter()
        M = np.zeros((n,n))
        building_block = np.zeros((2,2))
        for i in range(n-1):
            building_block[0,0] = operations.execute_tree(
                Simple_function([None, self.grid[i], self.grid[i+1]], sidedness="right"),
                Simple_function([None, self.grid[i], self.grid[i+1]], sidedness="right")
            )
            building_block[1,1] = operations.execute_tree(
                Simple_function([self.grid[i], self.grid[i+1], None], sidedness="left"),
                Simple_function([self.grid[i], self.grid[i+1], None], sidedness="left")
            )
            building_block[0,1] = operations.execute_tree(
                Simple_function([self.grid[i], self.grid[i+1], None], sidedness="left"),
                Simple_function([None, self.grid[i], self.grid[i+1]], sidedness="right"),
            )
            building_block[1,0] = building_block[0,1]

            M[i:i+2, i:i+2] += building_block
            
        self.assembled_bilineaform = M
        t_end = perf_counter()
        #print(f"Assembling Matrix took {t_end - t_start} seconds")
        
        if self.type == "Dirichlet":
            self.assembled_bilineaform[0] = np.eye(1,n)
            self.assembled_bilineaform[n-1] = np.eye(1,n,n-1)
            ##Speedup????
            t_start = perf_counter()
            y = np.zeros((n,))
            y[0] = self.boundary_values[0]
            y[n-1] = self.boundary_values[1]
            for i in range(1,n-1):
                dif1 = self.grid[i] - self.grid[i-1]
                dif2 = self.grid[i+1] - self.grid[i]
                f1 = self.lineaform * sym.sympify(f"(x - {self.grid[i-1]}) / {dif1}")
                f2 = self.lineaform * sym.sympify(f"({self.grid[i+1]} - x) / {dif2}")
                y[i] = f1.quadrature(self.grid[i-1], self.grid[i]) + f2.quadrature(self.grid[i], self.grid[i+1])
            self.assembled_lineaform = y
            t_end = perf_counter()
            #print(f"Assembling right side vector took {t_end - t_start} seconds")
            
        if self.type == "Neumann":
            t_start = perf_counter()
            y = np.zeros((n,))
            dif1 = self.grid[1] - self.grid[0]
            dif2 = self.grid[n-1] - self.grid[n-2]
            f1 = self.lineaform * sym.sympify(f"({self.grid[1]} - x) / {dif1}")
            f2 = self.lineaform * sym.sympify(f"(x - {self.grid[n-2]}) / {dif2}")
            y[0] = f1.quadrature(self.grid[0], self.grid[1]) - self.boundary_values[0]
            y[n-1] = f2.quadrature(self.grid[n-2], self.grid[n-1]) + self.boundary_values[1]
            for i in range(1,n-1):
                dif1 = self.grid[i] - self.grid[i-1]
                dif2 = self.grid[i+1] - self.grid[i]
                f1 = self.lineaform * sym.sympify(f"(x - {self.grid[i-1]}) / {dif1}")
                f2 = self.lineaform * sym.sympify(f"({self.grid[i+1]} - x) / {dif2}")
                y[i] = f1.quadrature(self.grid[i-1], self.grid[i]) + f2.quadrature(self.grid[i], self.grid[i+1])
            self.assembled_lineaform = y
            t_end = perf_counter()
            #print(f"Assembling right side vector took {t_end - t_start} seconds")

        if self.type == "Left-Dirichlet":
            t_start = perf_counter()
            y = np.zeros((n,))
            self.assembled_bilineaform[0] = np.eye(1,n)
            dif = self.grid[n-1] - self.grid[n-2]
            f = self.lineaform * sym.sympify(f"(x - {self.grid[n-2]}) / {dif}")
            y[0] = self.boundary_values[0]
            y[n-1] = f.quadrature(self.grid[n-2], self.grid[n-1]) + self.boundary_values[1]
            for i in range(1,n-1):
                dif1 = self.grid[i] - self.grid[i-1]
                dif2 = self.grid[i+1] - self.grid[i]
                f1 = self.lineaform * sym.sympify(f"(x - {self.grid[i-1]}) / {dif1}")
                f2 = self.lineaform * sym.sympify(f"({self.grid[i+1]} - x) / {dif2}")
                y[i] = f1.quadrature(self.grid[i-1], self.grid[i]) + f2.quadrature(self.grid[i], self.grid[i+1])
            self.assembled_lineaform = y
            t_end = perf_counter()
            #print(f"Assembling right side vector took {t_end - t_start} seconds")
        
        if self.type == "Right-Dirichlet":
            t_start = perf_counter()
            y = np.zeros((n,))
            self.assembled_bilineaform[n-1] = np.eye(1,n,n-1)
            dif = self.grid[1] - self.grid[0]
            f = self.lineaform * sym.sympify(f"({self.grid[1]} - x) / {dif}")
            y[0] = f.quadrature(self.grid[0], self.grid[1]) - self.boundary_values[0]
            y[n-1] = self.boundary_values[1]
            for i in range(1,n-1):
                dif1 = self.grid[i] - self.grid[i-1]
                dif2 = self.grid[i+1] - self.grid[i]
                f1 = self.lineaform * sym.sympify(f"(x - {self.grid[i-1]}) / {dif1}")
                f2 = self.lineaform * sym.sympify(f"({self.grid[i+1]} - x) / {dif2}")
                y[i] = f1.quadrature(self.grid[i-1], self.grid[i]) + f2.quadrature(self.grid[i], self.grid[i+1])
            self.assembled_lineaform = y
            t_end = perf_counter()
            #print(f"Assembling right side vector took {t_end - t_start} seconds")

    def assemble_v2(self, order: int=1):
        n = len(self.grid)
        assert n>=2, "must provide at least 2 points"
        operations = Operations_tree(operation_string=self.bilinearform)
        if order == 1:
            M = np.zeros((n,n))
            building_block = np.zeros((2,2))
            for i in range(n-1):
                coeffs_right = [self.grid[i+1]/(self.grid[i+1]-self.grid[i]), -1 / (self.grid[i+1]-self.grid[i])]
                coeffs_left = [-self.grid[i]/(self.grid[i+1]-self.grid[i]), 1 / (self.grid[i+1]-self.grid[i])]
                building_block[0,0] = operations.execute_tree_v2(
                    Polynomial(order=1, coeffs=np.array(coeffs_right)),
                    Polynomial(order=1, coeffs=np.array(coeffs_right)), a=self.grid[i], b=self.grid[i+1]
                )
                building_block[1,1] = operations.execute_tree_v2(
                    Polynomial(order=1, coeffs=np.array(coeffs_left)),
                    Polynomial(order=1, coeffs=np.array(coeffs_left)), a=self.grid[i], b=self.grid[i+1]
                )
                building_block[0,1] = operations.execute_tree_v2(
                    Polynomial(order=1, coeffs=np.array(coeffs_left)),
                    Polynomial(order=1, coeffs=np.array(coeffs_right)),a=self.grid[i], b=self.grid[i+1]
                )
                building_block[1,0] = building_block[0,1]

                M[i:i+2, i:i+2] += building_block
                
            self.assembled_bilineaform = M
        
            if self.type == "Dirichlet":
                self.assembled_bilineaform[0] = np.eye(1,n)
                self.assembled_bilineaform[n-1] = np.eye(1,n,n-1)
                y = np.zeros((n,))
                y[0] = self.boundary_values[0]
                y[n-1] = self.boundary_values[1]
                for i in range(1,n-1):
                    dif1 = self.grid[i] - self.grid[i-1]
                    dif2 = self.grid[i+1] - self.grid[i]
                    f1 = self.lineaform * sym.sympify(f"(x - {self.grid[i-1]}) / {dif1}")
                    f2 = self.lineaform * sym.sympify(f"({self.grid[i+1]} - x) / {dif2}")
                    y[i] = f1.quadrature(self.grid[i-1], self.grid[i]) + f2.quadrature(self.grid[i], self.grid[i+1])
                self.assembled_lineaform = y
                
            if self.type == "Neumann":
                y = np.zeros((n,))
                dif1 = self.grid[1] - self.grid[0]
                dif2 = self.grid[n-1] - self.grid[n-2]
                f1 = self.lineaform * sym.sympify(f"({self.grid[1]} - x) / {dif1}")
                f2 = self.lineaform * sym.sympify(f"(x - {self.grid[n-2]}) / {dif2}")
                y[0] = f1.quadrature(self.grid[0], self.grid[1]) - self.boundary_values[0]
                y[n-1] = f2.quadrature(self.grid[n-2], self.grid[n-1]) + self.boundary_values[1]
                for i in range(1,n-1):
                    dif1 = self.grid[i] - self.grid[i-1]
                    dif2 = self.grid[i+1] - self.grid[i]
                    f1 = self.lineaform * sym.sympify(f"(x - {self.grid[i-1]}) / {dif1}")
                    f2 = self.lineaform * sym.sympify(f"({self.grid[i+1]} - x) / {dif2}")
                    y[i] = f1.quadrature(self.grid[i-1], self.grid[i]) + f2.quadrature(self.grid[i], self.grid[i+1])
                self.assembled_lineaform = y

            if self.type == "Left-Dirichlet":
                y = np.zeros((n,))
                self.assembled_bilineaform[0] = np.eye(1,n)
                dif = self.grid[n-1] - self.grid[n-2]
                f = self.lineaform * sym.sympify(f"(x - {self.grid[n-2]}) / {dif}")
                y[0] = self.boundary_values[0]
                y[n-1] = f.quadrature(self.grid[n-2], self.grid[n-1]) + self.boundary_values[1]
                for i in range(1,n-1):
                    dif1 = self.grid[i] - self.grid[i-1]
                    dif2 = self.grid[i+1] - self.grid[i]
                    f1 = self.lineaform * sym.sympify(f"(x - {self.grid[i-1]}) / {dif1}")
                    f2 = self.lineaform * sym.sympify(f"({self.grid[i+1]} - x) / {dif2}")
                    y[i] = f1.quadrature(self.grid[i-1], self.grid[i]) + f2.quadrature(self.grid[i], self.grid[i+1])
                self.assembled_lineaform = y
            
            if self.type == "Right-Dirichlet":
                y = np.zeros((n,))
                self.assembled_bilineaform[n-1] = np.eye(1,n,n-1)
                dif = self.grid[1] - self.grid[0]
                f = self.lineaform * sym.sympify(f"({self.grid[1]} - x) / {dif}")
                y[0] = f.quadrature(self.grid[0], self.grid[1]) - self.boundary_values[0]
                y[n-1] = self.boundary_values[1]
                for i in range(1,n-1):
                    dif1 = self.grid[i] - self.grid[i-1]
                    dif2 = self.grid[i+1] - self.grid[i]
                    f1 = self.lineaform * sym.sympify(f"(x - {self.grid[i-1]}) / {dif1}")
                    f2 = self.lineaform * sym.sympify(f"({self.grid[i+1]} - x) / {dif2}")
                    y[i] = f1.quadrature(self.grid[i-1], self.grid[i]) + f2.quadrature(self.grid[i], self.grid[i+1])
                self.assembled_lineaform = y


        if order == 2:
            M = np.zeros((2*n-1,2*n-1))
            Y = np.zeros((2*n-1,))
            building_block = np.zeros((3,3))
            vector_block = np.zeros((3,))
            for i in range(0,n-1):
                coeffs_right = [self.grid[i+1]/(self.grid[i+1]-self.grid[i]), -1 / (self.grid[i+1]-self.grid[i])]
                coeffs_left = [-self.grid[i]/(self.grid[i+1]-self.grid[i]), 1 / (self.grid[i+1]-self.grid[i])]
                int_legendre = Polynomial(order=2, type="Integrated_Legendre")
                int_legendre = int_legendre.transform_to_interval(self.grid[i], self.grid[i+1])
                left_pol = Polynomial(order=1, coeffs=np.array(coeffs_left))
                right_pol = Polynomial(order=1, coeffs=np.array(coeffs_right))
                building_block[0,0] = operations.execute_tree_v2(right_pol, right_pol, a=self.grid[i], b=self.grid[i+1])
                building_block[1,1] = operations.execute_tree_v2(int_legendre, int_legendre, a=self.grid[i], b=self.grid[i+1])
                building_block[2,2] = operations.execute_tree_v2(left_pol,left_pol, a=self.grid[i], b=self.grid[i+1])
                building_block[1,0] = operations.execute_tree_v2(int_legendre,right_pol, a=self.grid[i], b=self.grid[i+1])
                building_block[2,0] = operations.execute_tree_v2(left_pol, right_pol, a=self.grid[i], b=self.grid[i+1])
                building_block[1,2] = operations.execute_tree_v2(int_legendre,left_pol, a=self.grid[i], b=self.grid[i+1])
                building_block[0,1] = building_block[1,0]
                building_block[0,2] = building_block[2,0]
                building_block[2,1] = building_block[1,2]
                M[2*i:2*i+3, 2*i:2*i+3] += building_block
                
                int_legendre = int_legendre.sympify()
                int_legendre = int_legendre.as_expr() * self.lineaform
                left_pol = left_pol.sympify()
                left_pol = left_pol.as_expr() * self.lineaform
                right_pol = right_pol.sympify()
                right_pol = right_pol.as_expr() * self.lineaform
                vector_block[0] = right_pol.quadrature(self.grid[i], self.grid[i+1])
                vector_block[1] = int_legendre.quadrature(self.grid[i], self.grid[i+1])
                vector_block[2] = left_pol.quadrature(self.grid[i], self.grid[i+1])
                Y[2*i:2*i+3] += vector_block

            self.assembled_lineaform = Y
            self.assembled_bilineaform = M
            
            if self.type == "Neumann":
                self.assembled_lineaform[0] -= self.boundary_values[0]
                self.assembled_lineaform[-1] += self.boundary_values[1]
            elif self.type == "Dirichlet":
                self.assembled_bilineaform[0] = np.eye(1,2*n-1)
                self.assembled_bilineaform[-1] = np.eye(1,2*n-1,2*n-2)
                self.assembled_lineaform[0] = self.boundary_values[0]
                self.assembled_lineaform[-1] = self.boundary_values[1] 

        if order == 3:
            M = np.zeros((3*n-2,3*n-2))
            Y = np.zeros((3*n-2,))
            building_block = np.zeros((4,4))
            vector_block = np.zeros((4,))
            for i in range(0,n-1):
                coeffs_right = [self.grid[i+1]/(self.grid[i+1]-self.grid[i]), -1 / (self.grid[i+1]-self.grid[i])]
                coeffs_left = [-self.grid[i]/(self.grid[i+1]-self.grid[i]), 1 / (self.grid[i+1]-self.grid[i])]
                int_legendre_1 = Polynomial(order=2, type="Integrated_Legendre")
                int_legendre_1 = int_legendre_1.transform_to_interval(self.grid[i], self.grid[i+1])
                int_legendre_2 = Polynomial(order=3, type="Integrated_Legendre")
                int_legendre_2 = int_legendre_2.transform_to_interval(self.grid[i], self.grid[i+1])
                left_pol = Polynomial(order=1, coeffs=np.array(coeffs_left))
                right_pol = Polynomial(order=1, coeffs=np.array(coeffs_right))
                building_block[0,0] = operations.execute_tree_v2(right_pol, right_pol, a=self.grid[i], b=self.grid[i+1])
                building_block[1,1] = operations.execute_tree_v2(int_legendre_1, int_legendre_1, a=self.grid[i], b=self.grid[i+1])
                building_block[2,2] = operations.execute_tree_v2(int_legendre_2, int_legendre_2, a=self.grid[i], b=self.grid[i+1])
                building_block[3,3] = operations.execute_tree_v2(left_pol,left_pol, a=self.grid[i], b=self.grid[i+1])
                building_block[1,0] = operations.execute_tree_v2(int_legendre_1,right_pol, a=self.grid[i], b=self.grid[i+1])
                building_block[2,0] = operations.execute_tree_v2(int_legendre_2, right_pol, a=self.grid[i], b=self.grid[i+1])
                building_block[3,0] = operations.execute_tree_v2(left_pol,right_pol, a=self.grid[i], b=self.grid[i+1])
                building_block[1,2] = operations.execute_tree_v2(int_legendre_1,int_legendre_2, a=self.grid[i], b=self.grid[i+1])
                building_block[1,3] = operations.execute_tree_v2(int_legendre_1,left_pol, a=self.grid[i], b=self.grid[i+1])
                building_block[2,3] = operations.execute_tree_v2(int_legendre_2,left_pol, a=self.grid[i], b=self.grid[i+1])
                building_block[0,1] = building_block[1,0]
                building_block[0,2] = building_block[2,0]
                building_block[0,3] = building_block[3,0]
                building_block[2,1] = building_block[1,2]
                building_block[3,1] = building_block[1,3]
                building_block[3,2] = building_block[2,3]
                M[3*i:3*i+4, 3*i:3*i+4] += building_block
                
                int_legendre_1 = int_legendre_1.sympify()
                int_legendre_1 = int_legendre_1.as_expr() * self.lineaform
                int_legendre_2 = int_legendre_2.sympify()
                int_legendre_2 = int_legendre_2.as_expr() * self.lineaform
                left_pol = left_pol.sympify()
                left_pol = left_pol.as_expr() * self.lineaform
                right_pol = right_pol.sympify()
                right_pol = right_pol.as_expr() * self.lineaform
                vector_block[0] = right_pol.quadrature(self.grid[i], self.grid[i+1])
                vector_block[1] = int_legendre_1.quadrature(self.grid[i], self.grid[i+1])
                vector_block[2] = int_legendre_2.quadrature(self.grid[i], self.grid[i+1])
                vector_block[3] = left_pol.quadrature(self.grid[i], self.grid[i+1])
                Y[3*i:3*i+4] += vector_block
            self.assembled_lineaform = Y
            self.assembled_bilineaform = M
            
            if self.type == "Neumann":
                self.assembled_lineaform[0] -= self.boundary_values[0]
                self.assembled_lineaform[-1] += self.boundary_values[1]
            elif self.type == "Dirichlet":
                self.assembled_bilineaform[0] = np.eye(1,3*n-2)
                self.assembled_bilineaform[-1] = np.eye(1,3*n-2,3*n-3)
                self.assembled_lineaform[0] = self.boundary_values[0]
                self.assembled_lineaform[-1] = self.boundary_values[1] 

    def dismantle(self):
        self.assembled_bilineaform = None
        self.assembled_lineaform = None


    def solve(self, order =1, plot=True):
        try:
            _ = self.assembled_bilineaform.shape
            _ = self.assembled_lineaform.shape
        except AttributeError:
            self.assemble_v2(order=order)
        result = solve(self.assembled_bilineaform, self.assembled_lineaform)
        result = result[::order]
        if plot:
            plt.figure()
            if self._debug_solution:
                self._debug_solution.plot(self.grid[0], self.grid[-1], show=False)
            plt.plot(self.grid,result, color = "r", lw = 0, marker = "x", markersize=2, rasterized=True)
            plt.grid()
            plt.show()

        return result
    


def create_test_problem(
        bilinearform: str,
        funk,
        grid: list,
        constant: float = 0,
        type: str = "Neumann",
        plot: bool = True,
        order = 1
        ):
    right_hand_side = funk.differentiate("x").differentiate("x")
    right_hand_side = right_hand_side * sym.sympify("-1")
    temp = right_hand_side.funk
    temp = temp + constant * funk.funk
    right_hand_side = funktion(temp, "x")

    if type == "Dirichlet":
        boundary_values = [funk.funk_eval(grid[0]), funk.funk_eval(grid[-1])]
    if type == "Neumann":
        f_diff = funk.differentiate("x")
        boundary_values = [f_diff.funk_eval(grid[0]), f_diff.funk_eval(grid[-1])]
    if type =="Left-Dirichlet":
        f_diff = funk.differentiate("x")
        boundary_values = [funk.funk_eval(grid[0]), f_diff.funk_eval(grid[-1])]
    if type == "Right-Dirichlet":
        f_diff = funk.differentiate("x")
        boundary_values = [f_diff.funk_eval(grid[0]), funk.funk_eval(grid[-1])]

    test_problem = Fem(
        bilinearform=bilinearform,
        linearform=right_hand_side,
        type=type,
        grid=grid,
        boundary_values=boundary_values,
        )
    test_problem._debug_solution = funk
    result = test_problem.solve(plot=plot, order=order)
    
    return test_problem, result
    
    
def plot_l2_error(
        bilinearform: str,
        funk,
        constant: float = 0,
        type: str = "Dirichlet",
        a: float = 0,
        b: float = 5,
        max_n = 50
        ):
    
    errors = []
    step_sizes = []

    for n in range(5,max_n,5):
        grid = np.linspace(a,b,n)
        problem, approximate_solution = create_test_problem(
            bilinearform=bilinearform,
            funk=funk,
            grid=grid,
            constant=constant,
            type=type,
            plot=False,
        )
        exact_solution_l2_approx = funk.l2_projection(
            a=a,
            b=b,
            n=n,
            partition=grid,
        )

        l2_projection_string = "integrate u * v"
        l2_projection_fem = Fem(
            bilinearform=l2_projection_string,
            linearform=funktion("x","x"),
            grid=grid,
            boundary_values=[0,0]
        )
        l2_projection_fem.assemble()
        M = l2_projection_fem.assembled_bilineaform
        vec = approximate_solution - exact_solution_l2_approx
        errors.append(m.sqrt(vec.dot(M).dot(vec)))
        step_sizes.append((b-a)/n)

    plt.loglog(step_sizes, errors, "r", step_sizes, np.array(step_sizes) ** 2, "b")
    plt.title("approximated L2_error of FEM solution")
    plt.xlabel("step_size 'h'")
    plt.ylabel("l2_errors")
    plt.grid(visible=True, which="both")
    plt.show()

    return errors


def plot_h1_error(
        bilinearform: str,
        funk,
        constant: float = 0,
        type: str = "Dirichlet",
        a: float = 0,
        b: float = 5,
        max_n = 50,
        order = 1
        ):
    
    errors = []
    step_sizes = []
        
    for n in range(5,max_n,5):
        grid = np.linspace(a,b,n)
        problem, approximate_solution = create_test_problem(
            bilinearform=bilinearform,
            funk=funk,
            grid=grid,
            constant=constant,
            type=type,
            plot=False,
            order=order
        )
        exact_solution = funk.l2_projection(
            a=a,
            b=b,
            n=n,
            partition=grid,
        )
        #exact_solution = funk.funk_eval(grid)
        if problem.bilinearform == "integrate grad u * grad v + integrate u * v":
            M = problem.assembled_bilineaform
        else:
            h1_projection_string = "integrate grad u * grad v + integrate u * v"
            h1_projection_fem = Fem(
                bilinearform=h1_projection_string,
                linearform=funktion("1","x"),
                grid=grid,
                boundary_values=[0,0]
            )
            h1_projection_fem.assemble()
            M = h1_projection_fem.assembled_bilineaform

        vec = approximate_solution - exact_solution
        errors.append(m.sqrt(abs(vec.dot(M).dot(vec))))
        step_sizes.append((b-a)/n)

    plt.loglog(
        step_sizes, errors, "r", step_sizes,
        np.array(step_sizes)**2, "b",
        step_sizes, np.array(step_sizes), "g",
        step_sizes, np.array(step_sizes)**3, "y")
    plt.title("approximated H1_error of FEM solution")
    plt.xlabel("step_size 'h'")
    plt.ylabel("h1_errors")
    plt.grid(visible=True, which="both")
    plt.show()

    return errors


def plot_h1_error_via_quadrature(
        bilinearform: str,
        funk,
        constant: float = 0,
        type: str = "Dirichlet",
        a: float = 0,
        b: float = 5,
        max_n = 50,
        ):
    
    errors = []
    step_sizes = []
        
    for n in range(5,max_n,5):
        grid = np.linspace(a,b,n)
        mid_points = (grid + np.roll(grid,-1)) / 2
        mid_points = mid_points[:-1]
        steps = grid - np.roll(grid,1)
        steps = steps[1:]
        step_size_reciprocal = np.reciprocal(steps)
        problem, approximate_solution = create_test_problem(
            bilinearform=bilinearform,
            funk=funk,
            grid=grid,
            constant=constant,
            type=type,
            plot=False,
        )
        temp_1 = approximate_solution[:-1] * -1 * step_size_reciprocal
        temp_2 = approximate_solution[1:] * step_size_reciprocal
        approximate_solution_midpoints = temp_1 + temp_2
        exact_solution = funk.differentiate("x")
        exact_solution = exact_solution.funk_eval(mid_points)
        
        error = (approximate_solution_midpoints - exact_solution)**2
        error = m.sqrt(error.dot(steps))
        errors.append(error)
        step_sizes.append((b-a)/n)

    plt.loglog(step_sizes, errors, "r", step_sizes, np.array(step_sizes)**2, "b", step_sizes, np.array(step_sizes), "g")
    plt.title("approximated H1_error of FEM solution")
    plt.xlabel("step_size 'h'")
    plt.ylabel("h1_errors")
    plt.grid(visible=True, which="both")
    plt.show()

    return errors