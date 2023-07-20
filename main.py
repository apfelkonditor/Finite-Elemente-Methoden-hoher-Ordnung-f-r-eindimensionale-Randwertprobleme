from Simple_Functions import *
from operation_tree import *
from pprint import pprint
import numpy as np
import math as m

from fem import *
from Funktionen_Klasse import funktion
from Hilfsfunktionen import *


f = Simple_derivative([1,2,3], [-4,4])
g = Simple_function([-1,3,5], 3, sidedness="both")
g_2 = Simple_function([-1,3,None], 3, sidedness="left")
g_3 = Simple_function([None,-1,3], 3, sidedness="right")
h = Simple_function([4,20,25], 2)

#g.plot()
#g_2.plot()
#g_3.plot()
#print(g_3*g_2)
#(g_3*g_2).plot()
#print(g.differentiate(), "\n", g_2.differentiate(), "\n", g_3.differentiate())




#Tree = Operations_tree(operation_string=string)
#Tree.draw()
#print(Tree)
#Tree.initialize_operations()
#pprint(Tree.get_execution_layers())
#result = Tree.execute_tree(g,h)
#print(result)
#res =  (g*h).integrate()
#print(res)
#print(Tree.find_root())
#print(len(Tree.nodes))
#for node in Tree.walk_through_tree():
#    print(node)

#g = g.differentiate()

#g = Polynomial(order=3, type="Legendre")
#print(g.sympify())
#print(g.coeffs)
#g.plot()
#a,b = -1, 2
#h = g.transform_to_interval(a,b)
#h = g.differentiate()
#h = g.integrate(a,b)
#print(h)

#h.plot(a,b)
#print(h.coeffs)
#print(i.coeffs)
#Polynomial.plot_multiple_legendre([2,3,4,5], safe=True)
#Polynomial.plot_multiple_integrated_legendre([3,4,5])

k = -5
string = f"integrate   grad u * grad v + integrate {k} * u * v"
funk = funktion("sin(x)*x*2", "x")
grid = np.linspace(0, m.pi, 7)
#plot_l2_error(funk,0,m.pi)


_, res = create_test_problem(
    bilinearform=string,
    funk=funk,
    grid=grid,
    constant=k,
    type = "Dirichlet",
    plot = True,
    order = 2
)

#res=plot_l2_error(
#    bilinearform=string,
#    funk=funk,
#    constant = k,
#    a=0,
#   b=m.pi,
#)
#res=plot_h1_error_via_quadrature(
#    bilinearform=string,
#    funk=funk,
#    constant = k,
#    a=0,
#    b=m.pi,
#)
res=plot_h1_error(
    bilinearform=string,
    funk=funk,
    constant = k,
    type="Neumann",
    a=0,
    b=m.pi,
    order = 1
)