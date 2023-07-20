from Simple_Functions import *
from re import finditer, findall, escape
from functools import reduce
import matplotlib.pyplot as plt
import networkx as nx


class Node:
    
    def __init__(
        self,
        name: str,
        priority:int,
        operation = None,
        unary = False,
        ) -> None:

        self.name = name
        self.priority = priority
        self.operation = operation
        self.left_child = None  
        self.right_child = None
        self.unary = unary


    def __repr__(self) -> str:
        return f"""Node(name = {self.name}, priority = {self.priority}, operation = {self.operation}, unary = {self.unary})"""
        

    def __str__(self):
        return self.name


    def execute(self, Trialfunction: Simple_function = None, Testfunction: Simple_function = None, **kwargs):
        if self.name == "u":
            return Trialfunction
        if self.name == "v":
            return Testfunction
        if findall(r"[-]?(?:[1-9][0-9]*|0)(?:\/[1-9][0-9]*)?", self.name):
            if "/" in self.name:
                numerator, denominator = self.name.split("/")
                return float(numerator) / float(denominator)
            return float(self.name)
        
        if self.unary:
            if self.left_child.name == "u":
                executed_child_node = self.left_child.execute(Trialfunction = Trialfunction, **kwargs)
                return self.operation(executed_child_node, **kwargs)
            if self.left_child.name == "v":
                executed_child_node = self.left_child.execute(Testfunction = Testfunction, **kwargs)
                return self.operation(executed_child_node, **kwargs)
            if self.left_child.name == "*":
                executed_child_node = self.left_child.execute(Trialfunction, Testfunction, **kwargs)
                return self.operation(executed_child_node, **kwargs)

        executed_left_child = self.left_child.execute(Trialfunction, Testfunction, **kwargs)
        executed_right_child = self.right_child.execute(Trialfunction, Testfunction, **kwargs)
        return self.operation(
            executed_left_child,
            executed_right_child,
            **kwargs
            )
    

class Operations_tree:

    def __init__(
        self,
        nodes: list[Node] | None = None,
        operation_string: str | None = None,
        execution_order: list[list[Node]] | None = None,
        ) -> None:

        self.nodes = nodes
        self.execution_order = execution_order

        if operation_string:
            self.nodes = Operations_tree.build_tree(operation_string)
    

    def __repr__(self) -> str:
        tokenized_string = [node.name for node in self.nodes]
        operation_string = reduce(lambda x,y: x+y, tokenized_string)
        return operation_string
    
    
    def draw(self) -> None:
        edges = []
        for node in self.nodes:
            if node.left_child:
                edges.append([node, node.left_child])
            if node.right_child:
                edges.append([node, node.right_child])

        G = nx.Graph()
        G.add_edges_from(edges)
        nx.draw_networkx(G)
        plt.show()
    

    def walk_through_tree(self, node: Node | None = None):
        if not node:
            node = self.find_root()
        
        stack = [node]

        while stack:
            node = stack.pop() #pop(0) if first in first out

            yield node

            if node.right_child:
                stack.append(node.right_child)
            if node.left_child:
                stack.append(node.left_child)
    
    
    def find_root(self) -> Node:
        nodes = list(reversed(self.nodes))
        root = nodes.pop()

        for node in nodes:
            if node.left_child == root or node.right_child == root:
                root = node
        
        return root


    def execute_tree(self, Trialfunction: Simple_function, Testfunction: Simple_function):
        self.initialize_operations()
        root = self.find_root()
        
        return root.execute(Trialfunction, Testfunction)

    def execute_tree_v2(self, Trialfunction: Simple_function, Testfunction: Simple_function, **kwargs):
        self.initialize_operations_v2()
        root = self.find_root()
        
        return root.execute(Trialfunction, Testfunction, **kwargs)


    def get_execution_layers(self) -> None:

        root = self.find_root()
        layers = []
        layers.append([root])
        stack = [[root]]

        while stack:
            #print(stack)
            nodes = stack.pop()
            children = []

            for node in nodes:
                if node.left_child:
                    children.append(node.left_child)
                if node.right_child:
                    children.append(node.right_child)

            if not children:
                break

            layers.append(children)
            stack.append(children)
        
        return layers


    def initialize_operations(self):
        for node in self.nodes:
            if node.name == "+":
                node.operation = float.__add__
            if node.name == "*":
                if node.left_child.name == "grad" or node.right_child.name == "grad":
                    node.operation = Simple_derivative.__mul__
                elif node.left_child.name in ["u", "v"] or node.right_child.name in ["u","v"]:
                    node.operation = Simple_function.__mul__
                else:
                    for walk_node in self.walk_through_tree(node):
                        if walk_node.name in ["u", "v"]:
                            node.operation = Simple_function.float_mul
                            break
                        if walk_node.name == "grad":
                            node.operation = Simple_derivative.float_mul
                            break
                    else:
                        raise Exception(f"Could not find right operation for node {node}")
            if node.name == "integrate":
                if node.left_child.left_child.name == "grad":
                    node.operation = Simple_derivative.integrate
                elif node.left_child.left_child.name == "u":
                    node.operation = Simple_quadratic.integrate
                else:
                    for walk_node in self.walk_through_tree(node):
                        if walk_node.name in ["u", "v"]:
                            node.operation = Simple_quadratic.integrate
                            break
                        if walk_node.name == "grad":
                            node.operation = Simple_derivative.integrate
                            break
                    else:
                        raise Exception(f"Could not find right operation for node {node}")
            if node.name == "grad":
                node.operation = Simple_function.differentiate

    def initialize_operations_v2(self):
        for node in self.nodes:
            if node.name == "+":
                def improvised_addition(x,y,**kwargs):
                    return x+y
                node.operation = improvised_addition
            if node.name == "*":
                if node.left_child.name == "grad" or node.right_child.name == "grad":
                    node.operation = Polynomial.__mul__
                else:
                    node.operation = Polynomial.float_mul
            if node.name == "integrate":
                node.operation = Polynomial.integrate
            if node.name == "grad":
                node.operation = Polynomial.differentiate


    @staticmethod
    def build_tree(operation_string: str) -> list[Node]:
        
        nodes: list[Node] = []
        #operations with their respective priority, following "punkt vor strich"
        operators = {"+": 4, "integrate": 3, "*": 2, "grad": 1, "u": 0, "v": 0}
        words = []
        word_positions = []

        for word in operators.keys():
            words = words + findall(escape(word), operation_string)
            word_positions = word_positions + [item.start() for item in finditer(escape(word), operation_string)]
        words = words + findall(r"[-]?(?:[1-9][0-9]*|0)(?:\/[1-9][0-9]*)?", operation_string)
        word_positions = word_positions + [item.start() for item in finditer(r"[-]?(?:[1-9][0-9]*|0)(?:\/[1-9][0-9]*)?", operation_string)]
    
        tokenized_string = [word for (word, _) in sorted(zip(words,word_positions), key= lambda tup: tup[1])]
        
        for token in tokenized_string:

            if token == "+":
                new_node = Node(name="+", priority= operators["+"])
            elif token == "*":
                new_node = Node(name= "*", priority= operators["*"])
            elif token == "grad":
                new_node = Node(name= "grad", priority= operators["grad"], unary = True)
            elif token == "integrate":
                new_node = Node(name= "integrate", priority= operators["integrate"], unary = True)
            elif token == "u":
                new_node = Node(name= "u", priority= operators["u"], unary = True)
            elif token == "v":
                new_node = Node(name= "v", priority= operators["v"], unary = True)
            else:
                #those tokens represent numbers
                new_node = Node(name = token, priority=0, unary=True)
            

            highest_prio_node = Node(name = "_", priority=-1)
            for node in nodes[::-1]:

                #check the nodes priority, if it has higher priority than the previous node with highest priority update it
                if node.priority > highest_prio_node.priority:
                    highest_prio_node = node
                
                #if the priority of the new node is smaller than one of the previous nodes, assign it as a child
                if node.priority >= new_node.priority:
                    #if node is unary the new node gets inserted as a node between the node and its child (the child of node becomes the child of new node, the new node becomes child of node)
                    if node.unary:
                        new_node.left_child = node.left_child
                        node.left_child = new_node
                        break
                    #if the node already has a left child (and is not unary) check if it already has a right child, if thats the case raise exception, else node get new node as right child
                    if node.left_child:
                        if node.right_child:
                            if node.right_child.priority <= new_node.priority:
                                new_node.left_child = node.right_child
                                node.right_child = new_node
                                break
                            raise Exception(f"something broke, the node {node.operation} already seems to have children with higher priority than {new_node.operation} ")
                        node.right_child = new_node
                        break
                    #if node does not have left child, and is not unary assign new node as left child
                    node.left_child = new_node
                    break
                #if the priority is not smaller continue with the next node 
                continue
            
            #if the for loop doesnt break, it means that the new node is the root of the tree, meaning the previus highest priority node is a child of the new node
            if new_node.priority > highest_prio_node.priority and highest_prio_node.name != "_":
                new_node.left_child = highest_prio_node
            
            #at last dd the new node to the existing nodes of the tree
            nodes.append(new_node)

        return nodes
            
