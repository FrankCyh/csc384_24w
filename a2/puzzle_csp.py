#Look for #IMPLEMENT tags in this file.
'''
All encodings need to return a CSP object, and a list of lists of Variable objects 
representing the board. The returned list of lists is used to access the 
solution. 

For example, after these three lines of code

    csp, var_array = caged_csp(board)
    solver = BT(csp)
    solver.bt_search(prop_FC, var_ord)

var_array[0][0].get_assigned_value() should be the correct value in the top left
cell of the FunPuzz puzzle.

The grid-only encodings do not need to encode the cage constraints.

1. binary_ne_grid (worth 10/100 marks)
    - An enconding of a FunPuzz grid (without cage constraints) built using only 
      binary not-equal constraints for both the row and column constraints.

2. nary_ad_grid (worth 10/100 marks)
    - An enconding of a FunPuzz grid (without cage constraints) built using only n-ary 
      all-different constraints for both the row and column constraints. 

3. caged_csp (worth 25/100 marks) 
    - An enconding built using your choice of (1) binary binary not-equal, or (2) 
      n-ary all-different constraints for the grid.
    - Together with FunPuzz cage constraints.

'''
from enum import Enum
from math import prod
from cspbase import *
import itertools

def binary_ne_grid(fpuzz_grid):
    ##IMPLEMENT
    """ CSC384 BEGIN """
    size = fpuzz_grid[0][0]

    dom = [i for i in range(1, size + 1)]
    var_l_l = [
        [
            Variable(f"R{row + 1}C{col + 1}", dom) for col in range(size)
        ] for row in range(size)
    ]
    flattened_var_l = [var for var_l in var_l_l for var in var_l]
    csp = CSP("binary_ne", flattened_var_l)

    pairwise_ne = [p for p in itertools.product(dom, dom) if p[0] != p[1]] # the pairs of variables in the same row/col should not be the same

    for row in range(size):
        for col in range(size):
            curr_cell = var_l_l[row][col]

            for i in range(col + 1, size): # traverse row wise
                cons = Constraint(f"C({row + 1}{col + 1},{row + 1}{i + 1})", [curr_cell, var_l_l[row][i]])
                cons.add_satisfying_tuples(pairwise_ne)
                csp.add_constraint(cons)

            for j in range(row + 1, size): # traverse col wise
                cons = Constraint(f"C({row + 1}{col + 1},{j + 1}{col + 1})", [curr_cell, var_l_l[j][col]])
                cons.add_satisfying_tuples(pairwise_ne)
                csp.add_constraint(cons)

    return csp, var_l_l
    """ CSC384 END """


def nary_ad_grid(fpuzz_grid):
    ##IMPLEMENT
    """ CSC384 BEGIN """
    size = fpuzz_grid[0][0]
    dom = [i for i in range(1, size + 1)]

    var_l_l = [
        [
            Variable(f"R{row + 1}C{col + 1}", dom) for col in range(size)
        ] for row in range(size)
    ]
    flattened_var_l = [var for var_l in var_l_l for var in var_l]
    csp = CSP("nary_ad", flattened_var_l)

    permutation = list(itertools.permutations(dom))

    for row in range(size):
        cons = Constraint(f"R{row}", var_l_l[row])
        cons.add_satisfying_tuples(permutation)
        csp.add_constraint(cons)

    for col in range(size):
        cons = Constraint(f"C{col}", [var_l_l[row][col] for row in range(size)])
        cons.add_satisfying_tuples(permutation)
        csp.add_constraint(cons)

    return csp, var_l_l
    """ CSC384 END """

""" CSC384 BEGIN """
class Operation(Enum):
    ADD = 0
    SUB = 1
    DIV = 2
    MULT = 3
    NONE = 4

class Position:
    def __init__(
        self,
        row_col: int,
    ):
        self.row = row_col // 10
        self.col = row_col % 10

class Cage:
    def __init__(
        self,
        raw_cage: list[int],
    ):
        if len(raw_cage) >= 3: # == 3 possible?
            self.size = len(raw_cage) - 2
            self.target = raw_cage[self.size]
            self.op: Operation = Operation(raw_cage[self.size + 1])
            self.pos_l = [Position(x) for x in raw_cage[:self.size]]

        elif len(raw_cage) == 2:
            self.size = 1
            self.target = raw_cage[1]
            self.op: Operation = Operation.NONE
            self.pos_l = [Position(raw_cage[0])]

        else:
            raise ValueError("Incorrect cage format")

    def get_sat_tuple_l(
        self,
        dom: list[int],
    ) -> list[tuple[int]]:
        res_l: list[tuple[int]] = []

        if self.size == 1:
            return [[self.target]] # return a list of list; all values in `dom` valid

        for x in itertools.product(dom, repeat=self.size):
            if self.op == Operation.ADD:
                if sum(x) == self.target:
                    res_l.append(x)

            elif self.op == Operation.SUB:
                if x[0] - sum(x[1:]) == self.target:
                    res_l.extend(list(itertools.permutations(x)))

            elif self.op == Operation.DIV:
                if x[0] == self.target * prod(x[1:]):
                    res_l.extend(list(itertools.permutations(x)))

            elif self.op == Operation.MULT:
                if prod(x) == self.target:
                    res_l.append(x)

        #print("Duplicates: ", list(set([item for item in res_l if res_l.count(item) > 1])))
        return list(set(res_l))
""" CSC384 END """

def caged_csp(fpuzz_grid):
    ##IMPLEMENT
    """ CSC384 BEGIN """
    size = fpuzz_grid[0][0]
    dom = [i for i in range(1, size + 1)]

    #csp, var_l_l = nary_ad_grid(fpuzz_grid) # Ran 10 tests in 0.857s
    csp, var_l_l = binary_ne_grid(fpuzz_grid) # Ran 10 tests in 0.185s
    csp.name = "caged_csp"

    cage_l: list[Cage] = [Cage(raw_cage) for raw_cage in fpuzz_grid[1:]]

    for cage in cage_l:
        cons = Constraint(f"Cage({cage.op.name},{cage.target})", [var_l_l[pos.row - 1][pos.col - 1] for pos in cage.pos_l])
        cons.add_satisfying_tuples(cage.get_sat_tuple_l(dom))
        csp.add_constraint(cons)

    return csp, var_l_l
    """ CSC384 END """