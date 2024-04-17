#   Look for #IMPLEMENT tags in this file. These tags indicate what has
#   to be implemented to complete the warehouse domain.

#   You may add only standard python imports
#   You may not remove any imports.
#   You may not import or otherwise source any of your own files

import os  # for time functions
import math  # for infinity
from search import *  # for search engines
from sokoban import sokoban_goal_state, SokobanState, Direction, PROBLEMS  # for Sokoban specific classes and problems

# SOKOBAN HEURISTICS
""" CSC384 BEGIN """
def manhatten_dist(
    box: tuple[int, int],
    storage: tuple[int, int],
) -> int:
    # finds the distance between a box and a store using manhattan distance
    return abs(box[0] - storage[0]) + abs(box[1] - storage[1])

def heur_zero(state) -> int:
    '''Zero Heuristic can be used to make A* search perform uniform cost search'''
    return 0

def heur_manhattan_distance(
    state: SokobanState,
) -> int:
    # IMPLEMENT
    '''admissible sokoban puzzle heuristic: manhattan distance'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    # We want an admissible heuristic, which is an optimistic heuristic.
    # It must never overestimate the cost to get from the current state to the goal.
    # The sum of the Manhattan distances between each box that has yet to be stored and the storage point nearest to it is such a heuristic.
    # When calculating distances, assume there are no obstacles on the grid.
    # You should implement this heuristic function exactly, even if it is tempting to improve it.
    # Your function should return a numeric value; this is the estimate of the distance to the goal.
    """ CSC384 BEGIN """
    res = 0
    for box in state.boxes:
        shortest_dist = math.inf
        if box in state.storage:
            continue # no need to move
        else:
            for storage in state.storage:
                dist = manhatten_dist(box, storage)
                if dist < shortest_dist:
                    shortest_dist = dist
            res += shortest_dist
    return res
    """ CSC384 END """

def fval_function(
    sN: sNode,
    weight: float,
) -> float:
    # IMPLEMENT
    """
    Provide a custom formula for f-value computation for Anytime Weighted A star.
    Returns the fval of the state contained in the sNode.

    @param sNode sN: A search node (containing a SokobanState)
    @param float weight: Weight given by Anytime Weighted A star
    @rtype: float
    """
    """ CSC384 BEGIN """
    return sN.gval + weight * sN.hval
    """ CSC384 END """

## get_box_to_storage_cost
def get_boundary_of_box_and_storage(
    storage: tuple[int, int],
    box: tuple[int, int],
) -> tuple[int, int, int, int]:
    return (min(box[0], storage[0]), max(box[0], storage[0]), min(box[1], storage[1]), max(box[1], storage[1]))

def is_item_in_boundary(
    item: tuple[int, int],
    boundary: tuple[int, int, int, int],
) -> bool:
    return boundary[0] < item[0] < boundary[1] and boundary[2] < item[1] < boundary[3]

def count_item_in_boundary(
    item_s: frozenset[tuple],
    boundary: tuple[int, int, int, int],
) -> int:
    count = 0
    for item in item_s:
        if is_item_in_boundary(item, boundary):
            count += 1
    return count

def get_box_to_storage_cost(
    box: tuple[int, int],
    storage: tuple[int, int],
    state: SokobanState,
) -> int:
    boundary = get_boundary_of_box_and_storage(storage, box)
    num_box = count_item_in_boundary(state.boxes, boundary)
    num_robot = count_item_in_boundary(state.robots, boundary)
    num_obstacle = count_item_in_boundary(state.obstacles, boundary)
    return num_box + num_robot + 3 * num_obstacle

## is_box_dead_case
def is_box_on_corner(
    box: tuple[int, int],
    width: int,
    height: int,
) -> bool:
    return (box[0] == 0 or box[0] == width - 1) and (box[1] == 0 or box[1] == height - 1)

def is_box_on_boarder_ajacent_to_box_or_obstacle(
    box: tuple,
    width: int,
    height: int,
    box_s: frozenset[tuple],
    obstacle_s: frozenset[tuple],
) -> bool:
    if box[0] == 0 or box[0] == width - 1:
        return (box[0], box[1] + 1) in box_s or (box[0], box[1] - 1) in box_s or\
            (box[0], box[1] + 1) in obstacle_s or (box[0], box[1] - 1) in obstacle_s

    elif box[1] == 0 or box[1] == height - 1:
        return (box[0] + 1, box[1]) in box_s or (box[0] - 1, box[1]) in box_s or\
            (box[0] + 1, box[1]) in obstacle_s or (box[0] - 1, box[1]) in obstacle_s

    else:
        return False

#def is_box_on_boarder_but_no_storage_on_boarder(
#    box: tuple[int, int],
#    width: int,
#    height: int,
#    storage_s: frozenset[tuple[int, int]],
#):
#    if box[0] == 0 or box[0] == width - 1:
#        return not any(x == box[0] for x, _ in storage_s)
#    elif box[1] == 0 or box[1] == height - 1:
#        return not any(y == box[1] for _, y in storage_s)

def is_box_blocked_by_obstacle_in_both_way(
    box: tuple[int, int],
    obstacle_s: frozenset[tuple[int, int]],
) -> bool:
    return ((box[0] + 1, box[1]) in obstacle_s or (box[0] - 1, box[1]) in obstacle_s) and ((box[0], box[1] + 1) in obstacle_s or (box[0], box[1] - 1) in obstacle_s)

def is_box_dead_case(
    box: tuple,
    width: int,
    height: int,
    box_s: frozenset[tuple],
    obstacle_s: frozenset[tuple],
) -> bool:
    if is_box_on_corner(box, width, height):
        return True

    if is_box_on_boarder_ajacent_to_box_or_obstacle(box, width, height, box_s, obstacle_s):
        return True
    
    #if is_box_on_boarder_but_no_storage_on_boarder(box, width, height, box_s):
    #    return True

    if is_box_blocked_by_obstacle_in_both_way(box, obstacle_s):
        return True

def heur_alternate(
    state: SokobanState,
) -> int:
    # IMPLEMENT
    '''a better heuristic'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    # heur_manhattan_distance has flaws.
    # Write a heuristic function that improves upon heur_manhattan_distance to estimate distance between the current state and the goal.
    # Your function should return a numeric value for the estimate of the distance to the goal.
    # EXPLAIN YOUR HEURISTIC IN THE COMMENTS. Please leave this function (and your explanation) at the top of your solution file, to facilitate marking.

    """ CSC384 BEGIN """
    # 18/22, [5, 9, 17, 18] unsolved, 19 is around 2 seconds, sometimes failed, sometimes passed
    """
    Explanation:
        1. First, find the boxes that need to be pushed to the storage and the available storage.
        2. Then, for each box
            2.1 Check if the box is in a dead case, if so, return infinity.
            2.2 Find the nearest storage and calculate the cost of pushing the box to the storage.
            2.3 Remove the storage from the available storage list.
        3. Return the total cost.
    """
    
    cost = 0

    need_push_box_l = []
    available_storage_l = list(state.storage)

    for box in state.boxes:
        if box in state.storage:
            available_storage_l.remove(box)
        else:
            need_push_box_l.append(box)

    for box in need_push_box_l:
        if is_box_dead_case(box, state.width, state.height, state.boxes, state.obstacles):
            return math.inf

        cost_l = [manhatten_dist(box, storage) + get_box_to_storage_cost(box, storage, state) for storage in available_storage_l]
        cost += min(cost_l)
        available_storage_l.pop(cost_l.index(min(cost_l)))

    return cost
    """ CSC384 END """

# SEARCH ALGORITHMS
def weighted_astar(initial_state, heur_fn, weight, timebound):
    # IMPLEMENT
    '''Provides an implementation of weighted a-star, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False as well as a SearchStats object'''
    '''implementation of weighted astar algorithm'''
    """ CSC384 BEGIN """
    # Mark: 17/22
    end_time = os.times()[0] + timebound - 0.02

    def wrapped_faval_function(sN): return fval_function(sN, weight)
    search_engine = SearchEngine('custom', 'full')
    search_engine.init_search(initial_state, sokoban_goal_state, heur_fn, wrapped_faval_function)

    time_left = end_time - os.times()[0]
    return search_engine.search(time_left)
    """ CSC384 END """

def iterative_astar(initial_state, heur_fn, weight=1, timebound=5):  # uses f(n), see how autograder initializes a search line 88
    # IMPLEMENT
    '''Provides an implementation of realtime a-star, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False as well as a SearchStats object'''
    '''implementation of iterative astar algorithm'''
    """ CSC384 BEGIN """
    # result: 14/22 solved, 14/14 matched or outperformed the benchmark
    weight = 10
    end_time = os.times()[0] + timebound - 0.02
    time_left = end_time - os.times()[0]
    
    se = SearchEngine('custom', 'full')
    def wrapped_faval_function(sN): return fval_function(sN, weight)
    se.init_search(initial_state, sokoban_goal_state, heur_fn, wrapped_faval_function)

    prune_fval = math.inf

    first_search_res = se.search(time_left)
    best_res = first_search_res

    if first_search_res[0]:
        prune_fval = first_search_res[0].gval
    else:
        return first_search_res # first search solution not found

    time_left = end_time - os.times()[0]

    while time_left > 0:
        weight *= 0.7
        se.init_search(initial_state, sokoban_goal_state, heur_fn, wrapped_faval_function) # new `weight` in the following search
        result = se.search(time_left, (math.inf, math.inf, prune_fval))
        time_left = end_time - os.times()[0]

        if result[0]:
            prune_fval = result[0].gval
            best_res = result

        if se.open.empty():
            return best_res

    return best_res
    """ CSC384 END """

def iterative_gbfs(initial_state, heur_fn, timebound=5):  # only use h(n)
    # IMPLEMENT
    '''Provides an implementation of anytime greedy best-first search, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    '''implementation of iterative gbfs algorithm'''
    """ CSC384 BEGIN """
    # result: 17/22 solved, 16/17 matched or outperformed the benchmark
    end_time = os.times()[0] + timebound - 0.02
    time_left = end_time - os.times()[0]

    se = SearchEngine('best_first', 'full')
    se.init_search(initial_state, sokoban_goal_state, heur_fn)

    prune_gval = math.inf

    first_search_res = se.search(time_left)
    best_res = first_search_res

    if first_search_res[0]:
        prune_gval = first_search_res[0].gval
    else:
        return first_search_res # first search solution not found

    time_left = end_time - os.times()[0]

    while time_left > 0:
        result = se.search(time_left, (prune_gval, math.inf, math.inf))
        time_left = end_time - os.times()[0]

        if result[0]:
            prune_gval = result[0].gval
            best_res = result

        if se.open.empty():
            return best_res

    return best_res
    """ CSC384 END """
