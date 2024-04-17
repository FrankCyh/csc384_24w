"""
An AI player for Othello. 
"""

from cmath import e
import random
import sys
import time

# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move

def eprint(*args, **kwargs): #you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)

""" CSC384 BEGIN """
LOG_ON = False
#LOG_ON = True
board_t = list[list[int]]
move_t = tuple[int]

min_cache = {}
max_cache = {}

alpha_cache = {}
beta_cache = {}

## Helper functions
def get_opponent_color(
    color: int,
) -> int:
    return 1 if color == 2 else 2

def get_color_name(
    color: int,
) -> str:
    return "Black" if color == 1 else "White"

def print_board(
    board: board_t,
) -> None:
    size = len(board)
    for row in range(size):
        for col in range(size):
            if board[row][col] == 1:
                eprint("⚫️", end=" ")
            elif board[row][col] == 2:
                eprint("⚪️", end=" ")
            else:
                eprint("◽️", end=" ")
        eprint()
    eprint()

def print_move_on_board(
    board: board_t,
    move: move_t,
    color: int,
    limit: int,
) -> None:
    size = len(board)

    new_board = play_move(board, color, move[0], move[1])

    flipped_pos = [
        [1 if new_board[row][col] != board[row][col] else 0 for col in range(size)] for row in range(size)
    ]

    eprint(f" {limit} - {get_color_name(color)} ({move[1]}, {move[0]}) util: {compute_utility(new_board, 1)} ".center(30 + limit * 10, "-")) # always compute the utility for black color
    for row in range(size):
        for col in range(size):
            if (col, row) == move: #* use (col, row) instead of (row, col)
                eprint("⚫️" if color == 1 else "⚪️", end=".")
            elif flipped_pos[row][col]:
                eprint("🔘", end=" ")
            elif new_board[row][col] == 1:
                eprint("⚫️", end=" ")
            elif new_board[row][col] == 2:
                eprint("⚪️", end=" ")
            else:
                eprint("◽️", end=" ")
        eprint()
    eprint()

""" CSC384 END """


# Method to compute utility value of terminal state
def compute_utility(
    board: board_t,
    color: int,
) -> int:
    #IMPLEMENT
    #return compute_heuristic(board, color)

    """ CSC384 BEGIN """
    assert color in (1, 2)

    score = get_score(board)

    if color == 1:
        return (score[0] - score[1])
    else:
        return (score[1] - score[0])
    """ CSC384 END """

""" CSC384 BEGIN """
# Heuristic from the following paper: https://courses.cs.washington.edu/courses/cse573/04au/Project/mini1/RUSSIA/Final_Paper.pdf

def heu_coin_parity(
    board: board_t,
    color: int,
) -> int:
    score = get_score(board)
    if color == 1:
        return 100 * (score[0] - score[1]) / (score[0] + score[1])
    else:
        return 100 * (score[1] - score[0]) / (score[0] + score[1])

def heu_mobility(
    board: board_t,
    color: int,
) -> int:
    self_poss_move = len(get_possible_moves(board, color))
    oppo_poss_move = len(get_possible_moves(board, get_opponent_color(color)))
    if self_poss_move + oppo_poss_move == 0:
        return 0
    return 100 * (self_poss_move - oppo_poss_move) / (self_poss_move + oppo_poss_move)

def heu_corner_captured(
    board: board_t,
    color: int,
) -> int:
    board_size = len(board)

    def count_corner(
        board: board_t,
        color: int,
    ) -> int:
        return sum([1 if board[i][j] == color else 0 for i in [0, board_size - 1] for j in [0, board_size - 1]])

    num_corner_color1 = count_corner(board, 1)
    num_corner_color2 = count_corner(board, 2)

    if num_corner_color1 + num_corner_color2 == 0:
        return 0

    if color == 1:
        return 100 * (num_corner_color1 - num_corner_color2) / (num_corner_color1 + num_corner_color2)
    else:
        return 100 * (num_corner_color2 - num_corner_color1) / (num_corner_color1 + num_corner_color2)


def heu_static_weight(
    board: board_t,
    color: int,
):
    board_size = len(board)
    weight_matrix_4_x_4 = [
        [4, -3, -3, 4],
        [-3, -4, -4, -3],
        [-3, -4, -4, -3],
        [4, -3, -3, 4],
    ]
    weight_matrix_5_x_5 = [
        [4, -3, 2, -3, 4],
        [-3, -4, -1, -4, -3],
        [2, -1, 1, -1, 2],
        [-3, -4, -1, -4, -3],
        [4, -3, 2, -3, 4],
    ]
    weight_matrix_6_x_6 = [
        [4, -3, 2, 2, -3, 4],
        [-3, -4, -1, -1, -4, -3],
        [2, -1, 1, 1, -1, 2],
        [2, -1, 1, 1, -1, 2],
        [-3, -4, -1, -1, -4, -3],
        [4, -3, 2, 2, -3, 4],
    ]

    def get_static_weight_matrix():
        if board_size <= 3:
            return 0
        elif board_size == 4:
            return weight_matrix_4_x_4
        elif board_size == 5:
            return weight_matrix_5_x_5
        elif board_size == 6:
            return weight_matrix_6_x_6
        else:
            for i in [0, 6 - 1]:
                weight_matrix_6_x_6[i][2:4] = [2 for _ in range(board_size - 4)]
            for i in [1, 6 - 2]:
                weight_matrix_6_x_6[i][2:4] = [-1 for _ in range(board_size - 4)]
            weight_matrix_6_x_6[2:4] = [
                [2, -1] + [1 for _ in range(board_size - 4)] + [-1, 2] for _ in range(board_size - 4)
            ]
            return weight_matrix_6_x_6

    weight_matrix = get_static_weight_matrix()

    res = 0
    for row in range(board_size):
        for col in range(board_size):
            if board[row][col] == color:
                res += weight_matrix[row][col]
            else:
                res -= weight_matrix[row][col]

    return res
""" CSC384 END """


# Better heuristic value of board
def compute_heuristic(
    board: board_t,
    color: int,
) -> int:
    #IMPLEMENT
    """ CSC384 BEGIN """
    return 25 * heu_coin_parity(board, color) + 5 * heu_mobility(board, color) + 10 * heu_corner_captured(board, color) + 30 * heu_static_weight(board, color)
    """ CSC384 END """

############ MINIMAX ###############################
def minimax_min_node(
    board: board_t,
    color: int,
    limit: int,
    caching: int = 0,
) -> tuple[move_t, int]:
    #IMPLEMENT (and replace the line below)
    """ CSC384 BEGIN """
    assert color in (1, 2)
    assert caching in (0, 1)

    if caching and board in min_cache:
        if LOG_ON:
            eprint(f" Cache hit for {limit} - {get_color_name(get_opponent_color(color))}: ".center(30 + limit * 10, "-"))
            print_board(board)
        return min_cache[board]

    if limit == 0:
        return (None, compute_utility(board, color))

    poss_move_l = get_possible_moves(board, get_opponent_color(color))
    if len(poss_move_l) == 0:
        return (None, compute_utility(board, color))

    best_move = None
    min_util = float('inf')
    for move in poss_move_l:
        if LOG_ON:
            print_move_on_board(board, move, get_opponent_color(color), limit)
        board_after_move = play_move(board, get_opponent_color(color), move[0], move[1])
        curr_util = minimax_max_node(board_after_move, color, limit - 1, caching)[1]
        if curr_util < min_util:
            if LOG_ON:
                eprint(f"Updating level {limit} min_util from {min_util} to {curr_util}")
            min_util = curr_util
            best_move = move

    if caching:
        min_cache[board] = (best_move, min_util)

    return (best_move, min_util)
    """ CSC384 END """

def minimax_max_node(
    board: board_t,
    color: int,
    limit: int,
    caching: int = 0,
) -> tuple[move_t, int]: #returns highest possible utility
    #IMPLEMENT (and replace the line below)
    """ CSC384 BEGIN """
    assert color in (1, 2)
    assert caching in (0, 1)

    if caching and board in max_cache:
        if LOG_ON:
            eprint(f" Cache hit for {limit} - {get_color_name(color)}: ".center(30 + limit * 10, "-"))
            print_board(board)
        return max_cache[board]

    if limit == 0:
        return (None, compute_utility(board, color))

    poss_move_l = get_possible_moves(board, color)
    if len(poss_move_l) == 0:
        return (None, compute_utility(board, color))

    best_move = None
    max_util = float('-inf')
    for move in poss_move_l:
        if LOG_ON:
            print_move_on_board(board, move, color, limit)
        board_after_move = play_move(board, color, move[0], move[1])
        curr_util = minimax_min_node(board_after_move, color, limit - 1, caching)[1]
        if curr_util > max_util:
            if LOG_ON:
                eprint(f"Updating level {limit} max_util from {max_util} to {curr_util}")
            max_util = curr_util
            best_move = move

    if caching:
        max_cache[board] = (best_move, max_util)

    return (best_move, max_util)
    """ CSC384 END """

def select_move_minimax(
    board: board_t,
    color: int,
    limit: int,
    caching: int = 0,
) -> move_t:
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    """
    #IMPLEMENT (and replace the line below)
    """ CSC384 BEGIN """
    global min_cache, max_cache
    min_cache.clear()
    max_cache.clear()
    return minimax_max_node(board, color, limit, caching)[0]
    """ CSC384 END """

############ ALPHA-BETA PRUNING #####################
""" CSC384 BEGIN """
def get_sorted_move_board_d(
    board: board_t,
    color: int,
    ordering: int,
):
    move_l = get_possible_moves(board, color)
    move_board_d = {move: play_move(board, color, move[0], move[1]) for move in move_l}

    if ordering:
        move_board_d = dict(sorted(move_board_d.items(), key=lambda item: compute_utility(item[1], color), reverse=True))

    return move_board_d
""" CSC384 END """

def alphabeta_min_node(
    board: board_t,
    color: int,
    alpha: int,
    beta: int,
    limit: int,
    caching: int = 0,
    ordering: int = 0,
) -> tuple[move_t, int]:
    #IMPLEMENT (and replace the line below)
    """ CSC384 BEGIN """
    assert color in (1, 2)
    assert caching in (0, 1)

    if caching and board in beta_cache:
        if LOG_ON:
            eprint(f" Cache hit for {limit} - {get_color_name(get_opponent_color(color))}: ".center(30 + limit * 10, "-"))
            print_board(board)
        return beta_cache[board]

    if limit == 0:
        return (None, compute_utility(board, color))

    sorted_move_board_d = get_sorted_move_board_d(board, get_opponent_color(color), ordering)  #* opponent's utility in descending order is the same as current color's utility in ascending order
    if len(sorted_move_board_d) == 0:
        return (None, compute_utility(board, color))

    best_move = None
    min_util = float('inf')
    for move, board_after_move in sorted_move_board_d.items():
        if LOG_ON:
            print_move_on_board(board, move, get_opponent_color(color), limit)
        curr_util = alphabeta_max_node(board_after_move, color, alpha, beta, limit - 1, caching, ordering)[1]

        if curr_util < min_util:
            min_util = curr_util
            best_move = move

        if beta > min_util:
            if LOG_ON:
                eprint(f"Updating level {limit} beta from {beta} to {min_util}")
            beta = min_util #* `min_util` is for the current children and `beta` is for the whole tree, so we need to separate them
            if beta <= alpha:
                if LOG_ON:
                    eprint(f"Pruning min node with alpha={alpha}, beta={beta}")
                break

    if caching:
        beta_cache[board] = (best_move, min_util)

    return best_move, min_util
    """ CSC384 END """

def alphabeta_max_node(
    board: board_t,
    color: int,
    alpha: int,
    beta: int,
    limit: int,
    caching: int = 0,
    ordering: int = 0,
) -> tuple[move_t, int]:
    #IMPLEMENT (and replace the line below)
    """ CSC384 BEGIN """
    assert color in (1, 2)
    assert caching in (0, 1)

    if caching and board in alpha_cache:
        if LOG_ON:
            eprint(f" Cache hit for {limit} - {get_color_name(color)}: ".center(30 + limit * 10, "-"))
            print_board(board)
        return alpha_cache[board]

    if limit == 0:
        return (None, compute_utility(board, color))

    sorted_move_board_d = get_sorted_move_board_d(board, color, ordering)
    if len(sorted_move_board_d) == 0:
        return (None, compute_utility(board, color))

    best_move = None
    max_util = float('-inf')
    for move, board_after_move in sorted_move_board_d.items():
        if LOG_ON:
            print_move_on_board(board, move, color, limit)
        curr_util = alphabeta_min_node(board_after_move, color, alpha, beta, limit - 1, caching, ordering)[1]

        if curr_util > max_util:
            max_util = curr_util
            best_move = move

        if alpha < max_util:
            if LOG_ON:
                eprint(f"Updating level {limit} alpha from {alpha} to {max_util}")
            alpha = max_util
            if beta <= alpha:
                if LOG_ON:
                    eprint(f"Pruning max node with alpha={alpha}, beta={beta}")
                break

    if caching:
        alpha_cache[board] = (best_move, max_util)

    return best_move, max_util
    """ CSC384 END """

def select_move_alphabeta(
    board: board_t,
    color: int,
    limit: int,
    caching: int = 0,
    ordering: int = 0,
) -> move_t:
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    If ordering is ON (i.e. 1), use node ordering to expedite pruning and reduce the number of state evaluations. 
    If ordering is OFF (i.e. 0), do NOT use node ordering to expedite pruning and reduce the number of state evaluations. 
    """
    #IMPLEMENT (and replace the line below)
    """ CSC384 BEGIN """
    global alpha_cache, beta_cache
    alpha_cache.clear()
    beta_cache.clear()
    return alphabeta_max_node(board, color, float("-inf"), float("inf"), limit, caching, ordering)[0]
    """ CSC384 END """

####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Othello AI") # First line is the name of this AI
    arguments = input().split(",")

    color = int(arguments[0]) #Player color: 1 for dark (goes first), 2 for light.
    limit = int(arguments[1]) #Depth limit
    minimax = int(arguments[2]) #Minimax or alpha beta
    caching = int(arguments[3]) #Caching
    ordering = int(arguments[4]) #Node-ordering (for alpha-beta only)

    if (minimax == 1):
        eprint("Running MINIMAX")
    else:
        eprint("Running ALPHA-BETA")    

    if (caching == 1):
        eprint("State Caching is ON")
    else:
        eprint("State Caching is OFF")

    if (ordering == 1):
        eprint("Node Ordering is ON")
    else:
        eprint("Node Ordering is OFF")

    if (limit == -1):
        eprint("Depth Limit is OFF")
    else:
        eprint("Depth Limit is ", limit)

    if (minimax == 1 and ordering == 1):
        eprint("Node Ordering should have no impact on Minimax")

    """ CSC384 BEGIN """
    move_count = 0
    """ CSC384 END """

    while True: # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL": # Game is over.
            print
        else:
            board = eval(input()) # Read in the input and turn it into a Python
            # object. The format is a list of rows. The
            # squares in each row are represented by
            # 0 : empty square
            # 1 : dark disk (player 1)
            # 2 : light disk (player 2)

            # Select the move and send it to the manager
            if (minimax == 1): #run this if the minimax flag is given
                movei, movej = select_move_minimax(board, color, limit, caching)
            else: #else run alphabeta
                movei, movej = select_move_alphabeta(board, color, limit, caching, ordering)

            """ CSC384 BEGIN """
            print("{} {}".format(movei, movej))
            move_count += 1
            eprint(f" Move {move_count} ".center(60, "="))

if __name__ == "__main__":
    run_ai()
