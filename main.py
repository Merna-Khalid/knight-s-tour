import math

def not_valid_numbers(arr):
    for e in arr:
        if not(e.isnumeric()) or '-' in e or '.' in e:
            return True
    return False


# indices must be a list of integers
def outside_dim(indices, dims):
    for i, d in zip(indices, dims):
        if i < 1 or i > d:
            return True
    return False


def input_valid_dims(s):
    if len(s) != 2 or not_valid_numbers(s):
        return False, (0, 0)

    dims = [int(e) for e in s]
    if 0 in dims:
        return False, (0, 0)
    return True, dims


# s here is a list of strings
def input_valid_loc(s, dims):
    if len(s) != 2 or not_valid_numbers(s):
        return False, (0, 0)

    indices = [int(e) for e in s]
    if outside_dim(indices, dims):
        return False, (0, 0)

    return True, (indices[0], indices[1])


def get_knight_masks(dims, init_val):
    """
    Generates bitboard mask to get moves
    :param dims: dimensions of the board (column, row)
    :param init_val: maximum value for integer the size of dims multiplied
    :return: 4 masks for 4 directions
    """
    l1_mask = init_val
    l2_mask = init_val
    r1_mask = init_val
    r2_mask = init_val

    mask1 = 1 << (dims[0] - 1)
    if dims[0] > 1:
        mask2 = 3 << (dims[0] - 2)
    else:
        mask2 = 0
    mask3 = 1
    mask4 = 3

    for i in range(dims[1] + 1):
        l1_mask &= ~mask1
        l2_mask &= ~mask2
        r1_mask &= ~mask3
        r2_mask &= ~mask4
        mask1 <<= dims[0]
        mask2 <<= dims[0]
        mask3 <<= dims[0]
        mask4 <<= dims[0]

    l1_mask &= init_val
    l2_mask &= init_val
    r1_mask &= init_val
    r2_mask &= init_val

    return [l1_mask, l2_mask, r1_mask, r2_mask]


def get_indices_bitboard(bitboard, dims):
    """
    From bitboard representation returns occupied indices
    :param bitboard: an integer
    :param dims: dimensions of the board (column, row)
    :return: List of indices (row, column)
    """
    indices = []
    for i in range((dims[1] * dims[0])):
        di = 1 << i
        if di & bitboard:
            # row, column
            indices.append((dims[1] - (i // dims[0]) - 1, (dims[0]-1) - (i % dims[0])))
    return indices


def get_knight_moves(knights, dims, masks=None):  # position of knights
    """
    This function can be used
    :param knights: integer (bitboard) of knights
    :param dims: dimensions of the board (column, row)
    :param masks: a list
    :return: list of indices of moves
    """

    if masks is None:
        max_uint = 0
        for i in range(dims[1] * dims[0]):
            max_uint = (max_uint << 1) + 1
        l1_mask, l2_mask, r1_mask, r2_mask = get_knight_masks(dims, max_uint)
    else:
        l1_mask, l2_mask, r1_mask, r2_mask = masks

    l1 = (knights >> 1) & l1_mask
    l2 = (knights >> 2) & l2_mask
    r1 = (knights << 1) & r1_mask
    r2 = (knights << 2) & r2_mask
    h1 = l1 | r1
    h2 = l2 | r2

    final = (h1 << (dims[0] * 2)) | (h1 >> (dims[0] * 2)) | (h2 << dims[0]) | (h2 >> dims[0])

    return get_indices_bitboard(final, dims)


def get_knight_moves_test(knight, dims):
    """
    For testing purpose (only one knight)
    :param knight: the position of knight after flattening the matrix
    :param dims: dimensions of the board (column, row)
    :return: list of indices of moves
    """
    pos = 1 << ((dims[0] * dims[1]) - knight - 1)
    max_uint = 0
    for i in range(dims[1] * dims[0]):
        max_uint = (max_uint << 1) + 1

    if dims[0] > 1:
        nonoea = pos << (dims[0]*2 + 1)
        sosowe = pos >> (dims[0]*2 + 1)
        sosoea = pos >> (dims[0] * 2 - 1)
        nonowe = pos << (dims[0] * 2 - 1)
    else:
        nonoea, sosowe, sosoea, nonowe = 0, 0, 0, 0

    noeaea = pos << (dims[0] + 2)
    sowewe = pos >> (dims[0] + 2)
    soeaea = pos >> (dims[0] - 2)
    nowewe = pos << (dims[0] - 2)

    final = nonoea | sosowe | noeaea | sowewe | soeaea | nowewe | sosoea | nonowe

    print("Bitboard of knight: ", pos)
    print("Bitboard of moves: ", final)

    return get_indices_bitboard(final, dims)


def display_map(matrix, dims, msg=None):
    """
    A function that displays the map
    :param dims: two ints first is columns second is rows
    :param matrix: A list of lists of chars
    :return: None
    """
    if msg is not None:
        print(msg)
    cell_size = len(str(dims[0] * dims[1]))
    dims_1_digits = len(str(dims[1]))
    print(" " + "-" * (dims[0] * (cell_size + 1) + 3))
    for row in range(dims[1], 0, -1):
        row_print = str(row)
        if len(row_print) != dims_1_digits:
            row_print = " " * (dims_1_digits - len(row_print)) + row_print
        print(f"{row_print}|", " ".join(matrix[dims[1] - row]), "|")
    print(" " + "-" * (dims[0] * (cell_size + 1) + 3))
    print("   " + " ".join([" " * (cell_size - 1) + str(i) for i in range(1, dims[0] + 1)]))


def input_verification(msg, f, args_f=None):
    """
    verifies the input based on another function
    :param msg: str to be displayed while asking for input
    :param f: The function of verification
    :param args_f: extra argument passed to f
    :return: the output of the function
    """
    input_s = input(msg).split(" ")
    if args_f is not None:
        valid, output_ = f(input_s, args_f)
    else:
        valid, output_ = f(input_s)
    while not valid:
        print("Invalid dimensions!")
        input_s = input(msg).split(" ")
        if args_f is not None:
            valid, output_ = f(input_s, args_f)
        else:
            valid, output_ = f(input_s)
    return output_


def get_next_move_count(knight, dims):
    return len(get_knight_moves(knight, dims))


def mat_rep(inp_, dims):
    """
    changes input representation to matrix representation
    :param inp_: pair of input
    :param dims: two ints first is columns second is rows
    :return: pair of ints
    """
    return 0, 0


def to_index(x, y, dims):
    return dims[1] - y, x - 1


class Board:

    cell_size = 0
    matrix = []
    dims = [0, 0]
    occupied = []
    knight = (0, 0)
    masks = []
    x_center = 0
    y_center = 0

    def __init__(self, dims):
        self.dims = dims
        self.cell_size = len(str(dims[0] * dims[1]))
        self.matrix = [['_' * self.cell_size for i in range(dims[0])] for j in range(dims[1])]
        self.x_center = math.ceil(self.dims[1] / 2.0) - 1
        self.y_center = math.ceil(self.dims[0] / 2.0) - 1

        max_uint = 0
        for i in range(dims[1] * dims[0]):
            max_uint = (max_uint << 1) + 1
        self.masks = get_knight_masks(dims, max_uint)

    def get_next_move_count(self, knight):
        return len(self.get_moves(knight))

    def update(self, knights=[], moves=[]):
        self.matrix = [['_' * self.cell_size for i in range(self.dims[0])] for j in range(self.dims[1])]

        for mx, my in knights:
            self.matrix[mx][my] = ' ' * (self.cell_size - 1) + 'X'

        for mx, my in moves:
            count_moves = self.get_next_move_count((mx, my)) - 1
            self.matrix[mx][my] = ' ' * (self.cell_size - 1) + str(count_moves)

        for mx, my in self.occupied:
            self.matrix[mx][my] = ' ' * (self.cell_size - 1) + '*'

    def get_moves(self, knight):
        x, y = knight
        knight = 1 << ((self.dims[0] * self.dims[1]) - (x * self.dims[0] + y) - 1)
        moves = get_knight_moves(knight, self.dims, self.masks)
        valid_moves = []
        for m in moves:
            if m not in self.occupied:
                valid_moves.append(m)
        return valid_moves

    def input_move(self, moves):
        (x, y) = input_verification("Enter your next move: ", input_valid_loc, self.dims)
        mx, my = to_index(x, y, self.dims)

        msg = "Invalid move! Enter your next move: "
        while (mx, my) not in moves:
            (x, y) = input_verification(msg, input_valid_loc, self.dims)
            mx, my = to_index(x, y, self.dims)

        return mx, my

    def make_move(self, knight, moves):
        if not moves:
            if len(self.occupied) == (self.dims[0] * self.dims[1] - 1):
                print("What a great tour! Congratulations!")
            else:
                print("No more possible moves!")
                print("Your knight visited {} squares!".format(len(self.occupied) + 1))
            return False
        else:
            x, y = self.input_move(moves)

            self.occupied.append(knight)

            moves = self.get_moves((x, y))

            self.update([(x, y)], moves)
            self.knight = (x, y)
            return True

    def distance(self, x, y):
        return abs(x - self.x_center) + abs(y - self.y_center)

    def solve(self, knight):
        self.occupied = [knight]
        solution = [knight]
        while len(self.occupied) < self.dims[1]*self.dims[0]:
            moves = self.get_moves(knight)
            if not moves:
                break

            min_count = self.dims[1] * self.dims[0]
            min_ind = 0
            counted_moves = []
            for i, m in zip(range(len(moves)), moves):
                mx, my = m
                count_moves = self.get_next_move_count((mx, my)) - 1
                counted_moves.append(count_moves)
                if count_moves < min_count:
                    min_count = count_moves
                    min_ind = i

            max_distance = 0
            max_ind = 0
            for i, m in zip(range(len(moves)), moves):
                mx, my = m
                if counted_moves[i] == min_count:
                    distance = self.distance(mx, my)
                    if distance > max_distance:
                        distance = max_distance
                        max_ind = i

            knight = moves[max_ind]
            solution.append(knight)
            self.occupied.append(knight)

        # print(len(self.occupied))
        # print(self.dims[1]*self.dims[0])
        if len(self.occupied) != self.dims[1]*self.dims[0]:
            print("No solution exists!")
            return []
        else:
            return solution

    def display_solution(self, solution):
        for i, m in zip(range(1, len(solution)+1), solution):
            mx, my = m
            self.matrix[mx][my] = ' ' * (self.cell_size - len(str(i))) + str(i)

        display_map(self.matrix, self.dims, "Here's the solution!")


def launch_board():
    """
    Initiates The boards
    Takes input of dimension and knight starting position
    prints board
    """

    # # #
    dims = input_verification("Enter your board dimensions: ", input_valid_dims)
    (x, y) = input_verification("Enter the knight's starting position: ", input_valid_loc, dims)

    board = Board(dims)
    mx, my = to_index(x, y, dims)
    board.knight = (mx, my)

    answer = input("Do you want to try the puzzle? (y/n): ")
    while answer not in ["y", "n"]:
        print("Invalid input!")
        answer = input("Do you want to try the puzzle? (y/n): ")

    solution = board.solve(board.knight)
    if solution:
        if answer == "n":
            board.display_solution(solution)
        else:
            board = Board(dims)
            board.knight = (mx, my)
            moves = board.get_moves((mx, my))
            board.update([(mx, my)], moves)

            display_map(board.matrix, board.dims)

            while board.make_move(board.knight, board.get_moves(board.knight)):
                display_map(board.matrix, board.dims)


launch_board()

