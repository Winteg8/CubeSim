import numpy as np
from PIL import Image
import requests
from io import BytesIO
import copy

face_keys = ['U', 'R', 'F', 'D', 'L', 'B']
slice_keys = ['E', 'M', 'S']
axis_keys = ['Y', 'X', 'Z']
all_keys = face_keys + slice_keys + axis_keys
# a dict mapping moves to axes: {face or slice key: axis key}
move_to_axis = dict(zip(all_keys, axis_keys + axis_keys + axis_keys + axis_keys))
valid_moves = all_keys + [key.lower() for key in all_keys]


# Return True and list of moves if parsing is successful
# Returns False and all the moves that could be performed until the parsing error
def parse_moves(move_str: str):
    move_list = []
    last_letter = ' '
    move_str += ' '
    for letter in move_str:
        if last_letter in valid_moves:
            move_list.append([last_letter, 1])
            if letter == "'":
                move_list[-1][1] = 3
            elif letter.isdigit():
                move_list[-1][1] = int(letter)
            last_letter = letter
        elif last_letter == "'" or last_letter.isdigit() or last_letter == ' ':
            if letter == ' ' or letter in valid_moves or letter == "'":
                last_letter = letter
            else:
                return False, move_list
        else:
            return False, move_list
    return True, move_list


def get_adj_faces(move: str):
    return [face for face in face_keys if move_to_axis[face] != move_to_axis[move]]


class Cube:
    def __init__(self, scheme_name: str = None, verbose=False):
        self.slice_to_index = {
            'U': 0,
            'E': 1,
            'D': 2,
            'L': 0,
            'M': 1,
            'R': 2,
            'F': 0,
            'S': 1,
            'B': 2
        }

        #
        self.axis_to_slices = {
            'X': ['L', 'M', 'R'],
            'Y': ['U', 'E', 'D'],
            'Z': ['F', 'S', 'B']
        }

        # standard indexing over faces for visualization
        self.std_face_mapping = {
            'U': ('Z', 'X'),
            'R': ('Y', 'Z'),
            'F': ('Y', 'X'),
            'D': ('Z', 'X'),
            'L': ('Y', 'Z'),
            'B': ('Y', 'X'),
        }

        # maps axis of a move to the faces where the natural order of facelets has to be reversed
        self.move_facelet_order_change = {
            'Y': ('R', 'F'),
            'X': ('F', 'D'),
            'Z': ('D', 'L')
        }
        self.color_schemes = self.init_css()
        self.colors = self.set_colors(scheme_name)  # active color scheme
        self.pieces = self.init_pieces(verbose)
        # the format is YXZ instead of XYZ is because numpy arrays are row-first, and rows are specified by Y-axis
        self.slices = {    # Y, X, Z
            'U': self.pieces[0, :, :],
            'E': self.pieces[1, :, :],
            'D': self.pieces[2, :, :],
            'L': self.pieces[:, 0, :],
            'M': self.pieces[:, 1, :],
            'R': self.pieces[:, 2, :],
            'F': self.pieces[:, :, 0],
            'S': self.pieces[:, :, 1],
            'B': self.pieces[:, :, 2]
        }
        self.string_to_index = dict()       # used to access pieces by their letter strings (e.g. "UFR", "UF", "U")
        for y in range(3):
            for x in range(3):
                for z in range(3):
                    if not (y == 1 and x == 1 and z == 1):
                        self.string_to_index[frozenset(self.pieces[y, x, z].keys())] = [y, x, z]

    def get_state(self):
        state = copy.deepcopy(self.pieces)
        return state

    def set_state(self, state: np.ndarray):
        self.pieces = copy.deepcopy(state)

    # set cube colors according to a color scheme
    def set_colors(self, scheme_name):
        if scheme_name is None or scheme_name not in self.color_schemes:
            self.colors = self.color_schemes['default']
        else:
            self.colors = self.color_schemes[scheme_name]
        return self.colors

    # initialize Color SchemeS
    # misleading name is misleading
    def init_css(self):
        # format: U, R, F, D, L, B
        default_colors = ['w', 'r', 'g', 'y', 'o', 'b']
        color_schemes = dict()
        color_schemes['default'] = dict(zip(face_keys, default_colors))
        return color_schemes

    # adds a new color scheme or overwrites an existing one
    # returns True if successful
    def set_color_scheme(self, name: str, scheme: dict):
        if name != 'default':
            self.color_schemes[name] = scheme
            return True
        else:
            return False

    # generates all the pieces according to the active color scheme
    def init_pieces(self, verbose):
        pieces = np.empty((3, 3, 3), dtype=dict)

        # set color of each center according to the active color scheme
        if verbose: print("Centers:")
        for face in face_keys:
            face_axis = move_to_axis[face]
            remaining_axes = axis_keys.copy()
            remaining_axes.remove(face_axis)
            center_coords = {face_axis: self.slice_to_index[face], remaining_axes[0]: 1, remaining_axes[1]: 1}
            pieces[center_coords['Y'], center_coords['X'], center_coords['Z']] = {face: self.colors[face]}
            if verbose:
                print("CENTER:", face, "\t\tAXIS:", face_axis, "\t\tCOORDS:", center_coords)

        # generate edges - unordered pairs of adjacent faces
        m_faces = ['U', 'F', 'D', 'B']
        edges = []
        for m_face in m_faces:
            for side_face in ['R', 'L']:
                edges.append([m_face, side_face])
        for m_edge in [['U', 'F'], ['U', 'B'], ['D', 'F'], ['D', 'B']]:
            edges.append(m_edge)
        # print("EDGES:", edges)

        # set colors of each edge according to the active color scheme
        if verbose: print("\nEdges:")
        for edge in edges:
            edge_axes = [move_to_axis[edge[0]], move_to_axis[edge[1]]]
            # print("ALL AXES:", plane_keys, "\t\tEDGE AXES:", edge_axes)
            remaining_axis = (set(axis_keys) - set(edge_axes)).pop()
            # print("REMAINING AXIS:", remaining_axis)
            edge_coords = {edge_axes[0]: self.slice_to_index[edge[0]], edge_axes[1]: self.slice_to_index[edge[1]],
                           remaining_axis: 1}
            pieces[edge_coords['Y'], edge_coords['X'], edge_coords['Z']] = dict(
                zip(edge, [self.colors[face] for face in edge]))
            if verbose:
                print("EDGE PAIR:", edge, "\t\tAXES:", edge_axes, "\t\tCOORDS:", edge_coords)

        # generate corners - unordered triplets of adjacent faces
        corners = [(x, y, z) for x in ['U', 'D'] for y in ['F', 'B'] for z in ['R', 'L']]
        # print("CORNERS:", corners)

        # set colors of each corner according to the active color scheme
        if verbose: print("\nCorners:")
        for corner in corners:
            corner_axes = [move_to_axis[corner[0]], move_to_axis[corner[1]], move_to_axis[corner[2]]]
            corner_coords = {corner_axes[0]: self.slice_to_index[corner[0]],
                             corner_axes[1]: self.slice_to_index[corner[1]],
                             corner_axes[2]: self.slice_to_index[corner[2]]}
            pieces[corner_coords['Y'], corner_coords['X'], corner_coords['Z']] = dict(
                zip(corner, [self.colors[face] for face in corner]))
            if verbose: print("CORNER TRIPLE:", corner, "\t\tAXES:", corner_axes, "\t\tCOORDS:", corner_coords)

        if verbose:
            print("\nAll Pieces:")
            pid = 0
            for y in range(3):
                for x in range(3):
                    for z in range(3):
                        if not (y == 1 and x == 1 and z == 1):
                            print(pid, ": ", pieces[y, x, z])
                            pid += 1
        return pieces

    # returns the value of a piece as a dictionary of format {face: color}
    # piece_keys can be a string representing an intersection of faces (e.g. "RUF")
    # piece_keys can also be a dictionary mapping axes to indices of a piece within the 3x3x3 array self.pieces
    # piece_keys can also be a list of indices or of letters
    def get_piece(self, piece_keys, mask=None):
        pieces = self.pieces
        if type(mask) == np.ndarray:
            pieces = mask
        if type(piece_keys) == str or (type(piece_keys) == list and type(piece_keys[0]) == str):
            index = self.string_to_index[frozenset(piece_keys)]
            return pieces[index[0], index[1], index[2]]
        elif type(piece_keys) == dict:
            return pieces[piece_keys['Y'], piece_keys['X'], piece_keys['Z']]
        elif type(piece_keys) == list:
            return pieces[piece_keys[0], piece_keys[1], piece_keys[2]]
        else:
            return None

    # sets ALL facelet values of a specified piece
    def set_piece(self, piece_keys, piece: dict):
        if type(piece_keys) == str or (type(piece_keys) == list and type(piece_keys[0]) == str):
            index = self.string_to_index[frozenset(piece_keys)]
            self.pieces[index[0], index[1], index[2]] = piece
        elif type(piece_keys) == dict:
            self.pieces[piece_keys['Y'], piece_keys['X'], piece_keys['Z']] = piece
        elif type(piece_keys) == list:
            self.pieces[piece_keys[0], piece_keys[1], piece_keys[2]] = piece

    # sets only the specified facelet values of a specified piece
    def set_piece_facelets(self, piece_keys, piece: dict, mask=None):
        pieces = self.pieces
        if type(mask) == np.ndarray:
            pieces = mask
        if type(piece_keys) == str or (type(piece_keys) == list and type(piece_keys[0]) == str):
            index = self.string_to_index[frozenset(piece_keys)]
            for facelet in piece.keys():
                pieces[index[0], index[1], index[2]][facelet] = piece[facelet]
        elif type(piece_keys) == dict:
            for facelet in piece.keys():
                print(facelet)
                pieces[piece_keys['Y'], piece_keys['X'], piece_keys['Z']][facelet] = piece[facelet]
        elif type(piece_keys) == list:
            for facelet in piece.keys():
                pieces[piece_keys[0], piece_keys[1], piece_keys[2]][facelet] = piece[facelet]

    # set colors of piece(s) sticker-by-sticker
    # takes a dict of form: {facelet: color}
    def set_piece_colors(self, piece_colors: dict, mask=None):
        pieces = self.pieces
        if type(mask) == np.ndarray:
            pieces = mask
        index = self.string_to_index[frozenset(piece_colors.keys())]
        for facelet in piece_colors.keys():
            pieces[index[0], index[1], index[2]][facelet] = piece_colors[facelet]

    # TO DO: change method so piece_keys can be multiple types (list, string, dict, etc)
    # set color of an entire piece
    def set_piece_color(self, piece_keys: list, color_str, mask=None, verbose=False):
        pieces = self.pieces
        if type(mask) == np.ndarray:
            pieces = mask
        new_colors = dict.fromkeys(piece_keys, color_str)
        index = self.string_to_index[frozenset(piece_keys)]
        pieces[index[0], index[1], index[2]] = new_colors
        if verbose:
            print("changed", piece_keys, " to", new_colors)

    # UNUSED METHOD
    # returns a dict of piece groups for a given slice
    # keys: 'centers', 'edges', 'corners'
    # values: lists of pieces
    # piece lists are not sorted
    def get_slice_piece_groups(self, move: str):
        slice_pieces = self.slices[move]
        piece_groups = {'centers': [], 'edges': [], 'corners': []}
        outer_slice = (self.slice_to_index[move] != 1)
        for row in range(3):
            for col in range(3):
                # if len(slice_pieces[row, col].keys()) == 2:
                if outer_slice:
                    if row % 2 == 0 and col % 2 == 0:   # both even: corner piece of slice
                        piece_groups['corners'].append(slice_pieces[row, col])
                    elif (row % 2 + col % 2) == 1:          # exactly one even: edge piece of slice
                        piece_groups['edges'].append(slice_pieces[row, col])
                    elif row == 1 and col == 1:
                        piece_groups['centers'].append(slice_pieces[row, col])
                else:
                    if row % 2 == 0 and col % 2 == 0:  # both even: corner piece of slice
                        piece_groups['edges'].append(slice_pieces[row, col])
                    elif (row % 2 + col % 2) == 1:  # exactly one even: edge piece of slice
                        piece_groups['centers'].append(slice_pieces[row, col])
        return piece_groups

    # returns a dict of piece groups for a given slice
    # each piece group lists are sorted by adjacency of pieces:
    #   meaning consecutive pieces in a group are 'adjacent' to each other
    # piece groups keys: 'centers', 'edges', 'corners'
    # piece groups values: lists of pieces
    def get_sorted_piece_groups(self, move: str):
        slice_pieces = self.slices[move]
        piece_groups = {'centers': [], 'edges': [], 'corners': []}
        adj_faces = get_adj_faces(move)
        if self.slice_to_index[move] == 1:  # inner slice
            for i in range(len(adj_faces)):
                piece_letters = [adj_faces[i]]
                piece_groups['centers'].append(self.get_piece(piece_letters))
                piece_letters.append(adj_faces[(i + 1) % len(adj_faces)])
                piece_groups['edges'].append(self.get_piece(piece_letters))
        else:   # outer slice
            piece_groups['centers'].append(self.get_piece([move]))
            for i in range(len(adj_faces)):
                piece_letters = [move, adj_faces[i]]
                piece_groups['edges'].append(self.get_piece(piece_letters))
                piece_letters.append(adj_faces[(i + 1) % len(adj_faces)])
                piece_groups['corners'].append(self.get_piece(piece_letters))
        return piece_groups

    # TO DO: move the code block into move() where it is called and make more efficient
    # returns a dict of piece groups for a given slice
    # each piece group lists are sorted by adjacency of pieces:
    #   meaning consecutive pieces in a group are 'adjacent' to each other
    # piece groups keys: 'edges', 'corners'
    # piece groups values: lists of pieces
    # these groups don't necessarily contain those types of pieces though
    # e.g. M slice: 'corners': UF, DF, DB, UB,  'edges': U, F, D, B
    def get_slice_groups(self, move: str):
        slice_pieces = self.slices[move]
        piece_groups = {'edges': [], 'corners': []}
        adj_faces = get_adj_faces(move)
        base_letters = []
        if self.slice_to_index[move] != 1:  # outer slice
            base_letters.append(move)
        for i in range(len(adj_faces)):
            piece_letters = copy.deepcopy(base_letters)
            piece_letters.append(adj_faces[i])
            piece_groups['edges'].append(self.get_piece(piece_letters))
            piece_letters.append(adj_faces[(i + 1) % len(adj_faces)])
            piece_groups['corners'].append(self.get_piece(piece_letters))
        return piece_groups

    # move performed piecewise
    def move(self, move: str, magnitude: int):
        moved_slices = []

        # find indices of slices that will be affected
        if move.upper() in axis_keys:       # rotations
            moved_slices = [0, 1, 2]
        else:
            if move.upper() in slice_keys:  # slice moves
                moved_slices = [1]
            else:                           # regular and wide moves
                moved_slices.append(self.slice_to_index[move.upper()])
                if move.islower():          # wide moves
                    moved_slices.append(1)
        move = move.upper()
        adj_faces = get_adj_faces(move)
        num_faces = len(adj_faces)      # should always be 4, but gotta generalize, lol
        axis_of_rot = move_to_axis[move]
        move_dir = num_faces - magnitude
        if move in ['R', 'D', 'B', 'E', 'X']:
           move_dir = magnitude
        for moved_slice in moved_slices:
            slice_letter = self.axis_to_slices[axis_of_rot][moved_slice]
            piece_groups = self.get_slice_groups(slice_letter)
            outer_slice = (moved_slice != 1)
            for group_key in piece_groups.keys():
                piece_group = piece_groups[group_key]
                temp_group = copy.deepcopy(piece_group)
                for i in range(len(piece_group)):
                    i_from = (i + move_dir) % num_faces
                    i2 = (i + 1) % num_faces
                    i2_from = (i2 + move_dir) % num_faces
                    # print(temp_group[i_from].keys(), ':\t', adj_faces[i_from], ':', temp_group[i_from][adj_faces[i_from]], '\t\t->\t\t', piece_group[i].keys(), ':\t', adj_faces[i], ':', piece_group[i][adj_faces[i]])
                    piece_group[i][adj_faces[i]] = temp_group[i_from][adj_faces[i_from]]
                    if group_key == 'corners':   # corners require mapping one extra sticker on adjacent
                        piece_group[i][adj_faces[i2]] = temp_group[i_from][adj_faces[i2_from]]
                    if outer_slice:              # outer slice pieces require mapping stickers on said outer slice
                        piece_group[i][slice_letter] = temp_group[i_from][slice_letter]

    # perform a string of moves
    def do_moves(self, move_str: str):
        parsed, moves = parse_moves(move_str)
        if not parsed:
            print("Parsing failed after", end='   ')
            for move in moves:
                print(move[0], end='')
                if move[1] == 3:
                    print("'", end='')
                elif move[1] != 1:
                    print(str(move[1]), end='')
                print(end=' ')
            print('')
        for move in moves:
            self.move(move[0], move[1])

    def CMLL_affects_EO(self, CMLL: str, CMLLsetup: str, initial_EO: str, show_img=False):
        self.do_moves(initial_EO)
        self.do_moves(CMLLsetup)
        previous_state = copy.deepcopy(self.pieces)
        self.do_moves(CMLL)
        # print(self.viscube_image(translucent=True, show_img=show_img, mask=previous_state))
        # print(self.viscube_image(translucent=True, show_img=show_img))
        LSE = ['UF', 'UR', 'UB', 'UL', 'DF', 'DB']
        LSE_before = self.EO_mask(previous_state)
        LSE_after =  self.EO_mask(self.pieces)
        for edge_str in LSE:
            previous_edge = self.get_piece(edge_str, previous_state)
            current_edge = self.get_piece(edge_str)
            if current_edge[edge_str[0]] == previous_edge[edge_str[0]] and current_edge[edge_str[1]] == previous_edge[edge_str[1]]:
                LSE_before[edge_str] = 't'
                LSE_after[edge_str] = 't'
            self.set_piece_color([char for char in edge_str], LSE_before[edge_str], mask=previous_state)
            self.set_piece_color([char for char in edge_str], LSE_after[edge_str])


        url_before = self.viscube_image(translucent=True, show_img=show_img, mask=previous_state)
        url_after = self.viscube_image(translucent=True, show_img=show_img)
        return url_before, url_after

    def EO_mask(self, pieces):
        LSE = ['UF', 'UR', 'UB', 'UL', 'DF', 'DB']
        mask = dict()
        for edge_str in LSE:
            edge = self.get_piece(edge_str, pieces)
            if edge[edge_str[0]] == 'w' or edge[edge_str[0]] == 'y':    # oriented edge
                mask[edge_str] = 'y'
            else:                                                       # misoriented edge
                mask[edge_str] = 'm'
        return mask

    # returns the colors of the stickers on each side
    def get_facelets(self, mask):
        pieces = self.pieces
        if type(mask) == np.ndarray:
            pieces = mask

        facelets = dict()
        for face in face_keys:
            face_facelets = []

            # set the correct range "directions" for rows and columns of different faces
            row_range = col_range = [0, 1, 2]
            if face == 'U':
                row_range = [2, 1, 0]
            if face == 'L' or face == 'B':
                col_range = [2, 1, 0]

            # axes = axis_keys.copy()
            # axes.remove(face_axes[face])  # axes[0] = i_axis, axes[1] = j_axis
            axes = self.std_face_mapping[face]
            index_dict = {move_to_axis[face]: self.slice_to_index[face], axes[0]: 0, axes[1]: 0}
            # print(face, ":\trow", axes[0], "\tcol", axes[1])
            for row in row_range:
                for col in col_range:
                    index_dict[axes[0]] = row
                    index_dict[axes[1]] = col
                    facelet_color = pieces[index_dict['Y']][index_dict['X']][index_dict['Z']][face]
                    face_facelets.append(facelet_color)
            facelets[face] = face_facelets
        return facelets

    # shows a 3D representation of the cube using Visual Cube API
    def viscube_image(self, translucent=False, verbose=False, show_img=True, mask=None):
        facelets: dict = self.get_facelets(mask=mask)
        fc_string = ""
        for face in face_keys:
            for char in facelets[face]:
                fc_string += char
        if verbose:
            print("Visual Cube string")
            for row in range(3):
                for face in range(6):
                    start = 9 * face + 3 * row
                    print(fc_string[start: start + 3], end='   ')
                print('')
            print(fc_string)
        if show_img:
            url = "https://www.rouxer.com/visualcube.php?size=150&bg=white&fmt="
        else:
            url = "https://www.rouxer.com/visualcube.php?size=200&bg=white&fmt=svg"
        if translucent:
            url += "&view=trans"
        url += ('&fc=' + fc_string)
        if show_img:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img.show()
        return url
