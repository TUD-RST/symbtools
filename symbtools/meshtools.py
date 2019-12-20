"""
Created on 2019-12-11 11:00:00

@author: Carsten Knoll

This module contains helper-algorithms to investigate the region off attraction of a dynamical system.
Mainly it contains the classes Grid, GridCell and Node which facilitate an adaptive mesh refinement (AMR) scheme in n
dimensions. The used method is quite simple its like parallel interval bisection in n dimensions.

The code was visually tested in 2 and 3 dimensions. No warrenty for correct results.

Documentation is not yet available. But unittests (test/test_meshtools.py) might mitigate that lack.
"""

import itertools
import collections
import numpy as np
from scipy.spatial.distance import hamming

debug_mode = 0
if debug_mode:
    # noinspection PyUnresolvedReferences
    from ipydex import IPS, activate_ips_on_exception
    activate_ips_on_exception()

PointCollection = collections.namedtuple("PointCollection", ["ip", "op", "ibp", "obp"])


class NodeDataBase(object):
    def __init__(self, grid):
        self.grid = grid
        self.levels = collections.defaultdict(list)
        self.all_nodes = []
        self.node_dict = {}

        # these dicts will hold a set (not a list to prevent duplicates) for each level with all nodes which are part
        # of the boundaray when this level was reached.
        # Thus a level-0 node can still be part of the level 1 (or level 2) boundary if it is part
        # of an inhomogeneous level 1 (or level 2)  cell.
        self.inner_boundary_nodes = collections.defaultdict(set)
        self.outer_boundary_nodes = collections.defaultdict(set)

        # same principle for all inner and outer nodes
        self.inner_nodes = collections.defaultdict(set)
        self.outer_nodes = collections.defaultdict(set)

        # which are new since the last func-application
        self.new_nodes = []

        self.recently_evaluated_nodes = []

    def add(self, node):
        """
        add a node to the appriate level-list and to the list of all nodes
        """

        assert isinstance(node, Node)
        self.levels[node.level].append(node)
        self.all_nodes.append(node)
        self.new_nodes.append(node)

    def apply_func(self, func):
        for node in self.new_nodes:
            node.apply(func)

        self.recently_evaluated_nodes = self.new_nodes

        # empty that list (prepare for next evaluation)
        self.new_nodes = []

    @staticmethod
    def is_inner(node):
        return bool(node.func_val)

    @staticmethod
    def is_outer(node):
        return not bool(node.func_val)

    def get_all_or_max_level_nodes(self, only_max_level):
        if only_max_level:
            target_nodes = self.levels[self.grid.max_level]
        else:
            target_nodes = self.all_nodes

        return target_nodes

    def get_inner_nodes(self, idcs=None, only_max_level=True):
        target_nodes = self.get_all_or_max_level_nodes(only_max_level)
        return self.node_list_to_array(target_nodes, idcs, cond_func=self.is_inner)

    def get_outer_nodes(self, idcs=None, only_max_level=True):
        target_nodes = self.get_all_or_max_level_nodes(only_max_level)
        return self.node_list_to_array(target_nodes, idcs, cond_func=self.is_outer)

    def get_inner_boundary_nodes(self, level):

        target_nodes = self.inner_boundary_nodes[level]
        return self.node_list_to_array(target_nodes)

    def get_outer_boundary_nodes(self, level):

        target_nodes = self.outer_boundary_nodes[level]
        return self.node_list_to_array(target_nodes)

    def node_list_to_array(self, nl, selected_idcs=None, cond_func=None):
        """
        Convert a list (length n) of m-dimensional nodes to an r x p array, where r <= n is the number of nodes
        for which all condition functions return True and p <= m is the number of indices (axes) which are selected by
        `selected_indices`.

        :param nl:              node list
        :param selected_idcs:   coord-axes which should be part of the result (for plotting two or or three make sense)

        :param cond_func:       function or sequence of functions mapping a Node-object to either True or False
        :return:

        """
        if isinstance(nl, Node):
            nl = [nl]

        if selected_idcs is None:
            selected_idcs = list(range(self.grid.ndim))

        if cond_func is None:
            # noinspection PyShadowingNames
            def cond_func(node):
                return True

        if isinstance(cond_func, (tuple, list)):
            cond_func_sequence = cond_func

            # noinspection PyShadowingNames
            def cond_func(node):
                r = [f(node) for f in cond_func_sequence]
                return all(r)

        res = [list() for i in range(len(selected_idcs))]
        # plot the first two dimensions
        for node in nl:
            if not cond_func(node):
                continue
            for idx in selected_idcs:
                res[idx].append(node.coords[idx])

        return np.array(res)


class Node(object):
    """
    represents a point in n-dimensional space and stores information about neighbours, parents and children,
    boundary-properties etc.
    """

    def __init__(self, coords, idcs, grid, level=0, node_class="main"):
        self.coords = coords
        self.idcs = tuple(idcs)
        self.axes = tuple(range(len(coords)))
        self.parents = list()
        self.grid = grid
        self.level = level
        self.func_val = None  # will be in {0, 1}
        self.boundary_flag = None  # will be in {-1, 0, 1}  i.e. outer_bound, no bound or inner bound

        self.osn_list = None  # orthogonal semin_neighbours

        # one of "main", "aux"
        self.node_class = node_class

        # will be a list of len-2-lists; outer list-index ≙ axis (dimension), inner list-entries: neg./pos. direction
        self.neighbours = [list([None, None]) for i in range(len(coords))]

        # counted positive in both directions
        self.distances = [list([None, None]) for i in range(len(coords))]
        self.idx_distances = [list([None, None]) for i in range(len(coords))]

        self.grid.ndb.add(self)

    def apply(self, func):
        """
        :param func:     function which is to apply
        """

        self.func_val = func(self.coords)

        if self.func_val:
            self.grid.ndb.inner_nodes[self.grid.max_level].add(self)
        else:
            self.grid.ndb.outer_nodes[self.grid.max_level].add(self)

    def __repr__(self):

        return "<N f:{} {}|{}>".format(self.func_val, self.idcs, self.coords)


class Grid(object):
    def __init__(self, mg):

        # original meshgrid as returned by numpy.meshgrid
        self.mg = mg
        self.all_mg_points = np.array([arr.flat[:] for arr in self.mg])
        self.ndim = len(mg)

        # save all cells in a list
        self.cells = []

        # cells are organized in (child-) levels, beginning from 0
        # in this dict we store lists of the respective cells. Keys are 0, 1, ...
        self.levels = collections.defaultdict(list)
        self.max_level = 0

        self.homogeneous_cells = collections.defaultdict(list)
        self.inhomogeneous_cells = collections.defaultdict(list)

        self.boundary_cells = collections.defaultdict(list)

        self.ndb = NodeDataBase(grid=self)

        minima = np.array([np.min(arr) for arr in self.mg])
        maxima = np.array([np.max(arr) for arr in self.mg])

        # note: all arrays in mg have the same shape
        shapes = np.array(mg[0].shape)
        assert np.all(shapes > 1)
        assert len(shapes) == self.ndim

        self.coord_extention = maxima - minima
        self.coord_stepwidth = self.coord_extention/(shapes - 1)

        self.coords_of_index_origin = minima

        # these objects store index-coordinates which only depend on the dimension
        # we just need to compute them once (see 2d examples)
        # they will be filled by methods below
        self.vertex_local_idcs = None  # e.g.: [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.edge_pair_local_idcs = None  # e.g.: [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((0, 1), (1, 1)), ...]
        self.idx_edge_pairs = None  # [(0, 1), (0, 2), (1, 3), ...] third edge consists of node1 and node3
        self.new_cell_idcs = None  # e.g: [[(0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5)], [...]]

        self._find_vertex_local_idcs()
        self._find_edge_pairs()
        self._find_child_idcs()

        self.create_cells()

    def _find_vertex_local_idcs(self):
        """
        Create a list of index-tuples which describe the indices of the nodes of the first cell. e.g. in 2d:
        [(0, 0), (0, 1), (1, 0), (1, 1)]

        :return:
        """
        self.vertex_local_idcs = list(itertools.product([0.0, 1.0], repeat=self.ndim))

    def _find_edge_pairs(self):
        """
        For a generic n-dim cell find all index pairs which together define an edge of this cell (no diagonals).
        We use hamming distance here (see docs of scipy.spatial.distance.hamming): An endge is a pair of nodes which
        differ in exactly one component

        :return:
        """

        assert self.edge_pair_local_idcs is None

        all_pairs = list(itertools.combinations(self.vertex_local_idcs, 2))

        def is_edge(pair):
            return hamming(pair[0], pair[1])*self.ndim == 1

        edge_pairs = list(filter(is_edge, all_pairs))

        self.edge_pair_local_idcs = edge_pairs

        # we also might need the index-numbers of the nodes, i.e. (0, 0) -> 0, (0, 1) -> 1, (1, 0) -> 2

        idx_nbr_dict = {}

        for idx_nbr, vertex in enumerate(self.vertex_local_idcs):
            idx_nbr_dict[vertex] = idx_nbr

        idx_edge_pairs = []
        for n1, n2 in edge_pairs:
            idx_edge_pairs.append((idx_nbr_dict[n1], idx_nbr_dict[n2]))

        self.idx_edge_pairs = idx_edge_pairs

    def _find_child_idcs(self):
        """
        For a generic n-dim cell find all 2**ndim subcells. A subcell here is a tuples of fractional indices,
        which describe the vertex nodes of the new smaller cell.

        Example in 2d case: Original nodes:  [(0, 0), (0, 1), (1, 0), (1, 1)]
        First subcell: [(0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5)] (remaining/reference vertex: (0, 0))

        :return:
        """

        assert self.new_cell_idcs is None

        self.new_cell_idcs = []

        step_array = np.array(self.vertex_local_idcs)*.5

        # the vertices of the first child cell (lower left child cell in 2d) are the reference nodes
        # (lower left corner) of all child cells
        new_reference_nodes = tuple(step_array)

        for new_reference_node in new_reference_nodes:
            # make use of broadcasting here
            tmp = new_reference_node + step_array
            self.new_cell_idcs.append(tmp)

    def indices_to_coords(self, idcs):

        return self.coords_of_index_origin + np.array(idcs)*self.coord_stepwidth

    def get_node_for_idcs(self, idcs, coords=None, level=0):
        """
        If `idcs` is a valid key of node_dict return the corresponding node, else create a new one.
        Only integer indices are possible here. coords are taken from the meshgrid.

        :param idcs:
        :param coords:
        :param level:
        :return:
        """

        idcs = tuple(idcs)
        if idcs in self.ndb.node_dict:
            the_node = self.ndb.node_dict[idcs]
        else:
            if coords is None:
                # !!! wont work with fractional indices
                # coords = get_coords_from_meshgrid(self.mg, idcs)
                coords = self.indices_to_coords(idcs)

            assert len(coords) == len(idcs)

            the_node = Node(coords, idcs, grid=self, level=level)
            self.ndb.node_dict[idcs] = the_node

        return the_node

    def create_cells(self):
        """

        :return:
        """
        lengths = self.mg[0].shape

        index_sequences = []
        for L in lengths:
            assert L > 2
            index_sequences.append(range(0, L - 1))

        inner_index_tuples = itertools.product(*index_sequences)

        cells = []
        for idcs in inner_index_tuples:
            cell = self.create_basic_cell_from_mg_node(idcs)

            cells.append(cell)

        return cells

    def create_basic_cell_from_mg_node(self, node_idcs):
        """

        :param node_idcs:
        :return:
        """

        idcs_arr = np.array(node_idcs)

        all_idcs = idcs_arr + np.array(self.vertex_local_idcs)

        nodes = []
        for idcs in all_idcs:
            node = self.get_node_for_idcs(idcs)
            nodes.append(node)

        cell = GridCell(nodes, self, level=0)

        return cell

    def classify_cells_by_homogenity(self):
        """
        Iterate over the cells of the current max_level and find those which are not homogenous
        :return:
        """

        for cell in self.levels[self.max_level]:
            if cell.is_homogeneous():
                self.homogeneous_cells[self.max_level].append(cell)
                cell.set_boundary_status(False)
            else:
                self.inhomogeneous_cells[self.max_level].append(cell)
                cell.set_boundary_status(True)
                self.boundary_cells[self.max_level].append(cell)

    def divide_boundary_cells(self):
        max_level = self.max_level

        assert self.boundary_cells == self.inhomogeneous_cells

        for cell in self.boundary_cells[max_level]:
            cell.make_childs()

    def refinement_step(self, char_func):

        # this will do nothing in the first call (because we have no boundary cells yet)
        self.divide_boundary_cells()

        self.ndb.apply_func(char_func)
        self.classify_cells_by_homogenity()

        return PointCollection( self.ndb.get_inner_nodes(), self.ndb.get_outer_nodes(),
                                self.ndb.get_inner_boundary_nodes(self.max_level),
                                self.ndb.get_outer_boundary_nodes(self.max_level) )


class GridCell(object):

    def __init__(self, nodes, grid, level=0):
        self.ndim = len(nodes[0].coords)
        assert len(nodes) == 2**self.ndim

        self.grid = grid
        self.vertex_nodes = nodes
        self.level = level
        self.parent_cell = None
        self.child_cells = None
        self._is_homogeneous = None

        self.grid.cells.append(self)
        self.grid.levels[self.level].append(self)
        if self.grid.max_level != self.level:
            assert self.grid.max_level == self.level - 1
            self.grid.max_level +=1

    def is_homogeneous(self):

        cell_res = np.array([node.func_val for node in self.vertex_nodes])

        self._is_homogeneous = np.alltrue(cell_res) or np.alltrue(np.logical_not(cell_res))

        return self._is_homogeneous

    def make_childs(self):

        new_level = self.level + 1
        new_cells = []
        for nci in self.grid.new_cell_idcs:

            index_diffs = nci*0.5**self.level
            nodes = []
            for index_diff in index_diffs:

                new_idcs = tuple(np.array(self.vertex_nodes[0].idcs) + index_diff)

                node = self.grid.get_node_for_idcs(idcs=new_idcs, level=self.level+1)
                # node = get_or_create_node(coords=None, idcs=new_idcs, grid=self.grid)
                nodes.append(node)

            new_cell = GridCell(nodes, self.grid, level=new_level)
            new_cell.parent_cell = self

            new_cells.append(new_cell)
        self.child_cells = new_cells

        return new_cells

    def get_edge_coords(self):

        edges = []
        for n1, n2 in self.grid.idx_edge_pairs:
            edges.append((self.vertex_nodes[n1].coords, self.vertex_nodes[n2].coords))

        return edges

    def get_vertex_coords(self):

        vertices = []
        for v in self.vertex_nodes:
            vertices.append(v.coords)

        return np.array(vertices)

    def set_boundary_status(self, flag):
        """
        Notify each node about its boundary status.

        :param flag:
        :return:
        """
        for node in self.vertex_nodes:
            if flag:
                func_val_int = int(node.func_val)
                assert func_val_int in (0, 1)
                if func_val_int == 0:
                    self.grid.ndb.outer_boundary_nodes[self.grid.max_level].add(node)
                    node.boundary_flag = -1
                else:
                    self.grid.ndb.inner_boundary_nodes[self.grid.max_level].add(node)
                    node.boundary_flag = 1

            elif node.boundary_flag is None:
                # only set to 0 if the node was not already flagged as boundary
                node.boundary_flag = 0

    def __repr__(self):
        return "<Cell: level:{}, rn:{}, rnc:{}>".format(self.level, self.vertex_nodes[0].idcs,
                                                        self.vertex_nodes[0].coords)


def get_index_difference(idcs1, idcs2):
    """
    Assume that the index tuples differ by exactly one index. Find out which dimension-index that is and the difference
    (i.e. direction: 0 or 1)

    :param idcs1:
    :param idcs2:

    :return: dim, dir
    """

    assert len(idcs1) == len(idcs2)

    for dim, (i1, i2) in enumerate(zip(idcs1, idcs2)):
        if i1 != i2:
            diff = i2 - i1
            assert -1 <= diff <= 1 and diff != 0
            break
    else:
        # there was no break-reason
        msg = "No difference between {} and {} was found which is against the assumption".format(idcs1, idcs2)
        raise ValueError(msg)

    dir = int(diff > 0)

    return dim, dir, diff


def func_circle(xx):
    return xx[0]**2 + xx[1]**2 < 1.3


def func_sphere_nd(xx):
    """
    Characteristic function of n-dimensional sphere

    :param xx:
    :return:
    """
    return np.sum(xx**2) < 1.3

