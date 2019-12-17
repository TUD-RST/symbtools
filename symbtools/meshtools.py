"""
This module contains algorithms to investigate the region off attraction of a dynamical system.
"""

import itertools
import collections
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import hamming

from ipydex import IPS, activate_ips_on_exception
activate_ips_on_exception()


class NodeDataBase(object):
    def __init__(self):
        # !! quick and dirty
        depth = 10
        self.levels = [list() for i in range(depth)]
        self.all_nodes = []
        self.node_dict = {}

        # which are new since the last func-application
        self.new_nodes = []

        self.recently_evaluated_nodes = []
        self.inner_boundary_nodes = []

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

        # empty that list
        self.new_nodes = []

    @staticmethod
    def is_inner(node):
        return bool(node.func_val)

    @staticmethod
    def is_outer(node):
        return not bool(node.func_val)

    def get_inner(self, idcs=None):
        return node_list_to_array(self.all_nodes, idcs, cond_func=self.is_inner)

    def get_outer(self, idcs=None):
        return node_list_to_array(self.all_nodes, idcs, cond_func=self.is_outer)

    def get_outer_boundary(self, idcs=None):
        return node_list_to_array(self.all_nodes, idcs, cond_func=lambda node: node.boundary_flag==-1)

    def get_inner_boundary(self, idcs=None):
        return node_list_to_array(self.all_nodes, idcs, cond_func=lambda node: node.boundary_flag==1)

    def set_boundary_flags(self):

        # !! boundary status might have changed (check all former boundary nodes)
        # for node in self.recently_evaluated_nodes:
        for node in self.all_nodes:
            fv = node.func_val
            assert fv in (True, False)

            boundary_flag = (-1, 1)[int(fv)]

            for nb in node.all_neighbours(omit_none=True):
                if nb.func_val != fv:
                    # at least one neigbour has a different value
                    node.boundary_flag = boundary_flag
                    if fv:
                        self.inner_boundary_nodes.append(node)
                    break
            else:
                # there was no break
                node.boundary_flag = 0

    def insert_new_nodes(self, insert_aux_nodes=True):

        new_nodes = []

        for node in self.inner_boundary_nodes:

            # find all neighbours with different function values
            different_neighbours = []
            for nb in node.all_neighbours(omit_none=True):
                if nb.func_val != node.func_val:
                    assert nb.boundary_flag != 0
                    different_neighbours.append(nb)

            for nb in different_neighbours:
                dim, dir, diff = get_index_difference(node.idcs, nb.idcs)

                new_node = halfway_node(node, nb, level=node.level+1)

                new_nodes.append(new_node)

        # auxnodes are inserted after main nodes becaus sometimes main nodes are "accidentally" placed at the
        # correct location

        if insert_aux_nodes:
            for new_node in new_nodes:

                for dim, osn_tuple in enumerate(new_node.osn_list):
                    if osn_tuple is None:
                        continue

                    (a0, a1), (b0, b1) = osn_tuple

                    aux0 = halfway_node(a0, a1, level=new_node.level, node_class="aux")
                    aux1 = halfway_node(b0, b1, level=new_node.level, node_class="aux")

                    new_node.set_neighbours(dim, aux0, aux1)


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
        self.level = level
        self.func_val = None
        self.boundary_flag = None

        self.osn_list = None  # orthogonal semin_neighbours

        # one of "main", "aux"
        self.node_class = node_class

        # will be a list of len-2-lists; outer list-index ≙ axis (dimension), inner list-entries: neg./pos. direction
        self.neighbours = [list([None, None]) for i in range(len(coords))]

        # counted positive in both directions
        self.distances = [list([None, None]) for i in range(len(coords))]
        self.idx_distances = [list([None, None]) for i in range(len(coords))]

        grid.ndb.add(self)

    def apply(self, func):
        """
        :param func:     function which is to apply
        """

        self.func_val = func(self.coords)

    def set_neighbours(self, dim, n0, n1, reciprocity=True):
        """

        :param dim:     axis (dimension)
        :param n0:      neighbour in negative direction
        :param n1:      neighbour in positive direction

        :return:        None
        """

        assert dim in self.axes

        if n0 is not None:
            assert isinstance(n0, Node)
            self.neighbours[dim][0] = n0

            dist = self.coords[dim] - n0.coords[dim]
            assert dist > 0
            self.distances[dim][0] = dist

            idx_dist = self.idcs[dim] - n0.idcs[dim]
            assert idx_dist > 0
            self.idx_distances[dim][0] = idx_dist

            if reciprocity:
                n0.set_neighbours(dim, None, self, reciprocity=False)

        if n1 is not None:
            assert isinstance(n1, Node)
            self.neighbours[dim][1] = n1

            dist = self.coords[dim] - n1.coords[dim]
            assert dist < 0
            self.distances[dim][1] = -dist

            idx_dist = self.idcs[dim] - n1.idcs[dim]
            assert idx_dist < 0
            self.idx_distances[dim][1] = -idx_dist

            if reciprocity:
                n1.set_neighbours(dim, self, None, reciprocity=False)

    def all_neighbours(self, omit_none=False):
        res = []
        for a, b in self.neighbours:
            if omit_none:
                if a is not None:
                    res.append(a)

                if b is not None:
                    res.append(b)
            else:
                res.extend((a, b))
        return res

    def __repr__(self):

        return "<N {} {}|{}>".format(self.node_class, self.idcs, self.coords)


class Grid(object):
    def __init__(self, mg):

        # original meshgrid as returned by numpy.meshgrid
        self.mg = mg
        self.ndim = len(mg)
        self.cells = []
        self.ndb = NodeDataBase()

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

        for local_node_idx in self.vertex_local_idcs:
            local_node_idx = np.array(local_node_idx)

            # produce an array which has a -1 where local_node_idx == 1 and 1 elswhere
            signs = (local_node_idx*0 - 1) ** local_node_idx

            # make use of broadcasting here
            tmp = local_node_idx + signs*step_array
            self.new_cell_idcs.append(tmp)

    def indices_to_coords(self, idcs):

        return self.coords_of_index_origin + np.array(idcs)*self.coord_stepwidth

    def get_node_for_idcs(self, idcs, coords=None):
        """
        If `idcs` is a valid key of node_dict return the corresponding node, else create a new one.
        Only integer indices are possible here. coords are taken from the meshgrid.

        :param mg:
        :param idcs:
        :param coords:
        :return:
        """

        idcs = tuple(idcs)
        if idcs in self.ndb.node_dict:
            the_node = self.ndb.node_dict[idcs]
        else:
            if coords is None:
                coords = get_coords_from_meshgrid(self.mg, idcs)

            assert len(coords) == len(idcs)

            the_node = Node(coords, idcs, grid=self)
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

        idcs_arr = np.array(node_idcs)

        all_idcs = idcs_arr + np.array(self.vertex_local_idcs)

        nodes = []
        for idcs in all_idcs:
            node = self.get_node_for_idcs(idcs)
            nodes.append(node)

        cell = GridCell(nodes, self, level=0)

        return cell


class GridCell(object):

    def __init__(self, nodes, grid, level=0):
        self.ndim = len(nodes[0].coords)
        assert len(nodes) == 2**self.ndim

        self.grid = grid
        self.vertex_nodes = nodes
        self.level = level
        self.parent_cell = None
        self.child_cells = None
        self.is_homogeneous = None

        self.grid.cells.append(self)

    def check_homogenity(self):

        cell_res = np.array([node.func_val for node in self.vertex_nodes])

        self.is_homogeneous = np.alltrue(cell_res) or np.alltrue(np.logical_not(cell_res))

        return self.is_homogeneous

    def make_childs(self):

        new_level = self.level + 1
        new_cells = []
        for nci in self.grid.new_cell_idcs:

            index_diffs = nci*0.5**self.level
            nodes = []
            for index_diff in index_diffs:

                new_idcs = tuple(np.array(self.idcs) + index_diff)
                node = get_or_create_node(coords=None, idcs=new_idcs, grid=self.grid)
                nodes.append(node)

            new_cell = GridCell(nodes, self.grid, level=new_level)
            new_cells.append(new_cell)

        return new_cells

    def get_edge_coords(self):

        edges = []
        for n1, n2 in self.grid.idx_edge_pairs:
            edges.append((self.vertex_nodes[n1].coords, self.vertex_nodes[n2].coords))

        return edges


###


###


def absmax(*args):
    """
    return the argument which has the maximum absolute value

    :param args:
    :return:
    """

    abs_values = np.abs(np.array(args))
    # noinspection PyTypeChecker
    return args[np.argmax(abs_values)]


def modify_tuple(tup, idx, diff):
    """
    takes a tuple, changes the value at index `idx` by diff and returns the new tuple
    :param tup:
    :param idx:
    :param diff:

    :return:        changed tuple
    """
    tmp = list(tup)
    tmp[idx] += diff
    return tuple(tmp)


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


def halfway_node(n0, n1, level, node_class="main"):
    """
    Takes two nodes which are "level-neighbours" and find/create a new node in between

    :param n0:
    :param n1:
    :param level:
    :return:
    """

    dim, dir, diff = get_index_difference(n0.idcs, n1.idcs)

    if diff < 0:
        dir = 1 - dir
        diff*=-1
        n0, n1 = n1, n0

    new_idcs = modify_tuple(n0.idcs, dim, diff/2)

    # TODO: why are coords not arrays?
    new_coords = tuple((np.array(n0.coords) + np.array(n1.coords)) / 2)

    new_node = get_or_create_node(new_coords, new_idcs, level=level, node_class=node_class)
    new_node.set_neighbours(dim, n0, n1)

    # this is used currently only for debugging
    new_node.parents.append((n0, n1))

    if node_class == "main" and new_node.osn_list is None:
        new_node.osn_list = get_all_othogonal_semi_neighbours(n0, n1, dim)

    return new_node


def get_orthognal_semi_neighbours(r0, r1, dim, dir):

    """
    Consider the following situation:

    (b0)                   (b1)

    (x?)

    (r0)        (bn)       (r1)        dim ↑  ref_dim →

    The base-node bn was created between r0 and r1 and has these two as reference neighbours.
    We are interested in b0 and b1 (to find/create an aux-node in between). Maybe on one side there is
    an additional node x in between r0 an b0. We omit that asymmetry.

    :param base_node:
    :param r0:
    :param r1:
    :param dim:
    :return:
    """

    # candidates:

    b0c = r0.neighbours[dim][dir]
    b1c = r1.neighbours[dim][dir]

    diff0 = get_index_difference(r0.idcs, b0c.idcs)[2]
    diff1 = get_index_difference(r1.idcs, b1c.idcs)[2]

    # this ensures to omit x (asymmetric resolution)
    diff = absmax(diff0, diff1)

    b0_idcs = modify_tuple(r0.idcs, dim, diff)
    b1_idcs = modify_tuple(r1.idcs, dim, diff)

    b0 = ndb.node_dict[b0_idcs]
    b1 = ndb.node_dict[b1_idcs]

    return b0, b1


def get_all_othogonal_semi_neighbours(r0, r1, ref_dim):
    """

    Consider the following situation:

    (b0)                   (b1)

    (x?)

    (r0)        (bn)       (r1)        dim ↑  ref_dim →


    (a0)                   (a1)

    Apply get_get_orthognal_semi_neighbours in all dimensions (but ref_dim) and all directions


    :param r0:
    :param r1:
    :param ref_dim:
    :return:
    """

    osn = []

    for dim in r0.axes:
        if dim == ref_dim:
            osn.append(None)
            continue

        a0, a1 = get_orthognal_semi_neighbours(r0, r1, dim, dir=0)
        b0, b1 = get_orthognal_semi_neighbours(r0, r1, dim, dir=1)

        osn.append(((a0, a1), (b0, b1)))

    return osn


def get_coords_from_meshgrid(mg, idcs):
    """

    :param mg:      sequence (len N) of equal shaped arrays (like returned by np.meshgrid)
    :param idcs:    N-tuple of ints (indices)
    :return:
    """

    idcs = tuple(int(idx) for idx in idcs)

    coords = [arr[idcs] for arr in mg]
    return coords


def get_node_for_idcs(mg, idcs, coords=None):
    """
    If `idcs` is a valid key of node_dict return the corresponding node, else create a new one.
    Only integer indices are possible here. coords are taken from the meshgrid.

    :param mg:
    :param idcs:
    :param coords:
    :return:
    """

    assert False, "Use grid.get_node_for_idcs instead"

    idcs = tuple(idcs)
    if idcs in ndb.node_dict:
        the_node = ndb.node_dict[idcs]
    else:
        if coords is None:
            coords = get_coords_from_meshgrid(mg, idcs)

        assert len(coords) == len(idcs)

        the_node = Node(coords, idcs)
        ndb.node_dict[idcs] = the_node

    return the_node


def get_or_create_node(coords, idcs, **kwargs):
    """
    If `idcs` is a valid key of node_dict return the corresponding node, else create a new one.

    :param coords:
    :param idcs:
    :param kwargs:
    :return:
    """

    assert False, "Use grid.get_node_for_idcs instead"

    if idcs in ndb.node_dict:
        the_node = ndb.node_dict[idcs]
    else:
        # node for the provided indices does not yet exist -> create new node

        grid = kwargs.get("grid")

        if coords is None:
            coords = grid.indices_to_coords(idcs)

        the_node = Node(coords, idcs, **kwargs)

        # add new node to node_dict
        ndb.node_dict[idcs] = the_node

    return the_node


def create_basic_cell_from_mg_node(grid, node_idcs):

    idcs_arr = np.array(node_idcs)

    all_idcs = idcs_arr + np.array(grid.vertex_local_idcs)

    nodes = []
    for idcs in all_idcs:
        node = get_node_for_idcs(grid.mg, idcs)
        nodes.append(node)

    cell = GridCell(nodes, grid, level=0)

    return cell


def create_grid_from_mg(grid):
    """

    :param grid:
    :return:
    """
    ndim = len(grid.mg)
    lengths = grid.mg[0].shape

    assert ndim == len(lengths)
    index_sequences = []
    for L in lengths:
        assert L > 2
        index_sequences.append(range(0, L - 1))

    inner_index_tuples = itertools.product(*index_sequences)

    cells = []
    for idcs in inner_index_tuples:

        cell = create_basic_cell_from_mg_node(grid, idcs)

        cells.append(cell)

    return cells


def create_nodes_from_mg(mg):
    """
    :param mg:  list (len N) of equal shaped arrays (like returned by np.meshgrid)
    :return:
    """
    ndim = len(mg)
    lengths = mg[0].shape

    assert ndim == len(lengths)
    index_sequences = []
    for L in lengths:
        assert L > 2
        index_sequences.append(range(1, L-1))

    # create a sequence of tuples of the inner indices like [(1, 1), (1, 2), ...]
    inner_index_tuples = itertools.product(*index_sequences)

    for idcs in inner_index_tuples:

        print(idcs)

        the_node = get_node_for_idcs(mg, idcs)

        # find neighbors and connect them
        for dim_idx, arr_idx in enumerate(idcs):

            idcs1, idcs2 = list(idcs), list(idcs)
            idcs1[dim_idx] -= 1
            idcs2[dim_idx] += 1
            idcs1, idcs2 = tuple(idcs1), tuple(idcs2)

            neighbour1 = get_node_for_idcs(mg, idcs1)
            neighbour2 = get_node_for_idcs(mg, idcs2)

            the_node.set_neighbours(dim_idx, neighbour1, neighbour2)

            neighbour1.set_neighbours(dim_idx, None, the_node)
            neighbour2.set_neighbours(dim_idx, the_node, None)

    return ndb.node_dict


def node_list_to_array(nl, selected_idcs=None, cond_func=None):
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
        selected_idcs = (0, 1)

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


def is_main_node(node):
    return node.node_class == "main"


def is_aux_node(node):
    return node.node_class == "aux"


def test1():

    root_node = nd[2, 2]
    nl0 = node_list_to_array(ndb.levels[0])
    nl1_main = node_list_to_array(ndb.levels[1], cond_func=is_main_node)
    nl1_aux = node_list_to_array(ndb.levels[1], cond_func=is_aux_node)

    plt.plot(*nl0, "k.")
    plt.plot(*nl1_main, "b.")
    plt.plot(*nl1_aux, "g.")

    plt.show()


def func_circle(xx):
    return xx[0]**2 + xx[1]**2 <= 1.3


if __name__ == "__main__":
    xx = np.linspace(-4, 4, 9)
    yy = np.linspace(-4, 4, 9)

    XX, YY = mg = np.meshgrid(xx, yy, indexing="ij")

    nd = create_nodes_from_mg(mg)

    self = Grid(mg)
    ndb = self.ndb

    ndb.apply_func(func_circle)
    ndb.set_boundary_flags()

    a_in0 = ndb.get_inner()
    a_out0 = ndb.get_outer()

    b_in0 = ndb.get_inner_boundary()
    b_out0 = ndb.get_outer_boundary()

    # plot inner and outer points (level 0)
    plt.plot(*a_out0, "bo", alpha=0.2, ms=5)
    plt.plot(*a_in0, "ro", alpha=0.2, ms=5)

    # plot inner and outer boundary points (level 0)
    plt.plot(*b_out0, "bo", ms=3)
    plt.plot(*b_in0, "ro", ms=3)

    plt.title("levels 0")
    plt.savefig("level0.png")

    # - - - - - - - -

    # insert level 1 nodes
    ndb.insert_new_nodes()

    # get and plot level 1 points (main and aux)
    nl1_main = node_list_to_array(ndb.levels[1], cond_func=is_main_node)
    nl1_aux = node_list_to_array(ndb.levels[1], cond_func=is_aux_node)

    plt.plot(*nl1_main, "m.")
    plt.plot(*nl1_aux, "gx", ms=3)

    plt.title("levels 0 and 1 (not evaluated)")

    plt.savefig("level1a.png")

    plt.figure()

    ndb.apply_func(func_circle)
    ndb.set_boundary_flags()

    a_in0 = ndb.get_inner()
    a_out0 = ndb.get_outer()

    b_in0 = ndb.get_inner_boundary()
    b_out0 = ndb.get_outer_boundary()

    # plot inner and outer points (level 0+1)
    plt.plot(*a_out0, "bo", alpha=0.2, ms=5)
    plt.plot(*a_in0, "ro", alpha=0.2, ms=5)

    # plot inner and outer boundary points (level 0+1)
    plt.plot(*b_out0, "bo", ms=3)
    plt.plot(*b_in0, "ro", ms=3)

    plt.title("levels 0 and 1")

    plt.savefig("level1b.png")

    # - - - - - - - -

    # insert level 2 nodes

    ndb.insert_new_nodes()

    nl2_main = node_list_to_array(ndb.levels[2], cond_func=is_main_node)
    nl2_aux = node_list_to_array(ndb.levels[2], cond_func=is_aux_node)

    plt.plot(*nl2_main, "m.")
    plt.plot(*nl2_aux, "gx", ms=3)

    plt.title("levels 0, 1 and 2 (not evaluated)")

    plt.savefig("level2a.png")

    plt.show()
    exit()

    ndb.apply_func(func_circle)
    ndb.set_boundary_flags()

    a_in0 = ndb.get_inner()
    a_out0 = ndb.get_outer()

    b_in0 = ndb.get_inner_boundary()
    b_out0 = ndb.get_outer_boundary()

    # plot inner and outer points (level 0+1)
    plt.plot(*a_out0, "bo", alpha=0.2, ms=5)
    plt.plot(*a_in0, "ro", alpha=0.2, ms=5)

    # plot inner and outer boundary points (level 0+1)
    plt.plot(*b_out0, "bo", ms=3)
    plt.plot(*b_in0, "ro", ms=3)

    plt.title("levels 0, 1 and 2")

    plt.savefig("level2b.png")


    # - - - - - - - -

    # insert level 3 nodes

    ndb.insert_new_nodes()

    nl_main = node_list_to_array(ndb.levels[2], cond_func=is_main_node)
    nl_aux = node_list_to_array(ndb.levels[2], cond_func=is_aux_node)

    plt.plot(*nl_main, "m.")
    plt.plot(*nl_aux, "gx", ms=3)

    plt.title("levels 0, 1, 2 and 3 (not evaluated)")

    plt.savefig("level3a.png")

    plt.show()


    """
    General procedure:
    
    1. generate initial nodes
    2. evaluate function on new nodes
    3. determine boundary status (outer boundary -1, no boundary 0, inner boundary 1)
    4. insert new main nodes where necessary
        implicitly add auxiliary nodes such that every main node has well defined neighbours
        thereby node_class can be upgraded from aux to main
    5.  update neighbour-relations    
    6.  go to 2.
    
    """




    IPS()

