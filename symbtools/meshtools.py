"""
This module contains algorithms to investigate the region off attraction of a dynamical system.
"""

import itertools
import numpy as np
from matplotlib import pyplot as plt

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
        for node in self.recently_evaluated_nodes:
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



ndb = NodeDataBase()


class Node(object):
    """
    represents a point in n-dimensional space and stores information about neighbours, parents and children,
    boundary-properties etc.
    """

    def __init__(self, coords, idcs, parent=None, level=0, node_class="main"):
        self.coords = coords
        self.idcs = tuple(idcs)
        self.axes = tuple(range(len(coords)))
        self.parent = parent
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

        ndb.add(self)

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

    :param mg:      list (len N) of equal shaped arrays (like returned by np.meshgrid)
    :param idcs:    N-tuple of ints (indices)
    :return:
    """

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
    if idcs in ndb.node_dict:
        the_node = ndb.node_dict[idcs]
    else:
        the_node = Node(coords, idcs, **kwargs)

    return the_node


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

    ndb.apply_func(func_circle)
    ndb.set_boundary_flags()

    a_in0 = ndb.get_inner()
    a_out0 = ndb.get_outer()

    b_in0 = ndb.get_inner_boundary()
    b_out0 = ndb.get_outer_boundary()

    ndb.insert_new_nodes()

    # plot inner and outer points (level 0)
    plt.plot(*a_out0, "bo", alpha=0.2, ms=5)
    plt.plot(*a_in0, "ro", alpha=0.2, ms=5)

    # plot inner and outer boundary points (level 0)
    plt.plot(*b_out0, "bo", ms=3)
    plt.plot(*b_in0, "ro", ms=3)

    plt.title("levels 0")
    plt.savefig("level0.png")

    # get and plot level 1 points (main and aux)
    nl1_main = node_list_to_array(ndb.levels[1], cond_func=is_main_node)
    nl1_aux = node_list_to_array(ndb.levels[1], cond_func=is_aux_node)

    plt.plot(*nl1_main, "m.")
    plt.plot(*nl1_aux, "gx", ms=3)

    plt.title("levels 0 and 1 (not evaluated)")

    plt.savefig("level1a.png")

    plt.figure()

    # - - - - - - - -

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

