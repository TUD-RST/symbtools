"""
This module contains algorithms to investigate the region off attraction of a dynamical system.
"""

import itertools
import numpy as np
from matplotlib import pyplot as plt

from ipydex import IPS, activate_ips_on_exception
activate_ips_on_exception()

node_list = []

node_dict = {}  # save all nodes which we allready created


class NodeDataBase(object):
    def __init__(self):
        # !! quick and dirty
        depth = 10
        self.levels = [list() for i in range(depth)]
        self.all_nodes = []

        # which are new since the last func-application
        self.new_nodes = []

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


ndb = NodeDataBase()


class Node(object):
    """
    represents a point in n-dimensional space and stores information about neighbours, parents and children,
    boundary-properties etc.
    """

    def __init__(self, coords, idcs, parent=None, level=0, node_class="main"):
        self.coords = coords
        self.idcs = idcs
        self.axes = tuple(range(len(coords)))
        self.parent = parent
        self.level = level
        self.func_val = None
        self.boundary_flag = None

        # one of "main", "aux"
        self.node_class = node_class

        # will be a list of len-2-lists; outer list-index ≙ axis (dimension), inner list-entries: neg./pos. direction
        self.neighbours = [list([None, None]) for i in range(len(coords))]

        # counted positive in both directions
        self.distances = [list([None, None]) for i in range(len(coords))]
        self.idx_distances = [list([None, None]) for i in range(len(coords))]

        ndb.add(self)

        if self.level > 0:
            print("->", self)

    def apply(self, func):
        """
        :param func:     function which is to apply
        """

        self.func_val = func(self.coords)

    def set_neigbours(self, dim, n0, n1):
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

        if n1 is not None:
            assert isinstance(n1, Node)
            self.neighbours[dim][1] = n1

            dist = self.coords[dim] - n1.coords[dim]
            assert dist < 0
            self.distances[dim][1] = -dist

            idx_dist = self.idcs[dim] - n1.idcs[dim]
            assert idx_dist < 0
            self.idx_distances[dim][1] = -idx_dist

    def new_node(self, dim, dir):
        """
        create a new main-node between self and the respective neighbor (0 or 1)
        :param dim:
        :param dir:
        :return:
        """

        assert dim in self.axes
        assert dir in (0, 1)

        N1 = self
        N2 = self.neighbours[dim][dir]

        new_idcs = list(N1.idcs)  # make a copy
        new_idcs[dim] = (N1.idcs[dim] + N2.idcs[dim])/2

        new_coords = list(N1.coords)
        new_coords[dim] = (N1.coords[dim] + N2.coords[dim])/2

        new_node = Node(new_coords, new_idcs, node_class="main", level=self.level+1)

        if dir == 0:
            # N2 < new_node < N1
            new_node.set_neigbours(dim, N2, N1)
        else:
            # N1 < new_node < N2
            new_node.set_neigbours(dim, N1, N2)

        node_dict[tuple(new_idcs)] = new_node
        new_node.create_aux_neighbours(dim)

    def create_aux_neighbours(self, ref_dim):
        """

        :param ref_dim:     reference dim (here we already have neighbours)
        :return:
        """

        assert ref_dim in self.axes
        assert self.node_class == "main"

        # reference nodes for the creation of new auxiliary nodes
        R0, R1 = self.neighbours[ref_dim]

        for dim, (n0, n1) in enumerate(self.neighbours):
            if (n0, n1) == (None, None):
                # this is a dimension where new aux nodes need to be created

                new_idcs0 = list(self.idcs)
                new_idcs1 = list(self.idcs)
                new_coords0 = list(self.coords)
                new_coords1 = list(self.coords)

                dist0 = max(R0.distances[dim][0], R1.distances[dim][0])
                dist1 = max(R0.distances[dim][1], R1.distances[dim][1])

                new_coords0[dim] -= dist0
                new_coords1[dim] += dist1

                di0 = max(R0.idx_distances[dim][0], R1.idx_distances[dim][0])
                di1 = max(R0.idx_distances[dim][1], R1.idx_distances[dim][1])

                new_idcs0[dim] -= di0
                new_idcs1[dim] += di1

                # the neighbours get the same level
                N0 = Node(new_coords0, new_idcs0, level=self.level, node_class="aux")
                N1 = Node(new_coords1, new_idcs1, level=self.level, node_class="aux")

                node_dict[tuple(new_idcs0)] = N0
                node_dict[tuple(new_idcs1)] = N1

                self.set_neigbours(dim, N0, N1)

                pass
            else:
                assert dim == ref_dim
                assert isinstance(n0, Node)
                assert isinstance(n1, Node)

    def all_neighbours(self):
        res = []
        for a, b in self.neighbours:
            res.extend((a, b))
        return res

    def __repr__(self):

        return "<N {} ({})|({})>".format(self.node_class, self.idcs, self.coords)


def get_coords_from_meshgrid(mg, idcs):
    """

    :param mg:      list (len N) of equal shaped arrays (like returned by np.meshgrid)
    :param idcs:    N-tuple of ints (indices)
    :return:
    """

    coords = [arr[idcs] for arr in mg]
    return coords


def get_node_for_idcs(mg, idcs):

    if idcs in node_dict:
        the_node = node_dict[idcs]
    else:
        coords = get_coords_from_meshgrid(mg, idcs)
        the_node = Node(coords, idcs)
        node_dict[idcs] = the_node

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

            the_node.set_neigbours(dim_idx, neighbour1, neighbour2)

            neighbour1.set_neigbours(dim_idx, None, the_node)
            neighbour2.set_neigbours(dim_idx, the_node, None)

    return node_dict


def node_list_to_array(nl, selected_idcs=None, cond_func=None):
    """
    """
    if isinstance(nl, Node):
        nl = [nl]

    if selected_idcs is None:
        selected_idcs = (0, 1)

    if cond_func is None:
        def cond_func(n):
            return True

    res = [list() for i in range(len(selected_idcs))]
    # plot the first two dimensions
    for node in nl:
        if not cond_func(node):
            continue
        for idx in selected_idcs:
            res[idx].append(node.coords[idx])

    return np.array(res)


xx = np.linspace(-3, 3, 10)
yy = np.linspace(-3, 3, 5)

XX, YY = mg = np.meshgrid(xx, yy, indexing="ij")

nd = create_nodes_from_mg(mg)


def is_main_node(node):
    return node.node_class == "main"


def is_aux_node(node):
    return node.node_class == "aux"


def test1():

    root_node = nd[2, 2]
    root_node.new_node(0, 0)
    nl0 = node_list_to_array(ndb.levels[0])
    nl1_main = node_list_to_array(ndb.levels[1], cond_func=is_main_node)
    nl1_aux = node_list_to_array(ndb.levels[1], cond_func=is_aux_node)



    plt.plot(*nl0, "k.")
    plt.plot(*nl1_main, "b.")
    plt.plot(*nl1_aux, "g.")

    plt.show()


def func_circle(xx):
    return xx[0]**2 + xx[1]**2 <= 1.3

ndb.apply_func(func_circle)


a_in0 = ndb.get_inner()
a_out0 = ndb.get_outer()

plt.plot(*a_out0, "bo")
plt.plot(*a_in0, "ro", alpha=0.5)

plt.show()



root_node = nd[2, 2]




IPS()

