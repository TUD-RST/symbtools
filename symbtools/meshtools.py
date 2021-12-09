"""
Created on 2019-12-11 11:00:00

@author: Carsten Knoll
@author: Yuewen He

This module contains helper-algorithms to investigate the region of attra-
ction (ROA) of a dynamical system.
Mainly it contains the classes Grid, GridCell and Node which facilitate
an adaptive mesh refinement (AMR) scheme in n
dimensions. The used method is quite simple its like parallel interval
bisection in n dimensions.

The code was visually tested in 2 and 3 dimensions. No warrenty for correct
 results.

Documentation is not yet available. But unittests (test/test_meshtools.py)
might mitigate that lack.

Abbreviations and Terminology:
ROA: region of attraction
ROI: region of investigation (determined by the initial meshgrid)
AMR: adaptive mesh refinement

BC:  boundary cell (w.r.t ROA boundary)
DC:  domain cell (opposite of boundary cell)
iDC: inhomogeneous DC (vertices are homogenous but other points of the cell have different status (can only be
     determined at later refinement levels))

"""

import itertools
import collections
import numpy as np
from scipy.spatial.distance import hamming
import symbtools as st
import scipy.integrate as sc_integrate
import os
# this is for debugging
import matplotlib.pyplot as plt
import matplotlib.patches as mp
debug_mode = 1
if debug_mode:
    # noinspection PyUnresolvedReferences
    from ipydex import IPS, activate_ips_on_exception
    activate_ips_on_exception()

PointCollection = collections.namedtuple("PointCollection",
                                         ["ip", "op", "ibp", "obp"])


class NodeDataBase(object):
    """
    Node-databank
    """
    def __init__(self, grid):
        self.grid = grid
        self.levels = collections.defaultdict(list)
        self.all_nodes = []
        self.node_dict = {}
        self.roi_boundary_nodes = []

        # these dicts will hold a set (not a list to prevent duplicates) for
        # each level with all nodes which are part
        # of the boundaray when this level was reached.
        # Thus a level-0 node can still be part of the level 1 (or level 2)
        # boundary if it is part
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
        add a node to the appriate level-list and to the list of all nodes；
        if the node is on the boundary of the roi, then add the node to the
        boundary node list
        """

        assert isinstance(node, Node)
        self.levels[node.level].append(node)
        self.all_nodes.append(node)
        self.new_nodes.append(node)
        if node.is_on_roi_boundary():

            self.roi_boundary_nodes.append(node)

    def apply_func(self, func, simulation_mod):
        """

        :param func:
        :param simulation_mod:
        flag to determine whether simulation results should be stored inside node
        :return:
        """
        for node in self.new_nodes:
            node.apply(func, simulation_mod)

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
        """
        get all nodes in the maximal level or get all nodes
        """
        if only_max_level:
            target_nodes = self.levels[self.grid.max_level]
        else:
            target_nodes = self.all_nodes

        return target_nodes

    def get_inner_nodes(self, idcs=None, only_max_level=True):
        """
        get all inside nodes in the maximal level
        """
        target_nodes = self.get_all_or_max_level_nodes(only_max_level)
        return self.node_list_to_array(target_nodes,
                                       idcs, cond_func=self.is_inner)

    def get_outer_nodes(self, idcs=None, only_max_level=True):
        """
        get all outside nodes in the maximal level
        """
        target_nodes = self.get_all_or_max_level_nodes(only_max_level)
        return self.node_list_to_array(target_nodes,
                                       idcs, cond_func=self.is_outer)

    def get_inner_boundary_nodes(self, level):
        """
        get the nodes of boundary cells, which are also inside of the boundary.
        Notice: the boundary of the aim form will pass across the boundary cells
        and divides the nodes of the boundary cells into 2 type:
        inner_boundary_nodes and outer_boundary_nodes
        """
        target_nodes = self.inner_boundary_nodes[level]
        return self.node_list_to_array(target_nodes)

    def get_outer_boundary_nodes(self, level):
        """
        get the nodes of boundary cells, which are also outside of the boundary.
        """

        target_nodes = self.outer_boundary_nodes[level]
        return self.node_list_to_array(target_nodes)

    def node_list_to_array(self, nl, selected_idcs=None, cond_func=None):
        """
        Convert a list (length n) of m-dimensional nodes to an r x p array,
        where r <= n is the number of nodes
        for which all condition functions return True and p <= m is the number
        of indices (axes) which are selected by
        `selected_indices`.

        :param nl:              node list
        :param selected_idcs:   coord-axes which should be part of the result
                                (for plotting two or or three make sense)

        :param cond_func:       function or sequence of functions mapping a
                                Node-object to either True or False
        :return:

        """
        if isinstance(nl, Node):
            nl = [nl]

        if selected_idcs is None:
            # 2-dimension selected_idcs=[0, 1]; 3-d:[0, 1, 2]
            selected_idcs = list(range(self.grid.ndim))

        if cond_func is None:
            # noinspection PyShadowingNames,PyUnusedLocal
            def cond_func(node):
                return True

        # noinspection PyUnusedLocal
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
    represents a point in n-dimensional space,
    and stores information about neighbours, boundary-properties etc.
    """

    def __init__(self, coords, idcs, grid, level=0,
                 node_class="main", simulation_mod=0):
        self.coords = coords
        self.idcs = tuple(idcs)
        self.axes = tuple(range(len(coords)))
        self.parents = list()
        self.grid = grid
        self.level = level
        self.func_val = None  # will be in {0, 1}
        self.boundary_flag = None
        # will be in {-1, 0, 1}  i.e. outer_bound, no bound or inner bound
        self.simulation_mod = simulation_mod
        self.sim_time = None
        self.sim_final_state = None
        # one of "main", "aux"
        self.node_class = node_class

        # counted positive in both directions
        # noinspection PyUnusedLocal
        self.distances = [list([None, None]) for i in range(len(coords))]
        # noinspection PyUnusedLocal
        self.idx_distances = [list([None, None]) for i in range(len(coords))]

        # list of cells to which this node belongs
        self.cells = []

        self.grid.ndb.add(self)

    def is_on_roi_boundary(self):
        """
        Determin if the node is on the boundary of the grid
        True: on the boundary of the grid
        """

        temp1 = self.grid.maxima
        temp2 = self.grid.minima
        # noinspection PyTypeChecker
        result = any([any(self.coords == temp1), any(self.coords == temp2)])
        return result

    def apply(self, func, simulation_mod):
        """
        :param func:                function which is to apply
        :param simulation_mod:      flag to determine whether simulation results should be stored inside node

        """
        if simulation_mod == 0:
            self.func_val = func(self.coords)
        else:
            temp = func(self.coords)
            self.func_val = temp[0]
            self.sim_time = temp[1]
            self.sim_final_state = temp[2]

        if self.func_val:
            self.grid.ndb.inner_nodes[self.grid.max_level].add(self)
        else:
            self.grid.ndb.outer_nodes[self.grid.max_level].add(self)

    def get_neighbors(self):
        """
        :return:
        """

        neighbors = []

        # iterate over all cells which contains this node
        for c in self.cells:

            # get all edges of this cell where this node is a part of
            edges = c.get_edge_nodes()

            for n1, n2 in edges:
                if self == n1:
                    neighbors.append(n2)
                elif self == n2:
                    neighbors.append(n1)
                else:
                    # self is not part of this edge
                    continue

        # remove duplicates:

        return list(dict.fromkeys(neighbors))

    def __repr__(self):

        return "<N f:{} {}|{}>".format(self.func_val, self.idcs, self.coords)


class Grid(object):
    def __init__(self, mg):

        # original meshgrid as returned by numpy.meshgrid
        self.mg = mg
        self.all_mg_points = np.array([arr.flat[:] for arr in self.mg])
        # to 1d array
        self.ndim = len(mg)  # 2d -> 2, 3d -> 3, ...

        # save all cells in a list
        self.cells = []

        # store all cells in a dict. keys are tuples like: (level, cell.vertex_nodes[0].idcs)
        self.cell_dict = {}

        # cells are organized in (child-) levels, beginning from 0
        # in this dict we store lists of the respective cells.
        # Keys are 0, 1, ...
        self.levels = collections.defaultdict(list)
        self.max_level = 0

        self.homogeneous_cells = collections.defaultdict(list)
        self.inhomogeneous_cells = collections.defaultdict(list)
        self.inner_cells = collections.defaultdict(list)
        self.outer_cells = collections.defaultdict(list)
        self.boundary_cells = collections.defaultdict(list)

        # these cells have homogeneous vertices but other points are inhomogeneous.
        # Thes cells require special treatment
        # They are stored w.r.t to their levels
        self.inhom_domain_cells = collections.defaultdict(set)

        self.roi_in_cells = collections.defaultdict(list)
        self.roi_outer_cells = collections.defaultdict(list)
        self.roi_boundary_cells = collections.defaultdict(list)
        self.ndb = NodeDataBase(grid=self)

        self.minima = np.array([np.min(arr) for arr in self.mg])
        self.maxima = np.array([np.max(arr) for arr in self.mg])

        # note: all arrays in mg have the same shape
        shapes = np.array(mg[0].shape)
        # depend on the  c of np.linspace(a,b,c)
        assert np.all(shapes > 1)
        assert len(shapes) == self.ndim  # 2d (a,b) 3d (a,b,c)

        self.coord_extention = self.maxima - self.minima
        self.coord_stepwidth = self.coord_extention/(shapes - 1)

        # numer of level0-cells in each direction
        self.base_resolutions = shapes - 1

        self.coords_of_index_origin = self.minima

        # these objects store index-coordinates,
        # which only depend on the dimension.
        # we just need to compute them once (see 2d examples)
        # they will be filled by methods below
        self.vertex_local_idcs = None
        # e.g.: [(0, 0), (0, 1), (1, 0), (1, 1)]
        self.edge_pair_local_idcs = None
        # e.g.: [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((0, 1), (1, 1)), ...]
        self.idx_edge_pairs = None
        # [(0, 1), (0, 2), (1, 3), ...] third edge consists of node1 and node3
        self.new_cell_idcs = None
        # e.g: [[(0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5)], [...]]

        self._find_vertex_local_idcs()
        self._find_edge_pairs()
        self._find_child_idcs()

        self.create_cells()

    def _find_vertex_local_idcs(self):
        """
        Create a list of index-tuples,
        which describe the indices of the nodes of the first cell. e.g. in 2d:
        [(0, 0), (0, 1), (1, 0), (1, 1)]

        :return:
        """
        self.vertex_local_idcs = list(itertools.product([0.0, 1.0],
                                      repeat=self.ndim))

    def _find_edge_pairs(self):
        """
        For a generic n-dim cell find all index pairs,
        which together define an edge of this cell (no diagonals).
        We use hamming distance here.
        (see docs of scipy.spatial.distance.hamming):
        An endge is a pair of nodes which
        differ in exactly one component

        :return:
        """

        assert self.edge_pair_local_idcs is None

        all_pairs = list(itertools.combinations(self.vertex_local_idcs, 2))

        def is_edge(pair):
            return hamming(pair[0], pair[1]) * self.ndim == 1

        edge_pairs = list(filter(is_edge, all_pairs))

        self.edge_pair_local_idcs = edge_pairs

        # we also might need the index-numbers of the nodes,
        # i.e. (0, 0) -> 0, (0, 1) -> 1, (1, 0) -> 2

        idx_nbr_dict = {}

        for idx_nbr, vertex in enumerate(self.vertex_local_idcs):
            idx_nbr_dict[vertex] = idx_nbr

        idx_edge_pairs = []
        for n1, n2 in edge_pairs:
            idx_edge_pairs.append((idx_nbr_dict[n1], idx_nbr_dict[n2]))

        self.idx_edge_pairs = idx_edge_pairs

    def _find_child_idcs(self):
        """
        For a generic n-dim cell find all 2**ndim subcells.
        A subcell here is a tuples of fractional indices,
        which describe the vertex nodes of the new smaller cell.

        Example in 2d case: Original nodes:  [(0, 0), (0, 1), (1, 0), (1, 1)]
        First subcell: [(0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5)]
        (remaining/reference vertex: (0, 0))

        :return:
        """

        assert self.new_cell_idcs is None

        self.new_cell_idcs = []

        step_array = np.array(self.vertex_local_idcs)*.5

        # the vertices of the first child cell
        # (lower left child cell in 2d) are the reference nodes
        # (lower left corner) of all child cells
        new_reference_nodes = tuple(step_array)

        for new_reference_node in new_reference_nodes:
            # make use of broadcasting here
            tmp = new_reference_node + step_array
            self.new_cell_idcs.append(tmp)

    def indices_to_coords(self, idcs):

        return self.coords_of_index_origin + \
               np.array(idcs)*self.coord_stepwidth

    def coords_to_indices(self, coords):
        """
        Return the (fractional) grid-indices of a point, given its coords.
        :param coords:
        :return:
        """

        return (coords - self.coords_of_index_origin)/self.coord_stepwidth

    def get_node_for_idcs(self, idcs, coords=None, level=0, allow_creation=True):
        """
        If `idcs` is a valid key of node_dict return the corresponding node,
        else create a new one.
        Only integer indices are possible here.
        coords are taken from the meshgrid.

        :param idcs:
        :param coords:
        :param level:
        :param allow_creation:  True (default) or false; determine whether creation of new node is allowed
        :return:
        """

        idcs = tuple(idcs)
        if idcs in self.ndb.node_dict:
            the_node = self.ndb.node_dict[idcs]
        else:
            if not allow_creation:
                msg = "no existing node for indices {}".format(idcs)
                raise KeyError(msg)

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
        lengths = self.mg[0].shape  # (n1 of xx,n2 of yy)

        index_sequences = []
        for L in lengths:
            assert L > 2
            index_sequences.append(range(0, L - 1))  # [range(0,8),range(0,8)]

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

    def classify_cells_by_homogenity(self, cells=None):
        """
        Iterate over the given cells (default: cells of the current max_level)
        and find those which are not homogenous

        :return: None

        :set:
            self.homogeneous_cells: inner cells + outer cells
            self.inhomogeneous_cells: boundary cells
        """

        if cells is None:
            cells = self.levels[self.max_level]

        for cell in cells:
            if cell.is_homogeneous():
                self.homogeneous_cells[self.max_level].append(cell)

                # set roa_boundary status
                cell.set_boundary_status(False)

                # now decide whether cell is inside or outside of roa-boundary
                if cell.grid_status():
                    self.inner_cells[self.max_level].append(cell)
                else:
                    self.outer_cells[self.max_level].append(cell)
            else:
                self.inhomogeneous_cells[self.max_level].append(cell)
                cell.set_boundary_status(True)
                self.boundary_cells[self.max_level].append(cell)

    def find_roi_cells(self):

        for i in range(self.max_level+1):
            for cell in self.inner_cells[i]:
                if cell.is_roi_boindary():
                    self.roi_in_cells[i].append(cell)
        for i in range(self.max_level+1):
            for cell in self.outer_cells[i]:
                if cell.is_roi_boindary():
                    self.roi_outer_cells[i].append(cell)
        for cell in self.boundary_cells[self.max_level]:
            if cell.is_roi_boindary():
                self.roi_boundary_cells[self.max_level].append(cell)

    def divide_boundary_cells(self):
        max_level = self.max_level

        assert self.boundary_cells == self.inhomogeneous_cells

        for cell in self.boundary_cells[max_level]:
            cell.make_childs()

    def sum_volumes(self):
        """
        return: inner-, outer-,and boundary volume of grid
        """

        sum_inner = 0
        sum_outer = 0
        sum_boundary = 0

        for i in range(self.max_level+1):
            if self.inner_cells[i] == []:
                continue
            for j in self.inner_cells[i]:
                sum_inner += j.get_volumes()

        for i in range(self.max_level+1):
            if self.outer_cells[i] == []:
                continue
            for j in self.outer_cells[i]:
                sum_outer += j.get_volumes()

        if len(self.boundary_cells[self.max_level]) == 0:
            sum_boundary = 0
        else:
            for i in self.boundary_cells[self.max_level]:
                sum_boundary += i.get_volumes()

        return sum_inner, sum_outer, sum_boundary

    def volume_fraction(self):
        """
        return: volume fraction of region of attraction
        """
        aa = self.coord_extention
        overall_volumes = 1
        for i in range(aa.shape[0]):
            overall_volumes *= aa[i]
        res = self.sum_volumes()[0]
        return res/overall_volumes

    def roi_volumes(self):
        """
        In addition to the interior of the grid,
        the boundaries of the grid are also
        divided into internal and external by the judgment function.
        return: inner,outer and boundary volumes on the boundary of the grid
        """
        sum_inner = 0
        sum_outer = 0
        sum_boundary = 0
        self.find_roi_cells()
        for i in range(self.max_level+1):
            if self.roi_in_cells[i] == []:
                continue
            for j in self.roi_in_cells[i]:
                sum_inner += j.get_roi_volumes()
        for i in range(self.max_level+1):
            if self.roi_outer_cells[i] == []:
                continue
            for j in self.roi_outer_cells[i]:
                sum_outer += j.get_roi_volumes()

        for i in self.roi_boundary_cells[self.max_level]:

            sum_boundary += i.get_roi_volumes()

        return sum_inner, sum_outer, sum_boundary

    def refinement_step(self, char_func, simulation_mod=0, handle_iDC=True):
        """
        :param char_func:
        :param simulation_mod:  flag to determine whether simulation results should be stored inside node
        :return:
        """

        # this will do nothing in the first call
        # (because we have no boundary cells yet)
        self.divide_boundary_cells()
        self.ndb.apply_func(char_func, simulation_mod)
        self.classify_cells_by_homogenity()
        flag = simulation_mod
        # now handle cases where outer boundary points are on the surface of an inner homogeneous cell (and vice versa)

        if handle_iDC:
            # it might be required to run the refinement process multiple times
            # because in every step new boundary points and cells are created
            while True:
                new_cells = self.refine_inhom_domain_cells(char_func, flag)
                if len(new_cells) == 0:
                    break

        # self.classify_cells_by_grid_status()
        return PointCollection(
               self.ndb.get_inner_nodes(),
               self.ndb.get_outer_nodes(),
               self.ndb.get_inner_boundary_nodes(self.max_level),
               self.ndb.get_outer_boundary_nodes(self.max_level))

    def refine_inhom_domain_cells(self, char_func, simulation_mod):

        self.find_inhom_domain_cells()

        new_cells = []
        for c in self.inhom_domain_cells[self.max_level]:
            for i in range(self.max_level+1):
                if c in self.inner_cells[i]:
                    self.inner_cells[i].remove(c)
                    self.boundary_cells
                if c in self.outer_cells[i]:
                    self.outer_cells[i].remove(c)

            if c.child_cells:
                # this cell was alredy handled in an previous step
                continue

            # now the cell c changes its status and thus must be removed from the
            # respetive cell list
            # TODO: add unittest (because this once was a bug)
            # Also test, if only the list ..._cells[self.max_level-1]
            # is affected

            new_cells.extend(c.make_childs())

        self.ndb.apply_func(char_func, simulation_mod)
        self.classify_cells_by_homogenity(new_cells)

        return new_cells

    def find_inhom_domain_cells(self):

        # now we look for inhomogeneous domain cells:
        # test boundary nodes if they lay on the border of a homogeneous cell

        ibn = list(self.ndb.inner_boundary_nodes[self.max_level])
        obn = list(self.ndb.outer_boundary_nodes[self.max_level])
        all_boundary_nodes = ibn + obn

        iDC = []

        for n in all_boundary_nodes:

            # map from func_value \in {False, True} to boundary_flag_value \in (-1, 1)
            problematic_flag_value = (1, -1)[int(n.func_val)]

            cells = self.get_cells_for_point(n.idcs)
            for c in cells:
                if c.roa_boundary_flag == problematic_flag_value:
                    iDC.append(c)

        self.inhom_domain_cells[self.max_level].update(iDC)

    def get_cells_for_point(self, idcs):
        """
        For an arbitrary point return the all cells of the highest level to which it belongs. Note that this
        can be multiple cells of different levels if the point is located on a cell boundary.

        :param idcs:    index-coordinates of the requested point
        :return:
        """

        idcs = np.array(idcs)

        res_cells = []

        # check that the point is inside roi:
        assert np.all(0 <= idcs)
        assert np.all(self.base_resolutions >= idcs)

        for level in range(self.max_level + 1):

            # index-coord-distance between nodes at the respective level
            delta = .5**level

            # boolean array which is True for every direction in which the point lays on a cell-boundary
            cell_boundary_axes = (idcs % delta) == 0

            # a point on a cell boundary can be part of multiple cells
            # we construct the pivot vertex of each of these candidate cells

            base_pivot_vertex = (idcs // delta)*delta

            # create a list like [[0], [0, -1], [0], [0, -1], [0, -1]] (5d case)
            # [0] for every axis which is not on the boundary
            # [0, -delta] for every axis which is on the boundary
            idx_diff_list = []
            for cba_flag in cell_boundary_axes:
                if cba_flag:
                    idx_diff_list.append([0, -delta])
                else:
                    idx_diff_list.append([0])

            # calc the cartesian product of that list (list of all possible combinations)
            test_pivot_vertices = base_pivot_vertex + np.array(list(itertools.product(*idx_diff_list)))

            # now search if there are cells, which belong to theses vertices
            for vertex_idcs in test_pivot_vertices:
                key = (level, tuple(vertex_idcs))
                cell = self.cell_dict.get(key)

                if cell is None:
                    continue

                if not cell.child_cells:
                    res_cells.append(cell)

        return res_cells

    def sum_cells(self, grid_type):
        """
        collect all cells in different levels as a list
        """
        sum_cells = []
        for i in range(self.max_level):
            sum_cells += grid_type[i]
        return sum_cells


class GridCell (object):

    def __init__(self, nodes, grid, level=0):
        self.ndim = len(nodes[0].coords)
        assert len(nodes) == 2**self.ndim
        self.grid = grid
        self.vertex_nodes = nodes
        self.level = level
        self.parent_cell = None
        self.child_cells = None
        self._is_homogeneous = None  # this flag considers only the vertices of the cell
        self._grid_status = None  # -1 -> out, 1 -> in

        self.roa_boundary_flag = None  # -1 -> out, 0 1 -> in

        # flag that specifies whether this cell is an "inhomogeneous domain cell"
        # (inhomogeneous but with homogeneous vertices)
        self._is_iDC = None

        # min- and max-boundary-coincide-lists: (min_bcl, max_bcl)
        # these lists encode whether a particular boundary-hypersurface (b-hsf)
        # of a grid cell is part of the boundary of the overall region of
        # investigation (roi).
        # each cell has 2n B-HSFs.
        # If a cell lays in the "minimal corner"
        # (minimal for each direction) of the grid,
        # its boundary is part of n ROI-boundary-HSFs.
        # Same holds for the "maximal corner".
        # The lists are initialized with None
        self.min_bcl = [None]*self.ndim
        self.max_bcl = [None]*self.ndim
        self.set_roi_boindary_status()

        self.grid.cells.append(self)

        self.key = (level, self.vertex_nodes[0].idcs)

        # this dict stores all cells in an easily accessible way (see construction of self.key)
        self.grid.cell_dict[self.key] = self
        self.grid.levels[self.level].append(self)

        if self.grid.max_level < self.level:
            assert self.grid.max_level == self.level - 1
            self.grid.max_level += 1

        # let the node know to which cells it belongs
        for n in nodes:
            if self not in n.cells:
                n.cells.append(self)

    def is_homogeneous(self):
        """
        return:
        1: all nodes in this cell have the same func_value,
        this cell is homogeneous
        0: inhomogeneous
        """

        cell_res = np.array([node.func_val for node in self.vertex_nodes])
        valbool_alltrue = np.alltrue(cell_res)
        valbool_allfalse = np.alltrue(np.logical_not(cell_res))
        self._is_homogeneous = valbool_alltrue or valbool_allfalse

        return self._is_homogeneous

    def is_roi_boindary(self):
        """
        The cell is on the boundary of the grid
        return:
        True: the cell is on the boundary of the grid
        """
        res = any([any(self.min_bcl), any(self.max_bcl)])
        return res

    def grid_status(self):

        """
        This function is only meaningful if the cell is homogeneous.
        Determine whether the homogeneous cell is inside the roa
        or outside the roa.
        return:
        1: inner
        0: outer
        """
        a = self.vertex_nodes[0].func_val
        self._grid_status = a
        return self._grid_status

    def make_childs(self):

        if self.child_cells is not None:
            msg = "The cell {} already has child cells!".format(self)
            raise ValueError(msg)

        new_level = self.level + 1
        new_cells = []
        for nci in self.grid.new_cell_idcs:

            index_diffs = nci*0.5**self.level
            nodes = []
            for index_diff in index_diffs:
                cell0_idcs = np.array(self.vertex_nodes[0].idcs)
                new_idcs = tuple(cell0_idcs + index_diff)
                node = self.grid.get_node_for_idcs(idcs=new_idcs,
                                                   level=self.level+1)
                # node = get_or_create_node(coords=None, idcs=new_idcs,
                # grid=self.grid)
                nodes.append(node)

            new_cell = GridCell(nodes, self.grid, level=new_level)
            new_cell.parent_cell = self

            new_cells.append(new_cell)
        self.child_cells = new_cells

        return new_cells

    def get_edge_nodes(self):
        """
        Return a list of 2-tuples of vertices ("corners").

        :return:
        """

        edge_node_tuples = []
        for n1, n2 in self.grid.idx_edge_pairs:
            edge_node_tuples.append((self.vertex_nodes[n1],
                                     self.vertex_nodes[n2]))

        return edge_node_tuples

    # TODO reuse `get_edge_nodes` here
    def get_edge_coords(self):
        """
        Return a list of 2-tuples of vertices ("corners").

        :return:
        """

        edges = []
        for n1, n2 in self.grid.idx_edge_pairs:
            edges.append((self.vertex_nodes[n1].coords,
                          self.vertex_nodes[n2].coords))

        return edges

    def get_vertex_coords(self):

        """
        Return the "corners" of the cell

        :return:
        """

        vertices = []
        for v in self.vertex_nodes:
            vertices.append(v.coords)

        return np.array(vertices)

    def set_boundary_status(self, roa_boundary_flag):
        """
        - Notify each node of this cell about its ROA-boundary status.
        - Determine self.roa_boundary_flag whether this cell is inside (1),
          outside (-1) or at the boundary of the ROA

        :param roa_boundary_flag: boolean flag.
                                  True: cell is part of the ROA-boundary,
                                  False: otherwise
        :return:
        """

        node_func_vals = []

        for node in self.vertex_nodes:
            func_val_int = int(node.func_val)
            node_func_vals.append(func_val_int)
            if roa_boundary_flag:
                max_l = self.grid.max_level
                assert func_val_int in (0, 1)
                if func_val_int == 0:
                    self.grid.ndb.outer_boundary_nodes[max_l].add(node)
                    node.boundary_flag = -1
                else:
                    self.grid.ndb.inner_boundary_nodes[max_l].add(node)
                    node.boundary_flag = 1

            elif node.boundary_flag is None:
                # only set to 0 if the node was not already flagged as boundary
                node.boundary_flag = 0

        node_func_vals = np.array(node_func_vals)
        all_in = np.all(node_func_vals == 1)
        all_out = np.all(node_func_vals == 0)

        if all_in:
            self.roa_boundary_flag = 1
        elif all_out:
            self.roa_boundary_flag = -1
        else:
            self.roa_boundary_flag = 0
        return self.roa_boundary_flag

    def set_roi_boindary_status(self):
        """
        Determine which of the cell boundary hyper surfaces (B-HSF)
        coincides with the B-HSFs of the region of investigation (ROI).
        This information encoded in two boolean sequences of length n:
        self.min_bcl and self.max_bcl

        See also: respective comment in self.__init__(...)

        :return: None
        """

        # Explanation:
        # self.get_vertex_coords() -> (2**n, n)-array of coordinates of all
        # vertices of this cell
        # each row is a vertex
        # if a vertex has a value in common with self.grid.minima,
        # then this cell shares the respective part of the roi-boundary
        # (make a sketch)
        # with np.any(..., axis=0) we get this information separately
        # for each direction (axis=0 -> )
        vertex_coords = self.get_vertex_coords()
        self.min_bcl = np.any(vertex_coords == self.grid.minima, axis=0)
        self.max_bcl = np.any(vertex_coords == self.grid.maxima, axis=0)

    def get_volumes(self):

        """
        return: volume of each cell in different levels
        """

        grid_parameter = self.grid.coord_stepwidth
        v_0 = 1
        for i in range(grid_parameter.shape[0]):
            v_0 *= grid_parameter[i]
        vc = v_0 / (2**(self.level*self.ndim))

        return vc

    def get_roi_volumes(self):

        """
        In  Class Grid, we find the cells located on the boundary of the grid.
        This function at first find out which part of these cells
        is on the boundary of the grid,
        and then calculate the volumes.
        return: The volume of each cell at the boundary of the grid
        """

        r = self.grid.coord_extention / self.grid.coord_stepwidth
        # Resolution of each axis.e.g. as 3D,
        # x1:r1; x2:r2; x3:r3, r=[r1,r2,r3]
        s = self.grid.coord_extention
        # length of each axis.eg x1:a; x2:b;x3:c,s =[a,b,c]
        temp1 = np.prod(s) * r * 1/s
        # abc * [r1/a,r2/b,r3/c] --> [r1bc,r2ac,r3ab]
        temp2 = 2**(self.level*(self.ndim-1))
        # cells in different levesl
        re1 = np.dot(temp1, self.max_bcl) / np.prod(r) / temp2
        # e.g.: np.dot([r1bc,r2ac,r3ab],[True,True,False]) / (r1r2r3)
        # = bc/r2r3 +ac/r1r3
        re2 = np.dot(temp1, self.min_bcl) / np.prod(r) / temp2
        # self.max_bcl and self.min_bcl
        re = re1 + re2
        return re

    def __repr__(self):
        """
        :return: string-representation of this cell

        rn:  -> reference-node (index-coordinates)
        rnc: -> reference-node (cartesian-coordinates)
        """
        return "<Cell: bf: {}, level:{}, rn:{}, rnc:{}>".format(
            self.roa_boundary_flag,
            self.level,
            self.vertex_nodes[0].idcs,
            self.vertex_nodes[0].coords)


class object_wothout_simulation(object):
    """
    all data is saved in an object, then this
    object will be used in matplotlib.
    """
    def __init__(self, grid, max_level_refinement, func, pc):
        """
        grid: the grid after refinement, which contains cells of different
        types;
        max_level_refinement: number of the grid refinements;
        func: judge function, which can also be regarded as the target form of
        the gitterapproximation;
        pc: punkts collect. this collect concentrate on the punkts in the
        boundary cells.
        """
        self.grid = grid
        self.max_level_refinement = max_level_refinement
        self.func = func
        self.pc = pc


# noinspection PyPep8Naming
class ROA_Approximation(object):
    """
    Author Yuewen He
    arg:
    mg: meshgrid;
    mod: model of the system, which has right-hand-side equation;
    para: parameter of the system; mod_name;
    xx_res: Target balance point
    refinement: default means no refinement,
    otherweise start the refinement.
    """

    def __init__(self, mg, mod, Pole,
                 max_level_refinement, para,
                 xx_res=None, refinement=None):
        self.Pole = Pole
        self.mod = mod
        self.max_level_refinement = max_level_refinement
        self.mg = mg
        self.grid = Grid(mg)
        self.Pole = Pole
        self.para = para
        # parameters of the system
        if xx_res is None:
            self.xx_res = np.zeros(len(mod.xx))
        else:
            self.xx_res = xx_res
        self.K = self.calculate_K()
        self.rhs = self.rhs_bilden()
        if refinement is None:
            print("waiting for the refinement...")
        else:
            self.data = self.get_volume_fraction()
            self.fraction = self.data[1]
            self.nodes = self.data[0].ndb.all_nodes
            self.result = self.data[5]
            self.roi_volumes = self.data[6]
            self.roa_volumes = self.data[7]

    def calculate_K(self):
        """
        calculate the feedback matrix K
        """
        f = self.mod.f.subs(self.para)
        G = self.mod.g.subs(self.para)
        xx = self.mod.x
        replmts0 = list(zip(xx, [0]*len(xx)))
        A = f.jacobian(xx).subs(replmts0)
        B = G.subs(replmts0)
        K = st.siso_place(A, -B, self.Pole).T
        return K

    def calculate_u(self, xx, t):
        """
        generate the input funktion
        arg:
        xx: state vector
        t: simulation time
        """
        xx = np.array([xx])
        # xx_res = np.array([[0, 0]])
        lim = 2
        u = self.K @ (self.xx_res - xx).T
        # add the limit for the input
        if np.abs(u) > lim:
            return np.sign(u)*lim
        else:
            return u

    def rhs_bilden(self):
        """
        bild right-hand-side funktion
        """
        sm = st.SimulationModel(self.mod.f, self.mod.g,
                                self.mod.xx,
                                model_parameters=self.para)
        rhs = sm.create_simfunction(controller_function=self.calculate_u)
        return rhs

    def rhs2(self, t, xx):
        """
        rhs funktion deformation,
        is to prepare for the following integral solver
        """
        return self.rhs(xx, t)

    def judge(self, xx):
        """
        Funktion: ein Anfangswertpunkt bestimmen,
        ob der Punkt in der Einzugsbereich liegt
        arg: xx state vector
        returns:
        flag = [Value]
        Value: 0: the punkt is unstable
               1: the punkt is stable
        sol.t_events: time points when the events happen
        sol.y: result of the intergrate or the state vector
        """
        sim_steps = 200
        sim_time = 20
        tt = st.np.linspace(0, sim_time, sim_steps)
        y0 = np.asarray([*xx])

        # adapt args to be able to apply scipy.integrate.solve_ivo

        def event1(t, y):
            return np.sum(y**2) > 0.0001

        def event2(t, y):
            return np.sum(y**2) < 10000

        event1.terminal = True
        event2.terminal = True
        # noinspection PyTypeChecker
        sol = sc_integrate.solve_ivp(self.rhs2, [tt[0], tt[-1]],
                                     y0, method='RK45', t_eval=tt,
                                     events=(event1, event2),
                                     dense_output=False)

        # noinspection PyUnresolvedReferences
        if sol.t_events[0].size != 0:
            flag = 1
        else:
            flag = 0

        return flag, sol.t_events, sol.y

    def get_volume_fraction(self):
        """
        return:
        grid
        vf: volume fraction (value between 0 and 1)
        list of cells in different type. This is for matplotlib
        """

        grid = self.grid
        result = np.zeros((self.max_level_refinement, 3))
        for i in range(self.max_level_refinement):
            grid.refinement_step(self.judge, 1)
            aa = grid.sum_volumes()
            result[i, :] = aa
        # self.draw_process(result)
        roi_volumes = grid.roi_volumes()
        res_roi = roi_volumes[0]/sum(roi_volumes)
        res_roa = grid.volume_fraction()
        roa_volumes = grid.sum_volumes()
        return \
            grid, np.array([res_roa, res_roi]), \
            roi_volumes, roa_volumes

    def draw_process(self):
        """
        Record the changes of inner volume, outer volume and boundary volume
        after each refinement
        """
        x = np.arange(1, self.max_level_refinement+1, 1)
        plt.plot(x, self.result[:, 0], 'r--', label='inner')
        plt.plot(x, self.result[:, 1], 'g--', label='outer')
        plt.plot(x, self.result[:, 2], 'b--', label='boundary')
        plt.plot(x, self.result[:, 0], 'ro-',
                 x, self.result[:, 1], 'g+-',
                 x, self.result[:, 2], 'b^-')
        plt.title('process')
        plt.xlabel('Menge von refinement')
        plt.ylabel('Volumes')
        plt.legend()
        plt.show()
        plt.savefig("process")


class ROA_Approximation_judge(object):
    """
    Author Yuewen He
    arg:
    mg: meshgrid; mod: model of the system, which has right-hand-side equation,
    para: parameter of the system; mod_name, number: used to save the picture;
    xx_res: Target balance point
    refinement: default means no simulation, otherweise start the simulation
    judge: direct use judge as judge-function to refine the grid
    """

    def __init__(self, mg, judge, Pole,
                 max_level_refinement, mod_name, number,
                 refinement=None, show_process=None):
        """
        Pole: pole placement as methode to design the controller,
        here Pole is the target pole of the closed loop system.
        """
        self.Pole = Pole
        self.judge = judge
        # level of the refinement
        self.max_level_refinement = max_level_refinement
        self.mg = mg
        self.grid = Grid(mg)
        self.number = number
        self.mod_name = mod_name
        # parameters of the system
        if refinement is None:
            print("waiting for the refinement...")
        else:
            self.data = self.get_data()
            self.fraction = self.data[1]
            self.nodes = self.data[0].ndb.all_nodes
            self.result = self.data[5]
            self.roi_volumes = self.data[6]
            self.roa_volumes = self.data[7]
        if show_process is not None:
            self.draw_process()

    def get_data(self):
        """
        the grid will be refined, then the volumes of different types will also
        be calculated. related data is saved in a list.
        return:
        grid, volume fraction (value between 0 and 1),
        list of cells in different type. This is for matplotlib
        """
        grid = self.grid
        result = np.zeros((self.max_level_refinement, 3))
        for i in range(self.max_level_refinement):
            grid.refinement_step(self.judge, 1)
            aa = grid.sum_volumes()
            result[i, :] = aa
        # self.draw_process(result)
        roi_volumes = grid.roi_volumes()
        res_roi = roi_volumes[0]/sum(roi_volumes)
        res_roa = grid.volume_fraction()
        roa_volumes = grid.sum_volumes()
        return \
            grid, np.array([res_roa, res_roi]), \
            roi_volumes, roa_volumes

    def draw_process(self):
        """
        Record the changes of inner volume, outer volume and boundary volume
        after each refinement
        """
        x = np.arange(1, self.max_level_refinement+1, 1)
        plt.plot(x, self.result[:, 0], 'r--', label='volume of inside cells')
        plt.plot(x, self.result[:, 1], 'g--', label='voleme of outside cells')
        plt.plot(x, self.result[:, 2], 'b--', label='volume of boundary cells')
        plt.plot(x, self.result[:, 0], 'ro-',
                 x, self.result[:, 1], 'g+-',
                 x, self.result[:, 2], 'b^-')
        plt.title('process')
        plt.xlabel('number of refinement')
        plt.ylabel('Volumes')
        plt.legend()
        plt.show()
        # plt.savefig("process")


def get_index_difference(idcs1, idcs2):

    """
    Assume that the index tuples differ by exactly one index.
    Find out which dimension-index that is and the difference
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
        msg = "No difference between {} and {} was found which \
               is against the assumption".format(idcs1, idcs2)
        raise ValueError(msg)

    dir = int(diff > 0)

    return dim, dir, diff


def func_circle(xx):
    return xx[0]**2 + xx[1]**2 < 8


def func_sphere_nd_factory(radius):
    """
    :return: characteristic function of n-dimensional sphere for given radius
    """

    level = radius ** 2

    def sphere_func(xx):
        """
        Characteristic function of n-dimensional sphere

        :param xx:
        :return: True or False
        """
        return np.sum(xx**2) < level

    return sphere_func


def func_ellipse_2d_factory(radius, M_matrix, Rotation_angle):
    """
    :return: characteristic function of an ellipse for given radius, matirx
    """
    level = radius ** 2
    phi = Rotation_angle
    # rotation matrix
    R = np.array([[np.cos(phi), np.sin(phi)],
                 [-np.sin(phi), np.cos(phi)]])
    Q = R.T @ M_matrix @ R

    def ellipse_func(xx):
        """
        Characteristic function of 2-dimensional ellipse
        Use the ellipse equation as the judgment function
        Return：
        1 or True:the point is inside the ellipse;
        0 or False:the point is outside the ellipse
        :param xx:
        :return: True or False
        """
        return xx.T @ Q @ xx < level

    return ellipse_func


def func_bump(x, offset=0, amplitude=1, exponent=2):
    """
    Function that looks like: __/\__ and which can be scaled.

    :param x:
    :param offset:
    :param amplitude:
    :param exponent:
    :return:
    """
    return offset + amplitude/(1 + x**exponent)


def draw_cells(grid):
    max_level_refinement = grid.max_level + 1
    # collect all cells of different types
    all_inner_cells = sum_cells(grid.inner_cells, max_level_refinement)
    all_outer_cells = sum_cells(grid.outer_cells, max_level_refinement)
    boundary_cells_maxlevel = grid.boundary_cells[ max_level_refinement - 1]

    ax = plt.gca()

    for cell in (all_inner_cells):
        edges = np.array(cell.get_edge_coords())
        plt.plot(*edges.T, 'C2--')
        # vertex0: original punkt of the cell
        vertex0 = cell.get_vertex_coords()[0]
        # lw: taking vertex0 as the origin, the extension of the cell
        # in each dimension
        lw = grid.coord_stepwidth / (2 ** (cell.level))
        rect1 = mp.Rectangle(vertex0, lw[0], lw[1], facecolor='C2', alpha=0.3)
        ax.add_patch(rect1)

    for cell in (all_outer_cells):
        edges = np.array(cell.get_edge_coords())
        plt.plot(*edges.T, 'C1--')
        vertex0 = cell.get_vertex_coords()[0]
        lw = grid.coord_stepwidth / (2 ** (cell.level))
        rect2 = mp.Rectangle(vertex0, lw[0], lw[1], facecolor='C1', alpha=0.3)
        ax.add_patch(rect2)

    for cell in (boundary_cells_maxlevel):
        edges = np.array(cell.get_edge_coords())
        plt.plot(*edges.T, 'C0-')
        vertex0 = cell.get_vertex_coords()[0]
        lw = grid.coord_stepwidth / (2 ** (cell.level))
        rect3 = mp.Rectangle(vertex0, lw[0], lw[1], facecolor='C0', alpha=0.3)
        ax.add_patch(rect3)

def sum_cells(grid_type, max_level_refinement):
    """
    collect all cells of one type in a list
    """
    sum_cells = []
    for i in range(max_level_refinement):
        sum_cells += grid_type[i]
    return sum_cells

def grid_eval_2d_func(func, x_min_max=None, y_min_max=None, resolution=None, ax=None):
    """
    Auxiliary function for visualizing judge-functions during algorithm developnment
    This function creates a (2, N1, N2)-array (meshgrid) and applies func(y)
    for every point in the (N1, N2)-matrix of 2d-arrays.
    This result (ZZ) can than be plotted with ax.contour

    Example call:

        XX, YY, ZZ = grid_eval_2d_func(ellipse)
        plt.contour(XX, YY, ZZ)

    :param func:   callable func(arg), where arg is exected to be a (2,)-array
    :param x_min_max:       None or 2-tuple of default x-limits
    :param y_min_max:       None or 2-tuple of default y-limits
    :param resolution:      None or 2-tuple of default resolution
    :param ax:              axis object (currently not used)

    :returns:  XX, YY, ZZ
    """
    if resolution is None:
        N1, N2 = 300, 300
    else:
        N1, N2 = resolution

    if x_min_max is None:
        x_min_max = (-12, 12)

    if y_min_max is None:
        y_min_max = (-8, 8)

    xx = np.linspace(*x_min_max, N1)
    yy = np.linspace(*y_min_max, N2)
    XX, YY = mg = np.meshgrid(xx, yy, indexing="ij")
    mg_array = np.array(mg)

    # reshaping and transposing
    mg_array2 = mg_array.reshape(2, -1).T

    # feed every point into func and reshape the overall result
    ZZ = np.array([func(row) for row in mg_array2]).reshape(mg_array.shape[1:])

    return XX, YY, ZZ

