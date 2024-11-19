# -*- coding: utf-8 -*-
"""
Created on 2019-12-12 00:14:14

@author: Carsten Knoll
"""

import numpy as np
import matplotlib.pyplot as plt

# noinspection PyUnresolvedReferences
import mpl_toolkits.mplot3d as a3

# noinspection PyUnresolvedReferences
from symbtools import meshtools as met
import ipydex as ipd

import unittest


exec_show = False


class TestGrid2d(unittest.TestCase):
    def setUp(self):
        xx = np.linspace(-4, 4, 9)
        yy = np.linspace(-4, 4, 9)

        mg = np.meshgrid(xx, yy, indexing="ij")

        self.mg = mg

    def test_create_cell(self):
        # met.create_nodes_from_mg(self.mg)
        grid = met.Grid(self.mg)

        l1 = len(grid.cells)
        self.assertEqual(len(grid.ndb.levels[0]), 81)

        self.assertEqual(grid.idx_edge_pairs, [(0, 1), (0, 2), (1, 3), (2, 3)])

        childs1 = grid.cells[0].make_childs()

        self.assertEqual(len(childs1), 4)
        expected_vertices = np.array([[-4., -4.], [-4., -3.5], [-3.5, -4.],  [-3.5, -3.5]])
        self.assertTrue(np.all(childs1[0].get_vertex_coords() == expected_vertices))

        # five nodes had to be inserted
        self.assertEqual(len(grid.ndb.levels[1]), 5)

        childs2 = childs1[0].make_childs()
        self.assertEqual(len(grid.ndb.levels[2]), 5)

        self.assertEqual(childs1[0].child_cells, childs2)
        self.assertEqual(childs2[0].parent_cell, childs1[0])

        self.assertEqual(len(grid.levels[0]), l1)
        self.assertEqual(len(grid.levels[1]), len(childs1))
        self.assertEqual(len(grid.levels[2]), len(childs2))

        if None:
            plt.plot(*grid.all_mg_points, '.')
            # plot_cells2d([gc]+childs1+childs2, show=True)

    def test_make_childs_bug(self):
        """
        ensure that a certain bug remains fixed
        :return:
        """

        grid = met.Grid(self.mg)

        childs1 = grid.cells[0].make_childs()

        diff_arr = np.array(grid.vertex_local_idcs)*0.5

        # ensure that the lower left is the first vertex (this is needed as reference for the next child generation)
        for c in childs1:
            node_coords = np.array([vn.coords for vn in c.vertex_nodes])

            reference_node = node_coords[0, :]

            # the other nodes must be reachable from the reference node via the vectors in diff_arr
            node_coords2 = reference_node + diff_arr
            self.assertTrue(np.allclose(node_coords, node_coords2))

    def test_node_neighbours(self):

        grid = met.Grid(self.mg)

        # "inner corner" of first cell of the grid
        # it should have four neighbours
        cell = grid.levels[0][0]
        n1 = cell.vertex_nodes[-1]

        expected_neighbors = [grid.get_node_for_idcs((0, 1)),
                              grid.get_node_for_idcs((1, 0)),
                              grid.get_node_for_idcs((1, 2)),
                              grid.get_node_for_idcs((2, 1)),
                              ]

        self.assertEqual(n1.get_neighbors(), expected_neighbors)

        # now choose a corner of the region of investigation (ROI)
        # it should have only 2 neighbours
        n2 = cell.vertex_nodes[0]

        expected_neighbors = [grid.get_node_for_idcs((0, 1)),
                              grid.get_node_for_idcs((1, 0)),
                              ]

        self.assertEqual(n2.get_neighbors(), expected_neighbors)

    def test_get_cells_for_point(self):

        grid = met.Grid(self.mg)

        # "inner corner" of first cell of the grid
        # it should have four neighbours
        cell = grid.levels[0][0]
        n1 = cell.vertex_nodes[-1]

        # test for a corner of a cell

        p = n1.idcs
        cells = grid.get_cells_for_point(p)
        self.assertEqual(len(cells), 4)
        self.assertEqual(set(cells), set(n1.cells))

        # test for point on edge
        p = np.array([0.5, 1])
        cells = grid.get_cells_for_point(p)
        self.assertEqual(len(cells), 2)
        expected_cells = [grid.cell_dict[0, (0, 0)], grid.cell_dict[0, (0, 1)]]
        self.assertEqual(set(cells), set(expected_cells))

        # test for corner of roi  -> expect only one cell
        p = grid.base_resolutions
        cells = grid.get_cells_for_point(p)
        self.assertEqual(len(cells), 1)
        expected_cells = [grid.levels[0][-1]]
        self.assertEqual(set(cells), set(expected_cells))

        # test for point on edge of cell and edge of roi  -> expect only one cell
        p = grid.base_resolutions - np.array((.5, 0))
        cells = grid.get_cells_for_point(p)
        self.assertEqual(len(cells), 1)
        expected_cells = [grid.levels[0][-1]]
        self.assertEqual(set(cells), set(expected_cells))

        # --------------------------------------------------------------
        # now test that it works with refined cells also
        # see n2_level2.png (created by other test case) for orientation

        func_sphere_nd = met.func_sphere_nd_factory(radius=1.3)
        grid.refinement_step(func_sphere_nd)  # level 0
        grid.refinement_step(func_sphere_nd)  # level 1
        grid.refinement_step(func_sphere_nd)  # level 2
        self.assertEqual(grid.max_level, 2)

        # point on the common edge of a level-2-cell and level 0 cell
        p = grid.coords_to_indices((-1, -1.1))
        cells = grid.get_cells_for_point(p)
        self.assertEqual(len(cells), 2)
        expected_cells = [grid.cell_dict[0, (2, 2)], grid.cell_dict[2, (3, 2.75)]]
        self.assertEqual(set(cells), set(expected_cells))

        # point on the common edge of a level-1-cell and level 0 cell
        p = grid.coords_to_indices((-1, -1.8))
        cells = grid.get_cells_for_point(p)
        self.assertEqual(len(cells), 2)
        expected_cells = [grid.cell_dict[0, (2, 2)], grid.cell_dict[1, (3, 2)]]
        self.assertEqual(set(cells), set(expected_cells))

        # point in the interior of a level-1-cell
        p = grid.coords_to_indices((0.7, -1.8))
        cells = grid.get_cells_for_point(p)
        self.assertEqual(len(cells), 1)
        expected_cells = [grid.cell_dict[1, (4.5, 2)]]
        self.assertEqual(set(cells), set(expected_cells))

    def _test_plot(self):
        # create images where each new cell is shown
        grid = met.Grid(self.mg)

        plt.plot(*grid.all_mg_points, '.')
        plt.savefig("tmp_0.png")
        for i, cell in enumerate(grid.cells):
            edges = np.array(cell.get_edge_coords())
            plt.plot(*edges.T)
            plt.savefig("tmp_{:03d}.png".format(i))

        ipd.IPS()


class TestGrid3d(unittest.TestCase):

    def setUp(self):
        xx = np.linspace(-4, 4, 9)
        yy = np.linspace(-4, 4, 9)
        zz = np.linspace(-4, 4, 9)

        mg = np.meshgrid(xx, yy, zz, indexing="ij")

        self.mg = mg

    def test_create_cells(self):
        grid = met.Grid(self.mg)

        self.assertEqual(list(grid.cells[0].vertex_nodes[0].coords), [-4.0, -4.0, -4.0])
        self.assertEqual(list(grid.cells[0].vertex_nodes[3].coords), [-4.0, -3.0, -3.0])

    def test_node_neighbours(self):

        grid = met.Grid(self.mg)

        # "inner corner" of first cell of the grid
        # it should have four neighbours
        cell = grid.levels[0][0]
        n1 = cell.vertex_nodes[-1]

        expected_neighbors = [grid.get_node_for_idcs((0, 1, 1)),
                              grid.get_node_for_idcs((1, 0, 1)),
                              grid.get_node_for_idcs((1, 1, 0)),
                              grid.get_node_for_idcs((1, 1, 2)),
                              grid.get_node_for_idcs((1, 2, 1)),
                              grid.get_node_for_idcs((2, 1, 1)),
                              ]

        self.assertEqual(n1.get_neighbors(), expected_neighbors)

        # now choose a corner of the region of investigation (ROI)
        # it should have only 2 neighbours
        n2 = cell.vertex_nodes[0]

        expected_neighbors = [grid.get_node_for_idcs((0, 0, 1)),
                              grid.get_node_for_idcs((0, 1, 0)),
                              grid.get_node_for_idcs((1, 0, 0)),
                              ]

        self.assertEqual(n2.get_neighbors(), expected_neighbors)

    def test_plot(self):
        # create images where each new cell is shown
        grid = met.Grid(self.mg)
        l1 = len(grid.cells)

        childs1 = grid.cells[0].make_childs()

        self.assertEqual(len(grid.ndb.levels[1]), 19)

        l2 = len(grid.cells)
        self.assertEqual(len(childs1), 8)
        self.assertEqual(l2, l1 + len(childs1))

        expected_vertices = np.array([[-4.0, -4.0, -4.0],
                                      [-4.0, -4.0, -3.5],
                                      [-4.0, -3.5, -4.0],
                                      [-4.0, -3.5, -3.5],
                                      [-3.5, -4.0, -4.0],
                                      [-3.5, -4.0, -3.5],
                                      [-3.5, -3.5, -4.0],
                                      [-3.5, -3.5, -3.5]])
        self.assertTrue(np.all(childs1[0].get_vertex_coords() == expected_vertices))

        childs2 = childs1[0].make_childs()
        self.assertEqual(len(grid.ndb.levels[2]), 19)

        self.assertEqual(childs1[0].child_cells, childs2)
        self.assertEqual(childs2[0].parent_cell, childs1[0])

        self.assertEqual(len(grid.cells), l1 + len(childs1) + len(childs2))

        if None:
            plot_cells = grid.cells[:1] + [grid.cells[-16]] + grid.cells[-8:]
            plot_cells3d(plot_cells, imax=None, show=True, all_points=grid.all_mg_points)

    def test_get_cells_for_point(self):

        grid = met.Grid(self.mg)

        # "inner corner" of first cell of the grid
        # it should have 8 neighbours
        cell = grid.levels[0][0]
        n1 = cell.vertex_nodes[-1]

        # test for a corner of a cell
        p1 = n1.idcs
        cells = grid.get_cells_for_point(p1)
        self.assertEqual(len(cells), 8)
        self.assertEqual(set(cells), set(n1.cells))

        # test for point on edge
        p1 = np.array([1, 1, 0.5])
        cells = grid.get_cells_for_point(p1)
        self.assertEqual(len(cells), 4)
        expected_cells = [grid.cell_dict[0, (0, 0, 0)],
                          grid.cell_dict[0, (0, 1, 0)],
                          grid.cell_dict[0, (1, 0, 0)],
                          grid.cell_dict[0, (1, 1, 0)],
                          ]
        self.assertEqual(set(cells), set(expected_cells))

        # test for point on surface
        p1 = np.array([0.5, 1, 0.5])
        cells = grid.get_cells_for_point(p1)
        self.assertEqual(len(cells), 2)
        expected_cells = [grid.cell_dict[0, (0, 0, 0)],
                          grid.cell_dict[0, (0, 1, 0)],
                          ]
        self.assertEqual(set(cells), set(expected_cells))

        # --------------------------------------------------------------
        # now test that it works with refined cells also
        # see n2_level2.png (created by other test case) for orientation

        func_sphere_nd = met.func_sphere_nd_factory(radius=1.3)
        pc0 = grid.refinement_step(func_sphere_nd)  # level 0
        pc1 = grid.refinement_step(func_sphere_nd)  # level 1
        # pc2 = grid.refinement_step(func_sphere_nd)  # level 2

        c = grid.boundary_cells[1][0]
        p1 = c.vertex_nodes[0].idcs
        cells = grid.get_cells_for_point(p1)

        p2 = [2.2, 3, 3.1]
        p3 = [2.2, 2.5, 2.8]

        debug_plot = False
        if debug_plot:

            ax = plt.figure(figsize=(15, 15)).add_subplot(1, 1, 1, projection='3d')

            ax.plot(*grid.all_mg_points, '.', ms=1, color="k", alpha=0.5)

            ax.plot(*pc0.ibp, 'o', ms=3, color="m", alpha=1)
            ax.plot(*pc1.ibp, 'o', ms=3, color="r", alpha=1)
            plot_cells3d([c], ax=ax)

            # plot points p1, p2, p3
            ax.plot(*np.atleast_2d(grid.indices_to_coords(p1)).T, "o", ms=5, color="c")
            ax.plot(*np.atleast_2d(grid.indices_to_coords(p2)).T, "o", ms=5, color="m")
            ax.plot(*np.atleast_2d(grid.indices_to_coords(p3)).T, "o", ms=5, color="k")

            # all_level1_cells = grid.levels[1]
            # plot_cells3d(all_level1_cells, ax=ax, alpha=0.1)
            plot_cells3d(cells, ax=ax, alpha=0.5)

            plt.show()

        # this was manually confirmed by looking at the 3d visualization
        expected_cells = [grid.cell_dict[0, (2, 3, 2)],
                          grid.cell_dict[0, (2, 2, 3)],
                          grid.cell_dict[0, (2, 2, 2)],
                          grid.cell_dict[1, (2.5, 3, 3)],
                          grid.cell_dict[1, (2, 3, 3)],
                          ]

        self.assertEqual(len(cells), 5)
        self.assertEqual(set(cells), set(expected_cells))

        # p2 (see above)
        cells = grid.get_cells_for_point(p2)
        self.assertEqual(len(cells), 2)
        expected_cells = [grid.cell_dict[0, (2, 2, 3)],
                          grid.cell_dict[1, (2, 3, 3)],
                          ]
        self.assertEqual(set(cells), set(expected_cells))

        # p3 (see above)
        cells = grid.get_cells_for_point(p3)
        self.assertEqual(len(cells), 1)
        expected_cells = [grid.cell_dict[0, (2, 2, 2)],
                          ]
        self.assertEqual(set(cells), set(expected_cells))


class MeshRefinement2d(unittest.TestCase):

    def setUp(self):
        xx = np.linspace(-4, 4, 9)
        yy = np.linspace(-4, 4, 9)
        yy2 = np.linspace(-4, 4, 7)

        self.mg = np.meshgrid(xx, yy, indexing="ij")
        self.mg2 = np.meshgrid(xx, yy2, indexing="ij")  # more columns than rows (n1 != n2 helps to test)

    def test_refinement(self):

        plot_points = False

        grid = met.Grid(self.mg)

        func_sphere_nd = met.func_sphere_nd_factory(radius=1.3)
        pc = grid.refinement_step(func_sphere_nd)

        ic0 = grid.inhomogeneous_cells[0]
        ac0 = grid.levels[0]
        plot_cells2d(ac0, color="0.7", alpha=0.3)
        plot_cells2d(ic0, color="0.7", alpha=0.3)
        plt.gca().add_patch(plt.Circle((0, 0), 1.3 ** .5, alpha=0.5))

        # there are 12 inhomogeneous cells (manually verified)
        self.assertEqual(len(ic0), 12)

        self.assertEqual(pc.ibp.shape, (2, 5))
        self.assertEqual(pc.obp.shape, (2, 16))

        if plot_points:
            # plot inner and outer points (level 0)
            plt.plot(*pc.op, "bo", alpha=0.2, ms=5)
            plt.plot(*pc.ip, "ro", alpha=0.2, ms=5)

            plt.plot(*pc.obp, "bo", ms=3)
            plt.plot(*pc.ibp, "ro", ms=3)

        plt.title("levels 0")
        plt.axis("square")
        plt.savefig("n2_level0.png")

        # -----

        self.assertEqual(grid.max_level, 0)
        pc = grid.refinement_step(func_sphere_nd)
        self.assertEqual(grid.max_level, 1)

        self.assertEqual(pc.ibp.shape, (2, 16))
        self.assertEqual(pc.obp.shape, (2, 24))

        plt.cla()

        if plot_points:
            # plot inner and outer points (level 1)
            plt.plot(*pc.op, "go", alpha=0.2, ms=5)
            plt.plot(*pc.ip, "mo", alpha=0.2, ms=5)

            plt.plot(*pc.obp, "go", ms=3)
            plt.plot(*pc.ibp, "mo", ms=3)

        grid.refinement_step(func_sphere_nd)

        ac0 = grid.levels[0]
        ic0 = grid.inhomogeneous_cells[0]
        ic1 = grid.inhomogeneous_cells[1]
        ac1 = grid.levels[1]
        ac2 = grid.levels[2]
        ic2 = grid.inhomogeneous_cells[2]

        plot_cells2d(ac0, color="0.7", alpha=0.3)
        plot_cells2d(ac1, color="0.7", alpha=0.5)
        plot_cells2d(ac2, color="0.7", alpha=0.8)
        plot_cells2d(ic0, color=(.5, 0, 0))
        plot_cells2d(ic1, color=(.7, 0, 0))
        plot_cells2d(ic2, color=(.9, 0, 0))

        plt.gca().add_patch(plt.Circle((0, 0), 1.3**.5))

        plt.title("levels 0, 1, 2")
        plt.savefig("n2_level2.png")

        if exec_show:
            plt.show()

        # -----

        pc = grid.refinement_step(func_sphere_nd)
        # plot inner and outer points (level 2)
        plt.cla()
        plt.axis("square")
        plt.axis([-1.8, 1.8, -1.8, 1.8])
        plt.gca().add_patch(plt.Circle((0, 0), 1.3**.5, alpha=0.5))

        #
        # plt.plot(*pc.op, "go", alpha=0.2, ms=5)
        # plt.plot(*pc.ip, "mo", alpha=0.2, ms=5)

        plt.plot(*pc.obp, "go", ms=3)
        plt.plot(*pc.ibp, "mo", ms=3)

        plt.title("levels 0, 1, 2")
        plt.savefig("n2_level2_points.png")
        if exec_show:
            plt.show()

    def test_refinement_iDC(self):
        """
        test refinement of inhomogeneous domain cells (iDCs).

        # This was a bug before 2020-05-12
        """

        def nonconvex_jf(xx):
            return xx[1] < met.func_bump(xx[0], offset=1.2, amplitude=-6)

        xx = np.linspace(-10, 10, 4)
        yy = np.linspace(-11, 11, 4)
        mg = np.meshgrid(xx, yy, indexing="ij")

        grid = met.Grid(mg)

        grid.refinement_step(nonconvex_jf)

        # at level 0 nothing special has yet happend
        self.assertEqual(len(grid.inner_cells[0]), 3)
        self.assertEqual(len(grid.boundary_cells[0]), 3)
        self.assertEqual(len(grid.outer_cells[0]), 3)

        self.assertEqual(len(grid.inhom_domain_cells[0]), 0)

        grid.refinement_step(nonconvex_jf)

        # at level 1 we have one inhomogenous domain cell (here: an inner cell)
        self.assertEqual(len(grid.inhom_domain_cells[1]), 1)

    def test_refinement_iDC2(self):
        """
        test refinement of inhomogeneous domain cells (iDCs). with more complicated function

        # This was a bug before 2020-05-12
        """

        # noinspection PyPep8Naming
        def func_deep_M_shape(x):
            """
            Function that looks like: ____/\  /\___
                                            \/

            """
            a = 7
            f1 = met.func_bump(x, offset=1.2, amplitude=-2*a, exponent=6)
            f2 = met.func_bump(x - 1, offset=0, amplitude=a, exponent=6)
            f3 = met.func_bump(x + 1, offset=0, amplitude=a, exponent=6)
            return f1 + f2 + f3

        def nonconvex_jf(xx):
            return xx[1] < func_deep_M_shape(xx[0])

        xx = np.linspace(-10, 10, 4)
        yy = np.linspace(-11, 11, 4)
        mg = np.meshgrid(xx, yy, indexing="ij")

        grid = met.Grid(mg)

        grid.refinement_step(nonconvex_jf)  # Level 0

        # at level 0 nothing special has yet happend
        self.assertEqual(len(grid.inner_cells[0]), 3)
        self.assertEqual(len(grid.boundary_cells[0]), 3)
        self.assertEqual(len(grid.outer_cells[0]), 3)

        self.assertEqual(len(grid.inhom_domain_cells[0]), 0)

        grid.refinement_step(nonconvex_jf)  # Level 1
        grid.refinement_step(nonconvex_jf)  # Level 2

        # at level 1 we have one inhomogenous domain cell (here: an inner cell)
        self.assertEqual(len(grid.inhom_domain_cells[1]), 1)
        self.assertEqual(len(grid.inhom_domain_cells[2]), 3)

        expected_cells = [grid.cell_dict[0, (1, 2)],
                          grid.cell_dict[1, (1, 2)],
                          grid.cell_dict[1, (1.5, 2)],
                         ]

        # note that grid.inhom_domain_cells[2] (associated with level 2) contains cells of level 0 and level 1
        # -> all the cells which where detected to be inhomogeneous when handling level 2

        self.assertEqual(set(grid.inhom_domain_cells[2]), set(expected_cells))

    def test_cell_roa_boundary_flag(self):

        grid = met.Grid(self.mg)

        func_sphere_nd = met.func_sphere_nd_factory(radius=1.3)
        for level in range(4):
            grid.refinement_step(func_sphere_nd)

            for c in grid.inhomogeneous_cells[level]:
                self.assertEqual(c.roa_boundary_flag, 0)

            for c in grid.inner_cells[level]:
                self.assertEqual(c.roa_boundary_flag, 1)

            for c in grid.outer_cells[level]:
                self.assertEqual(c.roa_boundary_flag, -1)

    def test_cell_roi_boundary_status(self):

        grid = met.Grid(self.mg2)

        func_sphere_nd = met.func_sphere_nd_factory(radius=1.3)
        for level in range(4):
            grid.refinement_step(func_sphere_nd)

        n1, n2 = grid.base_resolutions

        # ##test level0 cells
        # note that the cells are column-wise stacked in the levels[0]-list
        # first test min-boundary
        self.assertEqual(tuple(grid.levels[0][0].min_bcl),    (True, True))
        self.assertEqual(tuple(grid.levels[0][1].min_bcl),    (True, False))
        self.assertEqual(tuple(grid.levels[0][n2-1].min_bcl), (True, False))
        self.assertEqual(tuple(grid.levels[0][n2].min_bcl),   (False, True))
        self.assertEqual(tuple(grid.levels[0][n2+1].min_bcl), (False, False))

        all_min_bcls = [tuple(c.min_bcl) for c in grid.levels[0]]

        self.assertEqual(all_min_bcls.count((True, True)), 1)

        self.assertEqual(all_min_bcls.count((True, False)), n2 - 1)  # left column of cells (except the corner)
        self.assertEqual(all_min_bcls.count((False, True)), n1 - 1)  # lower row of cells (except the corner)

        # now test max-boundary

        all_max_bcls = [tuple(c.max_bcl) for c in grid.levels[0]]
        self.assertEqual(all_max_bcls.count((True, True)), 1)
        self.assertEqual(tuple(grid.levels[0][0].max_bcl),    (False, False))
        self.assertEqual(tuple(grid.levels[0][1].max_bcl),    (False, False))
        self.assertEqual(tuple(grid.levels[0][n2-1].max_bcl), (False, True))  # upper end of left column
        self.assertEqual(tuple(grid.levels[0][-1].max_bcl),   (True, True))   # upper right corner
        self.assertEqual(tuple(grid.levels[0][-2].max_bcl),   (True, False))  # below upper right corner


class MeshRefinement3d(unittest.TestCase):

    def setUp(self):
        xx = np.linspace(-4, 4, 9)
        yy = np.linspace(-4, 4, 9)
        zz = np.linspace(-4, 4, 9)

        self.mg = np.meshgrid(xx, yy, zz, indexing="ij")

    def test_refinement(self):

        grid = met.Grid(self.mg)

        func_sphere_nd = met.func_sphere_nd_factory(radius=1.3)
        # perform refinement and get a point_collection (pc)
        pc0 = pc = grid.refinement_step(func_sphere_nd)

        # check inner and outer points
        self.assertEqual(pc.ip.shape, (3, 7))
        self.assertEqual(pc.op.shape, (3, 722))

        # check inner and outer boundary points
        self.assertEqual(pc.ibp.shape, (3, 7))
        self.assertEqual(pc.obp.shape, (3, 74))

        # plot inner and outer points (level 0)
        ax = plt.figure().add_subplot(1, 1, 1, projection='3d')

        ax.plot(*grid.all_mg_points, '.', ms=1, color="k", alpha=0.5)

        ax.plot(*pc.ip, 'o', ms=3, color="r", alpha=0.3)
        ax.plot(*pc.op, 'o', ms=3, color="b", alpha=0.01)

        ax.plot(*pc.ibp, 'o', ms=3, color="r", alpha=1)
        ax.plot(*pc.obp, 'o', ms=3, color="b", alpha=1)

        # plot level 1

        pc1 = pc = grid.refinement_step(func_sphere_nd)

        ax.plot(*pc.ibp, 'o', ms=3, color="r", alpha=1)
        ax.plot(*pc.obp, 'o', ms=3, color="b", alpha=1)

        # advance to level 3

        pc2 = grid.refinement_step(func_sphere_nd)
        pc3 = grid.refinement_step(func_sphere_nd)

        # new figure
        ax = plt.figure().add_subplot(1, 1, 1, projection='3d')

        # ax.plot(*grid.all_mg_points, '.', ms=1, color="k", alpha=0.5)

        ax.plot(*pc0.ibp, 'o', ms=3, color="m", alpha=1)
        ax.plot(*pc1.ibp, 'o', ms=3, color="r", alpha=1)
        ax.plot(*pc2.ibp, 'o', ms=3, color="g", alpha=1)
        ax.plot(*pc3.ibp, 'o', ms=3, color="b", alpha=1)

        # -> plot the involved cells for debugging

        plt.savefig("n3_level3.png")

        if exec_show:
            plt.show()


def plot_cells2d(cells, fname=None, show=False, **kwargs):
    for i, cell in enumerate(cells):
        edges = np.array(cell.get_edge_coords())
        plt.plot(*edges.T, **kwargs)
        if fname is not None:
            # expect something like "tmp_{:03d}.png"
            plt.savefig(fname.format(i))

    if show:
        plt.show()


def plot_cells3d(cells, ax=None, fname=None, show=False, imax=None, all_points=None, alpha=1.0):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    if all_points is not None:
        ax.plot(*all_points, '.', ms=1, color="k", alpha=alpha)

    for i, cell in enumerate(cells):
        edges = np.array(cell.get_edge_coords())

        for j, e in enumerate(edges):
            ax.plot(*e.T, alpha=alpha)

        if fname is not None:
            # expect something like "tmp_{:03d}.png"
            plt.savefig(fname.format(i))

        if imax is not None and i >= imax:
            break

    if show:
        plt.show()


# if this module is executed as script, then run all tests from this module
if __name__ == '__main__':
    unittest.main()

