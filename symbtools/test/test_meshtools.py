# -*- coding: utf-8 -*-
"""
Created on 2019-12-12 00:14:14

@author: Carsten Knoll
"""

import numpy as np

import symbtools.meshtools as met
import ipydex as ipd


import unittest


# noinspection PyShadowingNames,PyPep8Naming,PySetFunctionToLiteral
class TestHelperFuncs1(unittest.TestCase):

    def setUp(self):
        pass

    def test_absmax(self):
        l1 = [1, 2, 3]
        l2 = [-1, -2, -3]
        l3 = [1, 2, -3]

        self.assertEqual(met.absmax(*l1), 3)
        self.assertEqual(met.absmax(*l2), -3)
        self.assertEqual(met.absmax(*l3), -3)

    def test_modify_tuple(self):

        t1 = (3, 4.5, 10.7)

        self.assertEqual(met.modify_tuple(t1, 0, 1), (4, 4.5, 10.7))
        self.assertEqual(met.modify_tuple(t1, 1, 1), (3, 5.5, 10.7))
        self.assertEqual(met.modify_tuple(t1, 2, -1), (3, 4.5, 9.7))


class TestNode(unittest.TestCase):
    def setUp(self):
        xx = np.linspace(-4, 4, 9)
        yy = np.linspace(-4, 4, 9)

        XX, YY = mg = np.meshgrid(xx, yy, indexing="ij")

        met.create_nodes_from_mg(mg)
