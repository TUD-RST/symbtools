# -*- coding: utf-8 -*-
"""
Created on 2019-07-24 14:45:26

@author: Carsten Knoll
"""


import sympy as sp
from sympy import sin, cos, exp

import numpy as np

import symbtools as st
import ipydex as ipd


import unittest


# noinspection PyShadowingNames,PyPep8Naming,PySetFunctionToLiteral
class TestCases2(unittest.TestCase):

    def setUp(self):
        st.init_attribute_store(reinit=True)

    def test_minors_funcs(self):
        s = sp.Symbol("s")

        A = sp.Matrix([[-2, 0, 1], [3, 2, 0], [2, 0, -1]])
        C = sp.Matrix([0, 1, 0]).T

        AC = st.row_stack(s*sp.eye(3) - A, C)

        M1 = st.col_minor(AC.T, 0, 1, 2)
        M2 = st.col_minor(AC.T, 0, 1, 3)
        M3 = st.col_minor(AC.T, 0, 2, 3)
        M4 = st.col_minor(AC.T, 1, 2, 3)

        all_row_minors = st.all_k_minors(AC, k=3)

        self.assertEqual(all_row_minors, [M1, M2, M3, M4])
        all_row_minors_idcs = st.all_k_minors(AC, k=3, return_indices=True)

        eres = [(((0, 1, 2), (0, 1, 2)), s**3 + s**2 - 6*s),
                (((0, 1, 3), (0, 1, 2)), 3),
                (((0, 2, 3), (0, 1, 2)), -s**2 - 3*s),
                (((1, 2, 3), (0, 1, 2)), 3*s + 3)]

        self.assertEqual(all_row_minors_idcs, eres)

