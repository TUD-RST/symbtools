# -*- coding: utf-8 -*-
"""
Created 2019-10-12 13:25:50

@author: Carsten Knoll
"""

import os
import sys
import unittest
from typing import Sequence

import sympy as sp
from symbtools import unittesthelper as uth
import symbtools as st

import pytest

try:
    import symbtools.mpctools as mpc
except ImportError:
    mpc = None


@pytest.mark.optional_dependency
class AuxiliaryTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_test_type(self):

        import casadi as cs
        xx = st.symb_vector("x1:4")

        self.assertTrue(st.aux.test_type(list(xx), Sequence[sp.Symbol]))

        pp = cs.SX.sym('P', 10)
        self.assertTrue(st.aux.test_type(mpc.unpack(pp), Sequence[cs.SX]))

        self.assertTrue(st.aux.test_type([1, 2, 100], Sequence[int]))
        self.assertFalse(st.aux.test_type([1, 2, 100.0], Sequence[int]))

        self.assertTrue(st.aux.test_type(["10", "20", "XYZ"], Sequence[str]))
        self.assertFalse(st.aux.test_type([10, "20", "XYZ"], Sequence[str]))
