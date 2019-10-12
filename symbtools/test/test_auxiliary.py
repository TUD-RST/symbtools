# -*- coding: utf-8 -*-
"""
Created 2019-10-12 13:25:50

@author: Carsten Knoll
"""

import unittest
import sympy as sp
from symbtools.test import unittesthelper as uth
import symbtools as st
import symbtools.mpctools as mpc
from typing import Sequence


class AuxiliaryTest(unittest.TestCase):

    def setUp(self):
        pass

    @uth.optional_dependency
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


def main():
    unittest.main()


if __name__ == '__main__':
    main()
