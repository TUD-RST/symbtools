# -*- coding: utf-8 -*-
"""
Created 2019-05-27 17:56:03

@author: Carsten Knoll
"""

from symbtools.test import unittesthelper as uth
import unittest
import sys
import sympy as sp
from typing import Sequence

import symbtools as st
import symbtools.mpctools as mpc
import numpy as np


from IPython import embed as IPS


class MPCToolsTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(2001)

    @uth.optional_dependency
    def test_conversion1(self):
        x1, x2, x3 = xx = st.symb_vector("x1:4")
        u1, u2 = uu = st.symb_vector("u1:3")

        expr_sp = sp.Matrix([x1 + x2 + x3, sp.sin(x1)*x2**x3, 1.23, 0, u1*sp.exp(u2)])
        func_cs = mpc.create_casadi_func(expr_sp, xx, uu)

        xxuu = list(xx) + list(uu)
        func_np = st.expr_to_func(xxuu, expr_sp)
        argvals = np.random.rand(len(xxuu))

        argvals_cs = (argvals[:len(xx)], argvals[len(xx):])

        res_np = func_np(*argvals)
        res_cs = func_cs(*argvals_cs).full().squeeze()
        self.assertTrue(np.allclose(res_np, res_cs))

    @uth.optional_dependency
    def test_conversion2(self):
        x1, x2, x3 = xx = st.symb_vector("x1:4")
        u1, u2 = uu = st.symb_vector("u1:3")
        lmd1, lmd2 = llmd = st.symb_vector("lmd1:3")

        xxuullmd = list(xx) + list(uu) + list(llmd)

        expr_sp = sp.Matrix([x1 + x2 + x3, sp.sin(x1)*x2**x3, 1.23, 0, u1*sp.exp(u2), x1*lmd1 + lmd2**4])
        func_cs = mpc.create_casadi_func(expr_sp, xxuullmd)

        func_np = st.expr_to_func(xxuullmd, expr_sp)
        argvals = np.random.rand(len(xxuullmd))

        # unpack the array for lambdified function
        res_np = func_np(*argvals)

        # pass the whole array for casadi function
        res_cs = func_cs(argvals).full().squeeze()
        self.assertTrue(np.allclose(res_np, res_cs))

    @uth.optional_dependency
    def test_unpack(self):

        pp = mpc.SX.sym('p', 2, 5)
        xx = mpc.SX.sym('x', 5, 1)
        y = mpc.SX.sym('y')

        l_pp = mpc.unpack(pp)
        self.assertTrue(st.aux.test_type(l_pp, Sequence[mpc.SX]))
        self.assertEqual(len(l_pp), 10)

        l_ppxxy = mpc.unpack(pp, xx, y)

        self.assertTrue(st.aux.test_type(l_ppxxy, Sequence[mpc.SX]))
        self.assertEqual(len(l_ppxxy), 16)

    @uth.optional_dependency
    def test_casadify(self):

        x1, x2, x3 = xx = st.symb_vector("x1:4")
        u1, u2 = uu = st.symb_vector("u1:3")
        lmd1, lmd2 = llmd = st.symb_vector("lmd1:3")

        xxuullmd_sp = list(xx) + list(uu) + list(llmd)

        expr1_sp = sp.Matrix([x1 + x2 + x3,  sp.sin(x1)*x2**x3, 1.23, 0, u1*sp.exp(u2), x1*lmd1 + lmd2 ** 4])
        expr2_sp = sp.Matrix([x1**2 - x3**2, sp.cos(x1)*x2**x1, -0.123, 0, u2*sp.exp(-u2), x2*lmd1 + lmd2 ** -4])

        expr1_cs, cs_symbols1 = mpc.casidify(expr1_sp, xxuullmd_sp)
        expr2_cs, cs_symbols2 = mpc.casidify(expr2_sp, xxuullmd_sp, cs_vars=cs_symbols1)

        self.assertTrue(mpc.cs.is_equal(cs_symbols1, cs_symbols2))

    @uth.optional_dependency
    def test_conversion_all_funcs(self):
        x1, x2, x3 = xx = st.symb_vector("x1:4")
        u1, u2 = uu = st.symb_vector("u1:3")

        xxuusum = sum(xx) + sum(uu)

        arg = sp.tanh(xxuusum)  # limit the argument to (-1, 1)*0.99

        # see mpc.CassadiPrinter.__init__ for exlanation
        sp_func_names = mpc.CassadiPrinter().cs_func_keys.keys()

        blacklist = ["atan2", ]
        flist = [getattr(sp, name) for name in sp_func_names if name not in blacklist]

        # create the test_matrix
        expr_list = []
        for func in flist:
            if func is sp.acosh:
                # only defined for values > 1
                expr_list.append(func(1/arg))
            else:
                expr_list.append(func(arg))
        expr_sp = sp.Matrix(expr_list + [arg, xxuusum])

        func_cs = mpc.create_casadi_func(expr_sp, xx, uu)

        xxuu = list(xx) + list(uu)
        func_np = st.expr_to_func(xxuu, expr_sp)
        argvals = np.random.rand(len(xxuu))

        argvals_cs = (argvals[:len(xx)], argvals[len(xx):])

        res_np = func_np(*argvals)
        res_cs = func_cs(*argvals_cs).full().squeeze()
        self.assertTrue(np.allclose(res_np, res_cs))

    def test_distribute(self):
        arr1 = np.arange(23)

        a, b, c = mpc.distribute(arr1, (5, 2), (1, ), (3, 4))

        self.assertTrue(np.all(a == np.arange(10).reshape(5, 2)))
        self.assertTrue(np.all(b == np.array([10])))
        self.assertEqual(c.shape, (3, 4))

        # if we pass casadi matrices shape oder "F" is used to be consistent with casadi
        arr2 = mpc.cs.DM(np.arange(23).reshape(-1, 1))

        a, b, c = mpc.distribute(arr2, (2, 5), (1, ), (3, 4))

        self.assertTrue(np.all(a == np.arange(10).reshape(5, 2).T))
        self.assertTrue(np.all(b == np.array([10])))
        self.assertEqual(c.shape, (3,  4))


def main():
    # remove command line args which should not be passed to the testframework
    if 'all' in sys.argv:
        sys.argv.remove('all')

    unittest.main()


if __name__ == '__main__':
    main()
