# -*- coding: utf-8 -*-
"""
Created 2019-05-27 17:56:03

@author: Carsten Knoll
"""

import unittesthelper as uth
import unittest
import sys
import sympy as sp
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


def main():
    # remove command line args which should not be passed to the testframework
    if 'all' in sys.argv:
        sys.argv.remove('all')

    unittest.main()


if __name__ == '__main__':
    main()
