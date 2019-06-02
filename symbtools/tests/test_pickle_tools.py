
import os
import unittest
import sympy as sp
from sympy import sin, cos, exp

import symbtools as st
import symbtools.pickle_tools as pt

from ipydex import IPS


# noinspection PyShadowingNames,PyPep8Naming,PySetFunctionToLiteral
class Tests1(unittest.TestCase):

    def setUp(self):
        st.init_attribute_store(reinit=True)

    def test_pickle_full_dump_and_load_functions1(self):

        x1, x2, x3 = xx = st.symb_vector("x1, x2, x3")
        f1 = sp.Function("f1")(x1)
        M1 = sp.Matrix([f1+x2])

        pfname = "tmp_dump_test.pcl"
        st.pickle_full_dump(M1, pfname)

        M2 = st.pickle_full_load(pfname)

        self.assertEqual(M1, M2)

    def test_convert_functions_to_symbols(self):
        x1, x2, x3 = xx = st.symb_vector("x1, x2, x3")
        f1a = sp.Function("f1", commutative=True)(st.t)
        f1b = sp.Function("f1", commutative=False)(x1, x2)
        f2a = sp.Function("f2")(x1)
        f2b = sp.Function("f2")(x1 + x2)

        funcs = {f1a, f1b, f2a, f2b}

        rplmts, function_data = pt.convert_functions_to_symbols(funcs)

        self.assertEqual(set(list(zip(*rplmts))[0]), funcs)

        symb_keys = dict(rplmts)
        fd1a = function_data[symb_keys[f1a]]
        fd1b = function_data[symb_keys[f1b]]

        self.assertEqual(fd1a.args, (st.t,))
        self.assertEqual(fd1b.args, (x1, x2,))

        self.assertEqual(fd1a.assumptions["commutative"], True)
        self.assertEqual(fd1b.assumptions["commutative"], False)

    def test_find_relevant_attributes_and_function_keys(self):
        x1, x2, x3 = xx = st.symb_vector("x1, x2, x3")
        xdot1, xdot2, xdot3 = xxd = st.time_deriv(xx, xx)
        xddot1, xddot2, xddot3 = xxdd = st.time_deriv(xx, xx, order=2)

        relevant_attributes, function_keys = pt.find_relevant_attributes([x1, xdot2])

        Derivative = sp.Derivative
        t = st.t
        x1f = x1.ddt_func
        x2f = x2.ddt_func

        eres1 = {(x1, 'ddt_func'): x1f,
                 (x1, 'ddt_child'): xdot1,
                 (xdot2, 'difforder'): 1,
                 (xdot2, 'ddt_func'): Derivative(x2f, t),
                 (xdot2, 'ddt_parent'): x2,
                 (xdot2, 'ddt_child'): xddot2,
                 (xdot1, 'difforder'): 1,
                 (xdot1, 'ddt_func'): Derivative(x1f, t),
                 (xdot1, 'ddt_parent'): x1,
                 (xdot1, 'ddt_child'): xddot1,
                 (xddot2, 'difforder'): 2,
                 (xddot2, 'ddt_func'): Derivative(x2f, (t, 2)),
                 (xddot2, 'ddt_parent'): xdot2,
                 (x2, 'ddt_func'): x2f,
                 (x2, 'ddt_child'): xdot2,
                 (xddot1, 'difforder'): 2,
                 (xddot1, 'ddt_func'): Derivative(x1f, (t, 2)),
                 (xddot1, 'ddt_parent'): xdot1}

        self.assertEqual(relevant_attributes, eres1)

        eres2 = {x1f: {(x1, 'ddt_func'), (xdot1, 'ddt_func'), (xddot1, 'ddt_func')},
                 x2f: {(xdot2, 'ddt_func'), (xddot2, 'ddt_func'), (x2, 'ddt_func')}}

        self.assertEqual(function_keys, eres2)

    def test_pickle_full_dump_and_load(self):

        xx = st.symb_vector("x1, x2, x3")
        xdot1, xdot2, xdot3 = xxd = st.time_deriv(xx, xx)

        y1, y2, y3 = yy = st.symb_vector("y1, y2, y3")
        yyd = st.time_deriv(yy, yy)
        yydd = st.time_deriv(yy, yy, order=2)

        # add custom information to data container
        xxd.data = st.Container()

        xxd.data.z1 = yy
        xxd.data.z2 = sin(yyd[2])
        xxd.data.z3 = yydd

        pfname = "tmp_dump_test.pcl"
        st.pickle_full_dump(xxd, pfname)

        self.assertEqual(xdot1.difforder, 1)
        self.assertEqual(yydd[1].difforder, 2)

        # store assumptions to compare them later
        y1_assumptions = y1.assumptions0

        # forget all difforder attributes
        st.init_attribute_store(reinit=True)

        self.assertEqual(xdot1.difforder, 0)
        self.assertEqual(yydd[1].difforder, 0)
        del xdot1, xdot2, xdot3, xxd, yydd

        xdot1, xdot2, xdot3 = xxd = st.pickle_full_load(pfname)
        yydd_new = xxd.data.z3

        self.assertEqual(xdot1.difforder, 1)
        self.assertEqual(yydd_new[1].difforder, 2)

        new_y1_assumptions = xxd.data.z1[0].assumptions0
        self.assertEqual(new_y1_assumptions, y1_assumptions)

        os.remove(pfname)

    def test_pickle_full_dump_and_load2(self):
        """
        Test with non-sympy object
        """

        xx = st.symb_vector("x1, x2, x3")
        xdot1, xdot2, xdot3 = xxd = st.time_deriv(xx, xx)

        y1, y2, y3 = yy = st.symb_vector("y1, y2, y3")
        yyd = st.time_deriv(yy, yy)
        yydd = st.time_deriv(yy, yy, order=2)

        pdata = st.Container()

        pdata.z1 = yy
        pdata.z2 = sin(yyd[2])
        pdata.z3 = yydd
        pdata.abc = xxd

        pfname = "tmp_dump_test.pcl"
        st.pickle_full_dump(pdata, pfname)

        self.assertEqual(xdot1.difforder, 1)
        self.assertEqual(yydd[1].difforder, 2)

        # forget all difforder attributes
        st.init_attribute_store(reinit=True)

        self.assertEqual(xdot1.difforder, 0)
        self.assertEqual(yydd[1].difforder, 0)
        del xdot1, xdot2, xdot3, xxd, yydd, pdata

        pdata = st.pickle_full_load(pfname)
        xdot1, xdot2, xdot3 = xxd = pdata.abc
        yydd_new = pdata.z3

        self.assertEqual(xdot1.difforder, 1)
        self.assertEqual(yydd_new[1].difforder, 2)

        with self.assertRaises(TypeError) as cm:
            st.pickle_full_dump([], pfname)
        with self.assertRaises(TypeError) as cm:
            st.pickle_full_dump(xdot1, pfname)
        with self.assertRaises(TypeError) as cm:
            st.pickle_full_dump(st.Container, pfname)

        os.remove(pfname)

    def test_pickle_full_dump_and_load3(self):
        """
        Test for correct handling of assumptions
        """

        xx = st.symb_vector("x1, x2, x3")
        xdot1, xdot2, xdot3 = xxd = st.time_deriv(xx, xx)

        y1, y2, y3 = yy = st.symb_vector("y1, y2, y3")
        yyd = st.time_deriv(yy, yy)
        yydd = st.time_deriv(yy, yy, order=2)
        s_nc = sp.Symbol('s', commutative=False)
        sk_nc = sp.Symbol('sk', commutative=False)
        s_c = sp.Symbol('s')

        pdata1 = st.Container()
        pdata1.s1 = sk_nc # different names
        pdata1.s2 = s_c
        pdata1.xx = xx

        pdata2 = st.Container()
        pdata2.s1 = s_nc # same names
        pdata2.s2 = s_c
        pdata2.xx = xx

        pfname = "tmp_dump_test.pcl"

        # this should pass
        st.pickle_full_dump(pdata1, pfname)

        with self.assertRaises(ValueError) as cm:
            st.pickle_full_dump(pdata2, pfname)

        os.remove(pfname)