# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 11:35:00 2014

@author: Carsten Knoll
"""

import os
import inspect
import warnings

import sympy as sp
from sympy import sin, cos, exp

import numpy as np
import scipy as sc
import scipy.integrate

import symbtools as st
from symbtools import lzip

try:
    import control
except ImportError:
    control = None


import unittesthelper as uth
import unittest
import test_core1
import test_time_deriv
import test_pickle_tools

uth.inject_tests_into_namespace(globals(), test_time_deriv)
uth.inject_tests_into_namespace(globals(), test_core1)


def make_abspath(*args):
    """
    returns new absolute path, basing on the path of this module
    """
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return os.path.join(current_dir, *args)


# Avoid warnings of undefined symbols from the IDE,
# but still make use of st.make_global
x1 = x2 = x3 = x4 = None
y1 = y2 = y3 = None
a1 = z4 = z7 = z10 = None


# noinspection PyShadowingNames,PyPep8Naming,PySetFunctionToLiteral
class InteractiveConvenienceTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_no_IPS_call(self):
        """
        test whether there is some call to interactive IPython (legacy from debugging)
        """
        srclines = inspect.getsourcelines(st)[0]

        def filter_func(tup):
            idx, line = tup
            return 'IPS()' in line and not line.strip()[0] == '#'

        res = list(filter(filter_func, enumerate(srclines, 1)))

        self.assertEqual(res, [])

    def test_symbol_atoms(self):
        a, b, t = sp.symbols("a, b, t")
        x1 = a + b
        x2 = a + b - 3 + sp.pi
        M1 = sp.Matrix([x2, t, a**2])
        M2 = sp.ImmutableDenseMatrix(M1)

        self.assertEqual(set([a]), a.s)
        self.assertEqual(x1.atoms(), x1.s)
        self.assertEqual(x2.atoms(sp.Symbol), x2.s)

        self.assertEqual(set([a, b, t]), M1.s)
        self.assertEqual(set([a, b, t]), M2.s)

    def test_count_ops(self):
        a, b, t = sp.symbols("a, b, t")
        x1 = a + b
        x2 = a + b - 3 + sp.pi
        M1 = sp.Matrix([x2, t, a**2])
        M2 = sp.ImmutableDenseMatrix(M1)

        self.assertEqual(st.count_ops(a), a.co)
        self.assertEqual(st.count_ops(x1), x1.co)
        self.assertEqual(st.count_ops(x2), x2.co)
        self.assertEqual(st.count_ops(M1), M1.co)
        self.assertEqual(st.count_ops(M2), M2.co)

    def test_count_ops2(self):
        a, b, t = sp.symbols("a, b, t")
        x1 = a + b
        x2 = a + b - 3 + sp.pi
        M1 = sp.Matrix([x2, t, a**2, 0, 1])
        M2 = sp.ImmutableDenseMatrix(M1)

        self.assertEqual(st.count_ops(0), 0)
        self.assertEqual(st.count_ops(a), 1)
        self.assertEqual(st.count_ops(1.3), 1)
        self.assertEqual(st.count_ops(x1), 2)
        self.assertEqual(st.count_ops(x2), 4)
        self.assertEqual(st.count_ops(M1), sp.Matrix([4, 1, 2, 0, 1]))
        self.assertEqual(st.count_ops(M2), sp.Matrix([4, 1, 2, 0, 1]))

    def test_srn(self):
        x, y, z = xyz = st.symb_vector('x, y, z')
        st.random.seed(3319)
        self.assertAlmostEqual(x.srn01, 0.843044195656457)

        st.random.seed(3319)
        x_srn = x.srn
        self.assertNotAlmostEqual(x_srn, 8.59)
        self.assertAlmostEqual(x_srn, 8.58739776090811)

        # now apply round
        st.random.seed(3319)
        self.assertAlmostEqual(x.srnr, 8.59)

        # test compatibility with sp.Matrix
        # the order might depend on the platform (due to dict ordering)
        expected_res = [5.667115517927374668261109036393463611602783203125,
                        7.76957198624519962404377793063758872449398040771484375,
                        8.58739776090810946751474830307415686547756195068359375]

        st.random.seed(3319)
        xyz_srn = list(xyz.srn)
        xyz_srn.sort()
        for a, b in zip(xyz_srn, expected_res):
            self.assertAlmostEqual(a, b)

        # should live in a separate test !!
        st.random.seed(3319)

        # ensure that application to matrix does raise exception
        _ = xyz.srnr

        test_matrix = sp.Matrix(expected_res)

        rounded_res = sp.Matrix([[5.667], [ 7.77], [8.587]])

        self.assertNotEqual(test_matrix, rounded_res)
        self.assertEqual(test_matrix.ar, rounded_res)

    def test_subz(self):
        x1, x2, x3 = xx = sp.Matrix(sp.symbols("x1, x2, x3"))
        y1, y2, y3 = yy = sp.symbols("y1, y2, y3")

        a = x1 + 7*x2*x3
        M1 = sp.Matrix([x2, x1*x2, x3**2])
        M2 = sp.ImmutableDenseMatrix(M1)

        self.assertEqual(x1.subs(lzip(xx, yy)), x1.subz(xx, yy))
        self.assertEqual(a.subs(lzip(xx, yy)), a.subz(xx, yy))
        self.assertEqual(M1.subs(lzip(xx, yy)), M1.subz(xx, yy))
        self.assertEqual(M2.subs(lzip(xx, yy)), M2.subz(xx, yy))

    def test_smplf(self):
        x1, x2, x3 = xx = sp.Matrix(sp.symbols("x1, x2, x3"))
        y1, y2, y3 = yy = sp.symbols("y1, y2, y3")

        a = x1**2*(x2/x1 + 7) - x1*x2
        M1 = sp.Matrix([sin(x1)**2 + cos(x1)**2, a, x3])

        self.assertEqual(M1.smplf, sp.simplify(M1))
        self.assertEqual(a.smplf, sp.simplify(a))

    def test_subz0(self):
        x1, x2, x3 = xx = st.symb_vector("x1, x2, x3")
        y1, y2, y3 = yy = st.symb_vector("y1, y2, y3")

        XX = (x1, x2)

        a = x1 + 7*x2*x3
        M1 = sp.Matrix([x2, x1*x2, x3**2])
        M2 = sp.ImmutableDenseMatrix(M1)

        self.assertEqual(x1.subs(st.zip0(XX)), x1.subz0(XX))
        self.assertEqual(a.subs(st.zip0(XX)), a.subz0(XX))
        self.assertEqual(M1.subs(st.zip0(XX)), M1.subz0(XX))
        self.assertEqual(M2.subs(st.zip0(XX)), M2.subz0(XX))

        konst = sp.Matrix([1,2,3])
        zz = konst + xx + 5*yy
        self.assertEqual(zz.subz0(xx, yy), konst)


# noinspection PyShadowingNames,PyPep8Naming,PySetFunctionToLiteral
class LieToolsTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_involutivity_test(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4')
        st.make_global(xx)

        # not involutive
        f1 = sp.Matrix([x2*x3 + x1**2, 3*x1, 4 + x2*x3])
        f2 = sp.Matrix([x3 - 2*x1*x3, x2 - 5, 3 + x1*x2])

        dist1 = st.col_stack(f1, f2)

        # involutive
        f3 = sp.Matrix([-x2, x1, 0])
        f4 = sp.Matrix([0, -x3, x2])

        dist2 = st.col_stack(f3, f4)

        res, fail = st.involutivity_test(dist1, xx)

        self.assertFalse(res)
        self.assertEqual(fail, (0, 1))

        res2, fail2 = st.involutivity_test(dist2, xx)

        self.assertTrue(res2)
        self.assertEqual(fail2, [])

    def test_lie_deriv_cartan(self):
        x1, x2, x3 = xx = sp.symbols('x1:4')
        u1, u2 = uu = sp.Matrix(sp.symbols('u1:3'))

        # ordinary lie_derivative

        # source: inspired by the script of Prof. Kugi (TU-Wien)
        f = sp.Matrix([-x1**3, cos(x1)*cos(x2), x2])
        g = sp.Matrix([cos(x2), 1, exp(x1)])
        h = x3
        Lfh = x2
        Lf2h = f[1]
        Lgh = exp(x1)

        res1 = st.lie_deriv_cartan(h, f, xx)
        res2 = st.lie_deriv_cartan(h, f, xx, order=2)

        self.assertEqual(res1, Lfh)
        self.assertEqual(res2, Lf2h)

        # incorporating the input
        h2 = u1

        udot1, udot2 = uudot = st.time_deriv(uu, uu, order=1)
        uddot1, uddot2 = st.time_deriv(uu, uu, order=2)

        res_a1 = st.lie_deriv_cartan(h2, f, xx, uu, order=1)
        res_a2 = st.lie_deriv_cartan(h2, f, xx, uu, order=2)

        self.assertEqual(res_a1, udot1)
        self.assertEqual(res_a2, uddot1)

        res_a3 = st.lie_deriv_cartan(udot1, f, xx, [uu, uudot], order=1)
        self.assertEqual(res_a3, uddot1)

        # more complex examples
        h3 = x3 + u1
        fg = f + g * u2

        res_b1 = st.lie_deriv_cartan(h3, fg, xx, uu, order=1)
        res_b2 = st.lie_deriv_cartan(h3, fg, xx, uu, order=2)
        res_b3 = st.lie_deriv_cartan(res_b1, fg, xx, [uu, uudot], order=1)

        self.assertEqual(res_b1, Lfh + Lgh*u2 + udot1)
        self.assertEqual(sp.expand(res_b2 - res_b3), 0)

        h4 = x3 * sin(x2)
        fg = f + g * u2

        res_c1 = st.lie_deriv_cartan(h4, fg, xx, uu, order=1)
        res_c2 = st.lie_deriv_cartan(res_c1, fg, xx, uu, order=1)
        res_c3 = st.lie_deriv_cartan(h4, fg, xx, uu, order=2)

        self.assertEqual(sp.expand(res_c2 - res_c3), 0)

    def test_lie_deriv(self):
        xx = st.symb_vector('x1:4')
        st.make_global(xx)

        f = sp.Matrix([x1 + x3*x2, 7*exp(x1), cos(x2)])
        h1 = x1**2 + sin(x3)*x2
        res1 = st.lie_deriv(h1, f, xx)
        eres1 = 2*x1**2 + 2*x1*x2*x3 + 7*exp(x1)*sin(x3) + x2*cos(x2)*cos(x3)
        self.assertEqual(res1.expand(), eres1)

        res2a = st.lie_deriv(h1, f, xx, order=2).expand()
        res2b = st.lie_deriv(h1, f, xx, 2).expand()
        eres2 = st.lie_deriv(eres1, f, xx).expand()

        self.assertEqual(res2a, eres2)
        self.assertEqual(res2b, eres2)

        res2c = st.lie_deriv(h1, f, f, xx).expand()
        res2d = st.lie_deriv(h1, f, f, xx=xx).expand()
        self.assertEqual(res2c, eres2)
        self.assertEqual(res2d, eres2)

        F = f[:-1, :]
        with self.assertRaises(ValueError) as cm:
            # different lengths of vectorfields:
            res1 = st.lie_deriv(h1, F, f, xx)

    # noinspection PyTypeChecker
    def test_lie_bracket(self):
        xx = st.symb_vector('x1:4')
        st.make_global(xx)
        fx = sp.Matrix([[(x2 - 1)**2 + 1/x3], [x1 + 7], [-x3**2*(x2 - 1)]])
        v = sp.Matrix([[0], [0], [-x3**2]])

        dist = st.col_stack(v, st.lie_bracket(-fx, v, xx), st.lie_bracket(-fx, v, xx, order=2))

        v0, v1, v2 = st.col_split(dist)

        self.assertEqual(v1, sp.Matrix([1, 0, 0]))
        self.assertEqual(v2, sp.Matrix([0, 1, 0]))

        self.assertEqual(st.lie_bracket(fx, fx, xx), sp.Matrix([0, 0, 0]))

    def test_lie_deriv_covf(self):
        xx = st.symb_vector('x1:4')
        st.make_global(xx)

        # we test this by building the observability matrix with two different but equivalent approaches
        f = sp.Matrix([x1 + x3*x2, 7*exp(x1), cos(x2)])
        y = x1**2 + sin(x3)*x2
        ydot = st.lie_deriv(y, f, xx)
        yddot = st.lie_deriv(ydot, f, xx)

        cvf1 = st.gradient(y, xx)
        cvf2 = st.gradient(ydot, xx)
        cvf3 = st.gradient(yddot, xx)

        # these are the rows of the observability matrix

        # second approach
        dh0 = cvf1
        dh1 = st.lie_deriv_covf(dh0, f, xx)
        dh2a = st.lie_deriv_covf(dh1, f, xx)
        dh2b = st.lie_deriv_covf(dh0, f, xx, order=2)

        zero = dh0*0

        self.assertEqual((dh1 - cvf2).expand(), zero)
        self.assertEqual((dh2a - cvf3).expand(), zero)
        self.assertEqual((dh2b - cvf3).expand(), zero)


# noinspection PyShadowingNames,PyPep8Naming,PySetFunctionToLiteral
class TestSupportFunctions(unittest.TestCase):
    """
    Test functionality which is used indirectly by other functions
    """

    def setUp(self):
        pass

    def test_recursive_function_decorator(self):

        @st.recursive_function
        def myfactorial(thisfunc, x):
            if x == 0:
                return 1
            else:
                return x*thisfunc(x-1)

        nn = [0, 1, 3, 5, 10]
        res1 = [sp.factorial(x) for x in nn]
        res2 = [myfactorial(x) for x in nn]

        self.assertEqual(res1, res2)

    def test_get_custom_attr_map(self):

        t = st.t
        x1, x2 = xx = st.symb_vector("x1, x2")
        xdot1, xdot2 = xxd = st.time_deriv(xx, xx)
        xddot1, xddot2 = xxdd = st.time_deriv(xx, xx, order=2)

        m1 = st.get_custom_attr_map("ddt_child")
        em1 = [(x1, xdot1), (x2, xdot2), (xdot1, xddot1), (xdot2, xddot2)]
        # convert to set because sorting might depend on plattform
        self.assertEqual(set(m1), set(em1))

        m2 = st.get_custom_attr_map("ddt_parent")
        em2 = [(xdot1, x1), (xdot2, x2), (xddot1, xdot1), (xddot2, xdot2)]
        self.assertEqual(set(m2), set(em2))

        m3 = st.get_custom_attr_map("ddt_func")
        # ensure unique sorting
        m3.sort(key=lambda x: "{}_{}".format(x[0].difforder, str(x[0])))
        self.assertEqual(len(m3), 6)

        x2_func = sp.Function(x2.name)(t)

        self.assertEqual(type(type(m3[0][1])), sp.function.UndefinedFunction)
        self.assertEqual(m3[-1][1], x2_func.diff(t, t))


# noinspection PyShadowingNames,PyPep8Naming,PySetFunctionToLiteral
class SymbToolsTest2(unittest.TestCase):

    def setUp(self):
        pass

    def test_solve_scalar_ode_1sto(self):
        a, b = sp.symbols("a, b", nonzero=True)
        t, x1, x2 = sp.symbols("t, x1, x2")

        # x1_dot = <rhs>
        rhs1 = sp.S(0)
        rhs2 = sp.S(2.5)
        rhs3 = x1
        rhs5 = x1*(3-t)
        rhs6 = cos(b*t)  # coeff must be nonzero to prevent case distinction

        res1 = st.solve_scalar_ode_1sto(rhs1, x1, t)
        self.assertEqual(res1.diff(t), rhs1.subs(x1, res1))

        res2 = st.solve_scalar_ode_1sto(rhs2, x1, t)
        self.assertEqual(res2.diff(t), rhs2.subs(x1, res2))

        res3, iv3 = st.solve_scalar_ode_1sto(rhs3, x1, t, return_iv=True)
        self.assertEqual(res3.diff(t), rhs3.subs(x1, res3))
        self.assertEqual(res3, iv3*exp(t))

        res5 = st.solve_scalar_ode_1sto(rhs5, x1, t)
        test_difference5 = res5.diff(t) - rhs5.subs(x1, res5)
        self.assertEqual(test_difference5.expand(), 0)

        res6 = st.solve_scalar_ode_1sto(rhs6, x1, t)
        self.assertEqual(res6.diff(t), rhs6.subs(x1, res6).expand())

    @uth.skip_slow
    def test_solve_scalar_ode_1sto_2(self):
        a, b = sp.symbols("a, b", nonzero=True)
        t, x1, x2 = sp.symbols("t, x1, x2")
        rhs4 = sin(a*x1)

        # this test works but is slow
        with st.warnings.catch_warnings(record=True) as cm:
            res4 = st.solve_scalar_ode_1sto(rhs4, x1, t)

        self.assertEqual(len(cm), 1)
        self.assertTrue('multiple solutions' in str(cm[0].message))

        test_difference4 = res4.diff(t) - rhs4.subs(x1, res4)

        self.assertEqual(test_difference4.simplify(), 0)

    def test_calc_flow_from_vectorfield(self):
        a, b = sp.symbols("a, b", nonzero=True)
        t, x1, x2, x3, x4 = sp.symbols("t, x1, x2, x3, x4")
        xx = x1, x2, x3, x4

        vf1 = sp.Matrix([0, 1, x3])
        vf2 = sp.Matrix([0, 1, x3, sin(a*x2)])

        res1, fp, iv1 = st.calc_flow_from_vectorfield(vf1, xx[:-1], flow_parameter=t)
        vf1_sol = vf1.subs(lzip(xx[:-1], res1))
        self.assertEqual(fp, t)
        self.assertEqual(res1.diff(t), vf1_sol)

        res2, fp, iv2 = st.calc_flow_from_vectorfield(vf2, xx, flow_parameter=t)
        vf2_sol = vf2.subs(lzip(xx[:-1], res2))
        self.assertEqual(fp, t)
        self.assertEqual(res2.diff(t), vf2_sol)

        res3, fp, iv3 = st.calc_flow_from_vectorfield(sp.Matrix([x1, 1, x1]), xx[:-1])

        t = fp
        x1_0, x2_0, x3_0 = iv3
        ref3 = sp.Matrix([[x1_0*sp.exp(t)], [t + x2_0], [x1_0*sp.exp(t) - x1_0 + x3_0]])

        self.assertEqual(res3, ref3)

    def test_create_simfunction(self):
        x1, x2, x3, x4 = xx = sp.Matrix(sp.symbols("x1, x2, x3, x4"))
        u1, u2 = uu = sp.Matrix(sp.symbols("u1, u2"))  # inputs
        p1, p2, p3, p4 = pp = sp.Matrix(sp.symbols("p1, p2, p3, p4"))  # parameter
        t = sp.Symbol('t')

        A = A0 =  sp.randMatrix(len(xx), len(xx), -10, 10, seed=704)
        B = B0 = sp.randMatrix(len(xx), len(uu), -10, 10, seed=705)

        v1 = A[0, 0]
        A[0, 0] = p1
        v2 = A[2, -1]
        A[2, -1] = p2
        v3 = B[3, 0]
        B[3, 0] = p3
        v4 = B[2, 1]
        B[2, 1] = p4

        par_vals = lzip(pp, [v1, v2, v3, v4])

        f = A*xx
        G = B

        fxu = (f + G*uu).subs(par_vals)

        # some random initial values
        x0 = st.to_np( sp.randMatrix(len(xx), 1, -10, 10, seed=706) ).squeeze()

        # Test handling of unsubstituted parameters
        mod = st.SimulationModel(f, G, xx, model_parameters=par_vals[1:])
        with self.assertRaises(ValueError) as cm:
            rhs0 = mod.create_simfunction()

        self.assertTrue("unexpected symbols" in cm.exception.args[0])

        # create the model and the rhs-function
        mod = st.SimulationModel(f, G, xx, par_vals)
        rhs0 = mod.create_simfunction()
        self.assertFalse(mod.compiler_called)
        self.assertFalse(mod.use_sp2c)

        res0_1 = rhs0(x0, 0)
        dres0_1 = st.to_np(fxu.subs(lzip(xx, x0) + st.zip0(uu))).squeeze()

        bin_res01 = np.isclose(res0_1, dres0_1)  # binary array
        self.assertTrue( np.all(bin_res01) )

        # difference should be [0, 0, ..., 0]
        self.assertFalse( np.any(rhs0(x0, 0) - rhs0(x0, 3.7) ) )

        # simulate
        tt = np.linspace(0, 0.5, 100)  # simulation should be short due to instability
        res1 = sc.integrate.odeint(rhs0, x0, tt)

        # create and try sympy_to_c bridge (currently only works on linux
        # and if sympy_to_c is installed (e.g. with `pip install sympy_to_c`))
        # until it is not available for windows we do not want it as a requirement
        # see also https://stackoverflow.com/a/10572833/333403

        try:
            import sympy_to_c
        except ImportError:
            # noinspection PyUnusedLocal
            sympy_to_c = None
            sp2c_available = False
        else:
            sp2c_available = True

        if sp2c_available:

            rhs0_c = mod.create_simfunction(use_sp2c=True)
            self.assertTrue(mod.compiler_called)
            res1_c = sc.integrate.odeint(rhs0_c, x0, tt)
            self.assertTrue(np.all(np.isclose(res1_c, res1)))

            mod.compiler_called = None
            rhs0_c = mod.create_simfunction(use_sp2c=True)
            self.assertTrue(mod.compiler_called is None)

        # proof calculation
        # x(t) = x0*exp(A*t)
        Anum = st.to_np(A.subs(par_vals))
        Bnum = st.to_np(G.subs(par_vals))
        # noinspection PyUnresolvedReferences
        xt = [ np.dot( sc.linalg.expm(Anum*T), x0 ) for T in tt ]
        xt = np.array(xt)

        # test whether numeric results are close within given tolerance
        bin_res1 = np.isclose(res1, xt, rtol=2e-5)  # binary array

        self.assertTrue( np.all(bin_res1) )

        # test handling of parameter free models:

        mod2 = st.SimulationModel(Anum*xx, Bnum, xx)
        rhs2 = mod2.create_simfunction()
        res2 = sc.integrate.odeint(rhs2, x0, tt)
        self.assertTrue(np.allclose(res1, res2))

        # test input functions
        des_input = st.piece_wise((0, t <= 1 ), (t, t < 2), (0.5, t < 3), (1, True))
        des_input_func_scalar = st.expr_to_func(t, des_input)
        des_input_func_vec = st.expr_to_func(t, sp.Matrix([des_input, des_input]) )

        # noinspection PyUnusedLocal
        with self.assertRaises(TypeError) as cm:
            mod2.create_simfunction(input_function=des_input_func_scalar)

        rhs3 = mod2.create_simfunction(input_function=des_input_func_vec)
        # noinspection PyUnusedLocal
        res3_0 = rhs3(x0, 0)

    def test_create_simfunction2(self):
        x1, x2, x3, x4 = xx = sp.Matrix(sp.symbols("x1, x2, x3, x4"))
        u1, u2 = uu = sp.Matrix(sp.symbols("u1, u2"))  # inputs
        p1, p2, p3, p4 = pp = sp.Matrix(sp.symbols("p1, p2, p3, p4"))  # parameter
        t = sp.Symbol('t')

        A = A0 =  sp.randMatrix(len(xx), len(xx), -10, 10, seed=704)
        B = B0 = sp.randMatrix(len(xx), len(uu), -10, 10, seed=705)

        v1 = A[0, 0]
        A[0, 0] = p1
        v2 = A[2, -1]
        A[2, -1] = p2
        v3 = B[3, 0]
        B[3, 0] = p3
        v4 = B[2, 1]
        B[2, 1] = p4

        par_vals = lzip(pp, [v1, v2, v3, v4])

        f = A*xx
        G = B

        fxu = (f + G*uu).subs(par_vals)

        # some random initial values
        x0 = st.to_np( sp.randMatrix(len(xx), 1, -10, 10, seed=706) ).squeeze()
        u0 = st.to_np( sp.randMatrix(len(uu), 1, -10, 10, seed=2257) ).squeeze()

        # create the model and the rhs-function
        mod = st.SimulationModel(f, G, xx, par_vals)
        rhs_xx_uu = mod.create_simfunction(free_input_args=True)

        res0_1 = rhs_xx_uu(x0, u0, 0)
        dres0_1 = st.to_np(fxu.subs(lzip(xx, x0) + lzip(uu, u0))).squeeze()

        bin_res01 = np.isclose(res0_1, dres0_1)  # binary array
        self.assertTrue( np.all(bin_res01) )

    def test_num_trajectory_compatibility_test(self):
        x1, x2, x3, x4 = xx = sp.Matrix(sp.symbols("x1, x2, x3, x4"))
        u1, u2 = uu = sp.Matrix(sp.symbols("u1, u2"))  # inputs

        t = sp.Symbol('t')

        # we want to create a random but stable matrix

        np.random.seed(2805)
        diag = np.diag( np.random.random(len(xx))*-10 )
        T = sp.randMatrix(len(xx), len(xx), -10, 10, seed=704)
        Tinv = T.inv()

        A = Tinv*diag*T

        B = B0 = sp.randMatrix(len(xx), len(uu), -10, 10, seed=705)

        x0 = st.to_np( sp.randMatrix(len(xx), 1, -10, 10, seed=706) ).squeeze()
        tt = np.linspace(0, 5, 2000)

        des_input = st.piece_wise((2-t, t <= 1 ), (t, t < 2), (2*t-2, t < 3), (4, True))
        des_input_func_vec = st.expr_to_func(t, sp.Matrix([des_input, des_input]) )

        mod2 = st.SimulationModel(A*xx, B, xx)
        rhs3 = mod2.create_simfunction(input_function=des_input_func_vec)
        XX = sc.integrate.odeint(rhs3, x0, tt)
        UU = des_input_func_vec(tt)

        res1 = mod2.num_trajectory_compatibility_test(tt, XX, UU)
        self.assertTrue(res1)

        # slightly different input signal -> other results
        res2 = mod2.num_trajectory_compatibility_test(tt, XX, UU*1.1)
        self.assertFalse(res2)

    def test_expr_to_func(self):

        x1, x2 = xx = sp.Matrix(sp.symbols("x1, x2"))
        t, = sp.symbols("t,")
        r_ = np.r_

        f1 = st.expr_to_func(x1, 2*x1)
        self.assertEqual(f1(5.1), 10.2)

        XX1 = np.r_[1, 2, 3.7]
        res1 = f1(XX1) == 2*XX1
        self.assertTrue(res1.all)

        f2 = st.expr_to_func(x1, sp.Matrix([x1*2, x1+5, 4]))
        res2 = f2(3) == r_[6, 8, 4]
        self.assertTrue(res2.all())

        res2b = f2(r_[3, 10, 0]) == np.array([[6, 8, 4], [20, 15, 4], [0, 5, 4]])
        self.assertTrue(res2b.all())

        f3 = st.expr_to_func(xx, sp.Matrix([x1*2, x2+5, 4]))
        res3 = np.allclose(f3(-3.1, 4), r_[-6.2, 9, 4])

        self.assertTrue(res3)

        # test compatibility with Piecewise Expressions
        des_input = st.piece_wise((0, t <= 1 ), (t, t < 2), (0.5, t < 3), (1, True))
        f4s = st.expr_to_func(t, des_input)
        f4v = st.expr_to_func(t, sp.Matrix([des_input, des_input]) )

        self.assertEqual(f4s(2.7), 0.5)

        sol = r_[0, 1.6, 0.5, 1, 1]
        res4a = f4s(r_[0.3, 1.6, 2.2, 3.1, 500]) == sol
        self.assertTrue(res4a.all())

        res4b = f4v(r_[0.3, 1.6, 2.2, 3.1, 500])
        col1, col2 = res4b.T
        self.assertTrue(np.array_equal(col1, sol))
        self.assertTrue(np.array_equal(col2, sol))

        spmatrix = sp.Matrix([[x1, x1*x2], [0, x2**2]])

        fnc1 = st.expr_to_func(xx, spmatrix, keep_shape=False)
        fnc2 = st.expr_to_func(xx, spmatrix, keep_shape=True)

        res1 = fnc1(1.0, 2.0)
        res2 = fnc2(1.0, 2.0)

        self.assertEqual(res1.shape, (4, ))
        self.assertEqual(res2.shape, (2, 2))

        # noinspection PyTypeChecker
        self.assertTrue(np.all(res1 == [1, 2, 0, 4]))
        # noinspection PyTypeChecker
        self.assertTrue(np.all(res1 == res2.flatten()))

    def test_reformulate_Integral(self):
        t = sp.Symbol('t')
        c = sp.Symbol('c')
        F = sp.Function('F')
        x = sp.Function('x')(t)
        a = sp.Function('a')

        i1 = sp.Integral(F(t), t)
        j1 = st.reformulate_integral_args(i1)
        self.assertEqual(j1.subs(t, 0).doit(), 0)

        ode = x.diff(t) + x -a(t)*x**c
        sol = sp.dsolve(ode, x).rhs
        # the solution contains an undetemined integral
        self.assertTrue( len(sol.atoms(sp.Integral)) == 1)

        # extract the integration constant (not necessary for test)
        # C1 = list(sol.atoms(sp.Symbol)-ode.atoms(sp.Symbol))[0]

        sol2 = st.reformulate_integral_args(sol)
        self.assertTrue( len(sol2.atoms(sp.Integral)) == 1)

        sol2_at_0 = sol2.subs(t, 0).doit()
        self.assertTrue( len(sol2_at_0.atoms(sp.Integral)) == 0)


# noinspection PyShadowingNames,PyPep8Naming,PySetFunctionToLiteral
class SymbToolsTest3(unittest.TestCase):

    def setUp(self):
        st.init_attribute_store(reinit=True)

    def test_get_symbols_by_name(self):
        c1, C1, x, a, t, Y = sp.symbols('c1, C1, x, a, t, Y')
        F = sp.Function('F')

        expr1 = c1*(C1+x**x)/(sp.sin(a*t))
        expr2 = sp.Matrix([sp.Integral(F(x), x)*sp.sin(a*t) - \
                           1/F(x).diff(x)*C1*Y])

        res1 = st.get_symbols_by_name(expr1, 'c1')
        self.assertEqual(res1, c1)
        res2 = st.get_symbols_by_name(expr1, 'C1')
        self.assertEqual(res2, C1)
        res3 = st.get_symbols_by_name(expr1, *'c1 x a'.split())
        self.assertEqual(res3, [c1, x, a])

        with self.assertRaises(ValueError) as cm:
            st.get_symbols_by_name(expr1, 'Y')
        with self.assertRaises(ValueError) as cm:
            st.get_symbols_by_name(expr1, 'c1', 'Y')

        res4 = st.get_symbols_by_name(expr2, 'Y')
        self.assertEqual(res4, Y)
        res5 = st.get_symbols_by_name(expr2, 'C1')
        self.assertEqual(res5, C1)
        res6 = st.get_symbols_by_name(expr2, *'C1 x a'.split())
        self.assertEqual(res6, [C1, x, a])

    def test_general_attribute(self):
        st.register_new_attribute_for_sp_symbol("foo", save_setter=False)
        st.register_new_attribute_for_sp_symbol("bar", getter_default="__self__")
        x1 = sp.Symbol('x1')

        self.assertEqual(x1.foo, None)
        self.assertEqual(x1.bar, x1)

        x1.foo = 7
        self.assertEqual(x1.foo, 7)

        x1.foo = "some string"
        self.assertEqual(x1.foo, "some string")

        x1.foo = x1
        self.assertEqual(x1.foo, x1)

        x1.bar = 12

        # noinspection PyUnusedLocal
        with self.assertRaises(ValueError) as cm:
            x1.bar = 13

    def test_difforder_attribute(self):
        x1 = sp.Symbol('x1')

        self.assertEqual(x1.difforder, 0)

        xddddot1 = st.time_deriv(x1, [x1], order=4)
        self.assertEqual(xddddot1.difforder, 4)

        xx = sp.Matrix(sp.symbols("x1, x2, x3"))
        xxd = st.time_deriv(xx, xx)
        xxdd = st.time_deriv(xx, xx, order=2)
        for xdd in xxdd:
            self.assertEqual(xdd.difforder, 2)

        # once, this was a bug
        y = sp.Symbol('y')
        ydot = st.time_deriv(y, [y])
        yddot = st.time_deriv(ydot, [y, ydot])
        self.assertEqual(yddot.difforder, 2)

        z = sp.Symbol('z')
        zdot_false = sp.Symbol('zdot')
        st.global_data.attribute_store[(zdot_false, 'difforder')] = -7

        with self.assertRaises(ValueError) as cm:
            st.time_deriv( z, [z])

        # ensure that difforder is not changed after value_set
        z2 = sp.Symbol('z2')
        z2.difforder = 3

        z2.difforder = 3  # same value is allowed

        with self.assertRaises(ValueError) as cm:
            z2.difforder = 4  # not allowed

    def test_introduce_abreviations(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4')
        a1, a2, a3 = aa = st.symb_vector('a1:4')

        P1 = sp.eye(3)
        P2 = sp.Matrix([x1**2, a1+a2, a3*x2, 13.7, 1, 0])

        res1 = st.introduce_abreviations(P1)
        res2 = st.introduce_abreviations(P1, time_dep_symbs=xx)
        res3 = st.introduce_abreviations(P2, time_dep_symbs=xx)

        self.assertEqual(res1[0], P1)
        self.assertEqual(res2[0], P1)

        # test subs_tuples
        self.assertNotEqual(res3[0], P2)
        self.assertEqual(res3[0].subs(res3[1]), P2)

        # time dependend symbols
        tds = res3[2]
        original_expressions = tds.subs(res3[1])
        self.assertEqual(original_expressions, sp.Matrix([x1**2, a3*x2]))

    def _test_make_global(self):

        xx = st.symb_vector('x1:4')
        yy = st.symb_vector('y1:4')

        st.make_global(xx)
        self.assertEqual(x1 + x2, xx[0] + xx[1])

        # test if set is accepted
        st.make_global(yy.atoms(sp.Symbol))
        self.assertEqual(y1 + y2, yy[0] + yy[1])

        with self.assertRaises(TypeError) as cm:
            st.make_global(dict())

    def test_make_global(self):

        aa = tuple(st.symb_vector('a1:4'))
        xx = st.symb_vector('x1:4')
        yy = st.symb_vector('y1:4')
        zz = st.symb_vector('z1:11').reshape(2, 5)

        # tollerate if there are numbers in the sequences:
        zz[0] = 0
        zz[1] = 10

        st.make_global(xx, yy, zz, aa)

        res = a1 + x2 + y3 + z4 + z7 + z10
        res2 = aa[0] + xx[1] + yy[2] + zz[3] + zz[6] + zz[9]

        self.assertEqual(res, res2)


# noinspection PyShadowingNames,PyPep8Naming,PySetFunctionToLiteral
class SymbToolsTest4(unittest.TestCase):

    def setUp(self):
        st.init_attribute_store(reinit=True)

    def test_re_im(self):
        x, y = sp.symbols('x, y', real=True)
        M1 = sp.Matrix([[x, 0], [sp.pi, 5*x**2]])
        M2 = sp.Matrix([[y, 3], [sp.exp(1), 7/y]])

        M = M1 + 1j*M2
        R = st.re(M)
        I = st.im(M)

        self.assertEqual(R-M1, R*0)
        self.assertEqual(I-M2, R*0)

    def test_is_number(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4')

        self.assertTrue(st.is_number(x1/x1))
        self.assertTrue(st.is_number(5))
        self.assertTrue(st.is_number(5.3))
        self.assertTrue(st.is_number(sp.pi))
        self.assertTrue(st.is_number(sp.Rational(2, 7)))
        self.assertTrue(st.is_number(sp.Rational(2, 7).evalf(30)))
        self.assertTrue(st.is_number(sin(7)))
        self.assertTrue(st.is_number(np.float(9000)))

        self.assertFalse(st.is_number(x1))
        self.assertFalse(st.is_number(sin(x1)))

        with self.assertRaises(TypeError) as cm:
            st.is_number( sp.eye(3) )

        with self.assertRaises(TypeError) as cm:
            st.is_number( "567" )

    def test_is_scalar(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4')

        self.assertTrue(st.is_scalar(x1/x1))
        self.assertTrue(st.is_scalar(5))
        self.assertTrue(st.is_scalar(5.3))
        self.assertTrue(st.is_scalar(sp.pi))
        self.assertTrue(st.is_scalar(sp.Rational(2, 7)))
        self.assertTrue(st.is_scalar(sp.Rational(2, 7).evalf(30)))
        self.assertTrue(st.is_scalar(sin(7)))
        self.assertTrue(st.is_scalar(np.float(9000)))
        self.assertTrue(st.is_scalar(x1**2 + x3))

        self.assertFalse(st.is_scalar( sp.eye(3)*x2 ))
        self.assertFalse(st.is_scalar( sp.zeros(2, 4)*x2 ))
        self.assertFalse(st.is_scalar( sp.eye(0)*x2 ))

    def test_is_scalar2(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4')
        a1, a2, a3 = aa = st.symb_vector('a1:4')

        M1 = sp.Matrix([[0, 0], [a1, a2], [0, a3]])
        M2 = sp.ImmutableDenseMatrix(M1)

        iss = st.is_scalar

        self.assertTrue(iss(x1))
        self.assertTrue(iss(x1 ** 2 + sp.sin(x2)))
        self.assertTrue(iss(0))
        self.assertTrue(iss(0.1))
        self.assertTrue(iss(7.5 - 23j))
        self.assertTrue(iss(np.float64(0.1)))

        self.assertFalse(iss(M1))
        self.assertFalse(iss(M2))
        self.assertFalse(iss(M1[:1, :1]))
        self.assertFalse(iss(np.arange(5)))

    def test_sca_integrate(self):
        """
        test special case aware integrate
        """
        x1, x2, x3 = xx = st.symb_vector('x1:4')

        f = sp.log(cos(x1))
        df = f.diff(x1)
        F = st.sca_integrate(df, x1)
        self.assertEqual(F, f)

        if 1:
            f = 5*x1
            df = f.diff(x1)
            F = st.sca_integrate(df, x1)
            self.assertEqual(F, f)

            f = cos(x1)*x1
            df = f.diff(x1)
            F = st.sca_integrate(df, x1)
            self.assertEqual(F, f)


# noinspection PyShadowingNames,PyPep8Naming,PySetFunctionToLiteral
class TestNumTools(unittest.TestCase):

    def setUp(self):
        n = 5
        self.ev = sp.randMatrix(n, 1, seed=1631)
        d = sp.diag(*self.ev)

        self.T = T = sp.randMatrix(n, n, seed=1632)
        assert not T.det() == 0
        self.M1 = T*d*T.inv()

        self.ev_sorted = list(self.ev)
        self.ev_sorted.sort(reverse=True)

        # #

        self.M2 = sp.Matrix([[0, 1], [-1, 0]])

    def test_sorted_eigenvalues(self):

        res1 = st.sorted_eigenvalues(self.M1)
        self.assertEqual(res1, self.ev_sorted)

        # imaginary unit
        I = sp.I

        res2 = st.sorted_eigenvalues(self.M2)
        self.assertTrue(I in res2)
        self.assertTrue(-I in res2)
        self.assertEqual(2, len(res2))

    def test_sorted_eigenvectors(self):

        V1 = st.sorted_eigenvector_matrix(self.M1)

        ev1 = st.sorted_eigenvalues(self.M1)
        self.assertEqual(len(ev1), V1.shape[1])

        for val, vect in lzip(ev1, st.col_split(V1)):
            res_vect = self.M1*vect - val*vect
            res = (res_vect.T*res_vect)[0]
            self.assertTrue(res < 1e-15)
            self.assertAlmostEqual( (vect.T*vect)[0] - 1, 0)

        V2 = st.sorted_eigenvector_matrix(self.M1, numpy=True)
        V3 = st.sorted_eigenvector_matrix(self.M1, numpy=True, increase=True)

        # quotients should be +-1
        res1 = np.abs( st.to_np(V1) / st.to_np(V2) ) - np.ones_like(V1)
        res2 = np.abs( st.to_np(V1) / st.to_np(V3[:, ::-1]) ) - np.ones_like(V1)

        self.assertTrue(np.max(np.abs(res1)) < 1e-5)
        self.assertTrue(np.max(np.abs(res2)) < 1e-5)


# noinspection PyShadowingNames,PyPep8Naming,PySetFunctionToLiteral
class RandNumberTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_rnd_number_tuples(self):
        x1, x2, x3 = xx = sp.symbols('x1:4')

        s = sum(xx)
        res_a1 = st.rnd_number_subs_tuples(s)
        self.assertTrue(isinstance(res_a1, list))
        self.assertEqual(len(res_a1), len(xx))

        c1 = [len(e)==2 and e[0].is_Symbol and st.is_number(e[1])
              for e in res_a1]

        self.assertTrue( all(c1) )

        t = sp.Symbol('t')
        f = sp.Function('f')(t)

        fdot = f.diff(t)
        fddot = f.diff(t, 2)

        ff = sp.Matrix([f, fdot, fddot, x1*x2])

        for i in range(100):
            res_b1 = st.rnd_number_subs_tuples(ff, seed=i)

            expct_b1_set = set([f, fdot, fddot, t, x1, x2])
            res_b1_atom_set = set( lzip(*res_b1)[0] )

            self.assertEqual(expct_b1_set, res_b1_atom_set)
            # highest order has to be returned first
            self.assertEqual(res_b1[0][0], fddot)
            self.assertEqual(res_b1[1][0], fdot)
            self.assertTrue( all( [st.is_number(e[1]) for e in res_b1] ) )

    def test_rnd_number_tuples2(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4')
        yy = st.symb_vector('y1:4')

        s = sum(xx)
        res_a1 = st.rnd_number_subs_tuples(s, seed=1)
        res_a2 = st.rnd_number_subs_tuples(s, seed=2)
        self.assertNotEqual(res_a1, res_a2)

        res_b1 = st.rnd_number_subs_tuples(s, seed=2)
        self.assertEqual(res_b1, res_a2)

        xxyy = xx + yy
        rnst1 = st.rnd_number_subs_tuples(xxyy)
        rnst2 = st.rnd_number_subs_tuples(xxyy, exclude=x1)
        rnst3 = st.rnd_number_subs_tuples(xxyy, exclude=[x1, x2])
        rnst4 = st.rnd_number_subs_tuples(xxyy, exclude=xx)
        symbols1 = xxyy.subs(rnst1).atoms(sp.Symbol)
        symbols2 = xxyy.subs(rnst2).atoms(sp.Symbol)
        symbols3 = xxyy.subs(rnst3).atoms(sp.Symbol)
        symbols4 = xxyy.subs(rnst4).atoms(sp.Symbol)

        self.assertEqual(symbols1, set())
        self.assertEqual(symbols2, set([x1]))
        self.assertEqual(symbols3, set([x1, x2]))
        self.assertEqual(symbols4, set([x1, x2, x3]))

        # this was a bug:
        rnst = st.rnd_number_subs_tuples(xxyy, prime=True, exclude=[x1, x2])
        self.assertEqual(xxyy.subs(rnst).atoms(sp.Symbol), set([x1, x2]))

    def test_rnd_number_tuples3(self):
        a, b = sp.symbols('a, b', commutative=False)

        term1 = a*b - b*a
        st.warnings.simplefilter("always")
        with st.warnings.catch_warnings(record=True) as cm:
            st.rnd_number_subs_tuples(term1)

        self.assertEqual(len(cm), 1)
        self.assertTrue('not commutative' in str(cm[0].message))

        with st.warnings.catch_warnings(record=True) as cm2:
            st.subs_random_numbers(term1)

        self.assertEqual(len(cm2), 1)
        self.assertTrue('not commutative' in str(cm2[0].message))

    def test_generic_rank1(self):
        x1, x2, x3 = xx = st.symb_vector('x1:4')

        M1 = sp.Matrix([[x1, 0], [0, x2]])
        M2 = sp.Matrix([[1, 0], [sin(x1)**2, sin(x1)**2 + cos(x1)**2 - 1]])  # singular
        M3 = sp.Matrix([[1, 0], [1, sin(x1)**50]])  # regular

        M4 = sp.Matrix([[1, 0, 0], [1, sin(x1)**50, 1], [0, 0, 0]])  # rank 2

        M5 = sp.Matrix([[-x2,   0, -x3],
                        [ x1, -x3,   0],
                        [  0,  x2,  x1]])

        M6 = sp.Matrix([[1, 0, 0],
                        [sin(x1)**2, sin(x1)**2 + cos(x1)**2 - 1, 0],
                        [0, sp.pi, sin(-3)**50]])  # rank 2

        M7 = st.row_stack(M6, [sp.sqrt(5)**-20, 2, 0])  # nonsquare, rank 3

        M8 = sp.diag(1, sin(3)**2 + cos(3)**2 - 1, sin(3)**30, sin(3)**150)

        # test for a specific bug
        xxdd = st.symb_vector('xdot1, xdot2, xddot1, xddot2, xdddot1')
        xdot1, xdot2, xddot1, xddot2, xdddot1 = xxdd

        M9 = sp.Matrix([[1.00000000000000, 1.0*xdot1, 1.00000000000000, 1.0*x1, 1.00000000000000, 0,
                         0, 0, 0, 0], [1.0*x2, 1.0*x1*x2, 1.0*x2, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1.0*xddot1, 1.00000000000000, 2.0*xdot1, 1.00000000000000, 1.0*x1,
                         1.00000000000000, 0, 0, 0],
                        [1.0*xdot2, 1.0*x1*xdot2 + 1.0*x2*xdot1, 1.0*x2 + 1.0*xdot2, 1.0*x1*x2,
                         1.0*x2, 0, 0, 0, 0, 0],
                        [0, 1.0*xdddot1, 0, 3.0*xddot1, 1.00000000000000, 3.0*xdot1,
                         1.00000000000000, 1.0*x1, 1.00000000000000, 0],
                        [1.0*xddot2, 1.0*x1*xddot2 + 1.0*x2*xddot1 + 2.0*xdot1*xdot2,
                         1.0*xddot2 + 2.0*xdot2, 2.0*x1*xdot2 + 2.0*x2*xdot1, 1.0*x2 + 2.0*xdot2,
                         1.0*x1*x2, 1.0*x2, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

        res1 = st.generic_rank(M1, seed=98682)
        self.assertEqual(res1, 2)

        res2 = st.generic_rank(M2)
        self.assertEqual(res2, 1)

        res3 = st.generic_rank(M3, seed=1814)
        self.assertEqual(res3, 2)

        self.assertEqual(st.generic_rank(M2, seed=1529), 1)
        self.assertEqual(st.generic_rank(M4, seed=1814), 2)

        self.assertEqual(st.generic_rank(M5, seed=1814), 2)
        self.assertEqual(st.generic_rank(M6, seed=1814), 2)
        self.assertEqual(st.generic_rank(M7, seed=1814), 3)
        self.assertEqual(st.generic_rank(M7.T, seed=1814), 3)
        self.assertEqual(st.generic_rank(M8, seed=1814), 3)

        # TODO: This should raise a warning
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            res = st.generic_rank(M9, seed=2051)

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertTrue("Float" in str(w[-1].message))
        # nevertheless result should be correct
        self.assertEqual(res, 6)

    def test_rationalize_all_numbers(self):

        xxdd = st.symb_vector('x1, x2, xdot1, xdot2, xddot1, xddot2, xdddot1')
        x1, x2, xdot1, xdot2, xddot1, xddot2, xdddot1 = xxdd

        M1 = sp.Matrix([[1.00000000000000, 1.0*xdot1, 1.00000000000000, 1.0*x1, 1.00000000000000, 0,
                         0, 0, 0, 0], [1.0*x2, 1.0*x1*x2, 1.0*x2, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1.0*xddot1, 1.00000000000000, 2.0*xdot1, 1.00000000000000, 1.0*x1,
                         1.00000000000000, 0, 0, 0],
                        [1.0*xdot2, 1.0*x1*xdot2 + 1.0*x2*xdot1, 1.0*x2 + 1.0*xdot2, 1.0*x1*x2,
                         1.0*x2, 0, 0, 0, 0, 0],
                        [0, 1.0*xdddot1, 0, 3.0*xddot1, 1.00000000000000, 3.0*xdot1,
                         1.00000000000000, 1.0*x1, 1.00000000000000, 0],
                        [1.0*xddot2, 1.0*x1*xddot2 + 1.0*x2*xddot1 + 2.0*xdot1*xdot2,
                         1.0*xddot2 + 2.0*xdot2, 2.0*x1*xdot2 + 2.0*x2*xdot1, 1.0*x2 + 2.0*xdot2,
                         1.0*x1*x2, 1.0*x2, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, np.pi, 0, 0, 0, 0, 0]])
        types1 = [type(a) for a in M1.atoms(sp.Number)]
        self.assertTrue(sp.Float in types1)

        M2 = st.rationalize_all_numbers(M1)
        types2 = [type(a) for a in M2.atoms(sp.Number)]
        self.assertFalse(sp.Float in types2)

    @uth.skip_slow
    def test_generic_rank2(self):
        import pickle
        path = make_abspath('test_data', 'rank_test_matrices.pcl')
        with open(path, 'rb') as pfile:
            matrix_list = pickle.load(pfile)

        N = len(matrix_list)
        for i, m in enumerate(matrix_list):
            print("%i / %i" %(i, N))
            r1 = m.srnp.rank()
            r2 = st.generic_rank(m)

            self.assertEqual(r1, r2)


# noinspection PyShadowingNames,PyPep8Naming,PySetFunctionToLiteral
class TestTrajectoryPlanning(unittest.TestCase):

    def setUp(self):
        pass

    def test_transpoly(self):

        x, y = sp.symbols("x, y")
        res1 = st.trans_poly(x, 0, (0, 0), (2, 1))
        self.assertEqual(res1, x/2)

        res2 = st.trans_poly(x, 1, (0, 0, 1), (2, 1, 1))
        self.assertEqual(res2, x**3/4 - 3*x**2/4 + x)

    def test_condition_poly(self):
        x, y = sp.symbols("x, y")

        res1 = st.condition_poly(x, (0, 0, 1), (2, 1, 1))
        self.assertEqual(res1, x**3/4 - 3*x**2/4 + x)

        res2 = st.condition_poly(x, (0, 0), (2, -4, 0, 3))

        self.assertEqual(res2.subs(x, 0), 0)
        self.assertEqual(res2.subs(x, 2), -4)
        self.assertEqual(res2.diff(x).subs(x, 2), 0)
        self.assertEqual(res2.diff(x, x).subs(x, 2), 3)

        # now only with one condition
        res3 = st.condition_poly(x, (0, 1.75))
        self.assertEqual(res3.subs(x, 0), 1.75)

    def test_create_piecewise(self):
        t, x = sp.symbols('t, x')
        interface_points1 = [0, 4]
        expr1 = st.create_piecewise(t, interface_points1, [-1, x, -13])

        self.assertEqual(expr1.subs(t, -3), -1)
        self.assertEqual(expr1.subs(t, 0), x)
        self.assertEqual(expr1.subs(t, 3), x)
        self.assertEqual(expr1.subs(t, 4), x)
        self.assertEqual(expr1.subs(t, 4.00000001), -13)
        self.assertEqual(expr1.subs(t, 10**100), -13)

        interface_points2 = [0, 4, 8, 12]
        expr1 = st.create_piecewise(t, interface_points2, [-1, x, x**2, x**3, -13])

        self.assertEqual(expr1.subs(t, -2), -1)
        self.assertEqual(expr1.subs(t, 0), x)
        self.assertEqual(expr1.subs(t, 4), x**2)
        self.assertEqual(expr1.subs(t, 7), x**2)
        self.assertEqual(expr1.subs(t, 8), x**3)
        self.assertEqual(expr1.subs(t, 9), x**3)
        self.assertEqual(expr1.subs(t, 12), x**3)
        self.assertEqual(expr1.subs(t, 12.00000001), -13)
        self.assertEqual(expr1.subs(t, 1e50), -13)

    def test_create_piecewise_poly(self):
        x, t = sp.symbols("x, t")

        conditions = [(0, 0, 0), # t= 0: x=0, x_dot=0
                      (2, 1), # t= 2: x=1, x_dot=<not defined>

                      (3, 1, 0, 0 ), # t= 2: x=1, x_dot=0, x_ddot=0
                      (5, 2, 0, 0 ), # t= 2: x=1, x_dot=0, x_ddot=0
                        # smooth curve finished
                        ]

        res1 = st.create_piecewise_poly(t, *conditions)

        self.assertEqual(res1.func(0), 0)
        self.assertEqual(res1.func(2), 1)
        self.assertEqual(res1.func(3), 1)
        self.assertEqual(res1.func(5), 2)
        self.assertEqual(res1.expr.diff(t, 2).subs(t, 5), 0)

    def test_do_laplace_deriv(self):
        t, s = sp.symbols('t, s')
        x1, x2, x3 = xx = st.symb_vector('x1:4')

        x1dot, x2dot, x3dot = st.time_deriv(xx, xx)
        x1ddot, x2ddot, x3ddot = st.time_deriv(xx, xx, order=2)

        expr1 = 5
        expr2 = 5*s*t**2 - 7*t + 2
        expr3 = 1*s**2*x1 - 7*s*x2*t + 2

        res = st.do_laplace_deriv(expr1, s, t)
        ex_res = 5
        self.assertEqual(res, ex_res)

        res = st.do_laplace_deriv(expr2, s, t)
        ex_res = 10*t - 7*t + 2
        self.assertEqual(res, ex_res)

        res = st.do_laplace_deriv(expr3, s, t)
        ex_res = -7 * x2 + 2
        self.assertEqual(res, ex_res)

        res = st.do_laplace_deriv(expr3, s, t, tds=xx)
        ex_res = x1ddot - 7 * x2 + - 7*x2dot*t +  2
        self.assertEqual(res, ex_res)


# noinspection PyShadowingNames,PyPep8Naming,PySetFunctionToLiteral
class TestControlMethods1(unittest.TestCase):

    def setUp(self):
        pass

    def test_kalman_matrix(self):

        k, J, R, L = sp.symbols('k, J, R, L')
        A = sp.Matrix([[0, 1, 0], [0, 0, k/J], [0, -k/L, -R/L]])
        B = sp.Matrix([0, 0, 1/L])

        Qref = sp.Matrix([[0, 0, k/L/J], [0, k/L/J, -k*R/J/L**2 ],
                          [1/L, -R/L**2, -k**2/J/L**2 + R**2/L**3 ]])
        Q = st.kalman_matrix(A, B)

        self.assertEqual(Q, Qref)

    def test_nl_cont_matrix(self):

        # for simplicity test with a linear example
        k, J, R, L = sp.symbols('k, J, R, L')
        A = sp.Matrix([[0, 1, 0], [0, 0, k/J], [0, -k/L, -R/L]])
        B = sp.Matrix([0, 0, 1/L])

        Qref = sp.Matrix([[0, 0, k/L/J], [0, k/L/J, -k*R/J/L**2 ],
                          [1/L, -R/L**2, -k**2/J/L**2 + R**2/L**3 ]])

        xx = st.symb_vector("x1:4")
        ff = A*xx
        gg = B

        Qnl = st.nl_cont_matrix(ff, gg, xx)

        self.assertEqual(Qnl, Qref)

    def test_siso_place(self):

        n = 6
        A = sp.randMatrix(n, n, seed=1648, min=-10, max=10)
        b = sp.randMatrix(n, 1, seed=1649, min=-10, max=10)
        ev = np.sort(np.random.random(n) * 10)

        f = st.siso_place(A, b, ev)

        A2 = st.to_np(A + b*f.T)
        ev2 = np.sort( np.linalg.eigvals(A2) )

        diff = np.sum(np.abs((ev - ev2)/ev))
        self.assertTrue(diff < 1e-6)

    def test_siso_place2(self):

        n = 4
        A = sp.randMatrix(n, n, seed=1648, min=-10, max=10)
        b = sp.randMatrix(n, 1, seed=1649, min=-10, max=10)

        omega = np.pi*2/2.0
        ev = np.sort([1j*omega, -1j*omega, -2, -3])
        f = st.siso_place(A, b, ev)

        A2 = st.to_np(A + b*f.T)
        ev2 = np.sort( np.linalg.eigvals(A2) )
        diff = np.sum(np.abs((ev - ev2)/ev))

        self.assertTrue(diff < 1e-6)

    @uth.optional_dependency
    def test_sympy_to_tf(self):
        s = sp.Symbol("s")
        P1 = 1
        P2 = 1/(3*s + 1.5)
        P3 = s
        P4 = s*(0.8*s**5 - 7)/(13*s**7 + s**2 + 21*s - sp.pi)

        G1 = st.sympy_to_tf(P1)
        G2 = st.sympy_to_tf(P2)
        G3 = st.sympy_to_tf(P3)
        G4 = st.sympy_to_tf(P4)

        def tf_eq(tf1, tf2, atol=0):
            num = (tf1 - tf2).num[0][0]
            return np.allclose(num, 0, atol=atol)

        G2_ref = control.tf([1], [3, 1.5])
        self.assertTrue(tf_eq(G2, G2_ref))


def main():
    uth.smart_run_tests_in_ns(globals())


if __name__ == '__main__':
    main()
