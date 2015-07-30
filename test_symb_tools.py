# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 11:35:00 2014

@author: Carsten Knoll
"""

import unittest
import sys

import sympy as sp
from sympy import sin, cos, exp

import numpy as np
import scipy as sc
import scipy.integrate

import symb_tools as st
import inspect
from IPython import embed as IPS


if 'all' in sys.argv:
    FLAG_all = True
    sys.argv.remove('all')
else:
    FLAG_all = False


# own decorator for skipping slow tests
def skip_slow(func):
    return unittest.skipUnless(FLAG_all, 'skipping slow test')(func)


class InteractiveConvenienceTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_no_IPS_call(self):
        srclines = inspect.getsourcelines(st)[0]
        def filter_func(tup):
            idx, line = tup
            return 'IPS()' in line and not line.strip()[0] == '#'

        res = filter(filter_func, enumerate(srclines, 1))

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

    def test_subz(self):
        x1, x2, x3 = xx = sp.Matrix(sp.symbols("x1, x2, x3"))
        y1, y2, y3 = yy = sp.symbols("y1, y2, y3")

        a = x1 + 7*x2*x3
        M1 = sp.Matrix([x2, x1*x2, x3**2])
        M2 = sp.ImmutableDenseMatrix(M1)

        self.assertEqual(x1.subs(zip(xx, yy)), x1.subz(xx, yy))
        self.assertEqual(a.subs(zip(xx, yy)), a.subz(xx, yy))
        self.assertEqual(M1.subs(zip(xx, yy)), M1.subz(xx, yy))
        self.assertEqual(M2.subs(zip(xx, yy)), M2.subz(xx, yy))

    def test_subz0(self):
        x1, x2, x3 = xx = sp.Matrix(sp.symbols("x1, x2, x3"))
        y1, y2, y3 = yy = sp.symbols("y1, y2, y3")

        XX = (x1, x2)

        a = x1 + 7*x2*x3
        M1 = sp.Matrix([x2, x1*x2, x3**2])
        M2 = sp.ImmutableDenseMatrix(M1)

        self.assertEqual(x1.subs(st.zip0(XX)), x1.subz0(XX))
        self.assertEqual(a.subs(st.zip0(XX)), a.subz0(XX))
        self.assertEqual(M1.subs(st.zip0(XX)), M1.subz0(XX))
        self.assertEqual(M2.subs(st.zip0(XX)), M2.subz0(XX))


class SymbToolsTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_depends_on_t1(self):
        a, b, t = sp.symbols("a, b, t")

        res1 = st.depends_on_t(a+b, t, [])
        self.assertEqual(res1, False)

        res2 = st.depends_on_t(a + b, t, [a,])
        self.assertEqual(res2, True)

        res3 = st.depends_on_t(a(t) + b, t, [])
        self.assertEqual(res3, True)

        res4 = st.depends_on_t(a(t) + b, t, [b])
        self.assertEqual(res4, True)

        res5 = st.depends_on_t(t, t, [])
        self.assertEqual(res5, True)

        adot = st.perform_time_derivative(a, [a])
        res5 = st.depends_on_t(adot, t, [])
        self.assertEqual(res5, True)

        x1, x2 = xx = sp.symbols("x1, x2")
        x1dot = st.perform_time_derivative(x1, xx)
        self.assertTrue(st.depends_on_t(x1dot, t))

        y1, y2 = yy = sp.Matrix(sp.symbols("y1, y2", commutative=False))
        yydot = st.perform_time_derivative(yy, yy, order=1, commutative=False)
        self.assertTrue(st.depends_on_t(yydot, t))

    def test_symbs_to_func(self):
        a, b, t = sp.symbols("a, b, t")
        x = a + b
        #M = sp.Matrix([x, t, a**2])

        f_x = st.symbs_to_func(x, [a, b], t)
        self.assertEquals(str(f_x), "a(t) + b(t)")

    def test_perform_time_deriv1(self):
        a, b, t = sp.symbols("a, b, t")
        f1 = a**2 + 10*sin(b)
        f1_dot = st.perform_time_derivative(f1, (a, b))
        self.assertEquals(str(f1_dot), "2*a*adot + 10*bdot*cos(b)")

        f2 = sin(a)
        f2_dot2 = st.perform_time_derivative(f2, [a, b], order=2)
        f2_dot = st.perform_time_derivative(f2, [a])
        f2_ddot = st.perform_time_derivative(f2_dot, [a, sp.Symbol('adot')])
        self.assertEqual(f2_ddot, f2_dot2)

    def test_perform_time_deriv2(self):
        """
        test matrix compatibility
        """

        a, b, t = sp.symbols("a, b, t")
        x, y = sp.symbols("x, y")

        adot, bdot = st.perform_time_derivative( sp.Matrix([a, b]), (a, b) )

        A = sp.Matrix([sin(a), exp(a*b), -t**2*7*0, x + y]).reshape(2, 2)
        A_dot = st.perform_time_derivative(A, (a, b))

        A_dot_manual = \
            sp.Matrix([adot*cos(a), b*adot*exp(a*b) + a*bdot*exp(a*b),
                       -t**2*7*0, 0]).reshape(2, 2)

        self.assertEqual(A_dot.expand(), A_dot_manual)

    def test_perform_time_deriv3(self):
        """
        test to provide deriv_symbols
        """

        a, b, adot, bdot, addot, bddot = sp.symbols("a, b, adot, bdot, addot, bddot")

        f1 = 8*a*b**2
        f1d = st.perform_time_derivative(f1, (a, b), (adot, bdot))
        self.assertEquals(f1d, 8*adot*b**2 + 16*a*bdot*b)

        f1d_2 = st.perform_time_derivative(f1, (a, b), (adot, bdot)+ (addot, bddot),
                                           order=2)

        f1d_2_altntv = st.perform_time_derivative(f1d, (a, b, adot, bdot),
                                                 (adot, bdot)+ (addot, bddot) )
        self.assertEquals(f1d_2, f1d_2_altntv)

    def test_perform_time_deriv4(self):
        # test higher order derivatives

        a, b = sp.symbols("a, b")

        f1 = 8*a*b**2

        res_a1 = st.perform_time_derivative(f1, (a, b), order=5)

        a_str = 'a adot addot adddot addddot a_d5'
        b_str = a_str.replace('a', 'b')

        expected_symbol_names = a_str.split() + b_str.split()

        res_list = [res_a1.has(sp.Symbol(e)) for e in expected_symbol_names]

        self.assertTrue( all(res_list) )

        l1 = len( res_a1.atoms(sp.Symbol) )
        self.assertEqual(len(expected_symbol_names), l1)

    def test_perform_time_deriv5(self):
        # test numbered symbols

        x1, x2 = xx = sp.symbols("x1, x_2")


        res_a1 = st.perform_time_derivative(x1, xx)
        self.assertEqual(str(res_a1), 'xdot1')

        res_a2 = st.perform_time_derivative(x1, xx, order=2)
        self.assertEqual(str(res_a2), 'xddot1')

        res_a3 = st.perform_time_derivative(x1, xx, order=3)
        self.assertEqual(str(res_a3), 'xdddot1')

        res_a4 = st.perform_time_derivative(x1, xx, order=4)
        self.assertEqual(str(res_a4), 'xddddot1')

        # FIXME:
        res_a5 = st.perform_time_derivative(x1, xx, order=5)
        #self.assertEqual(str(res_a5), 'x1_d5')


        res_b1 = st.perform_time_derivative(x2, xx)
        self.assertEqual(str(res_b1), 'x_dot2')

        res_b2 = st.perform_time_derivative(x2, xx, order=2)
        self.assertEqual(str(res_b2), 'x_ddot2')

        res_b3 = st.perform_time_derivative(x2, xx, order=3)
        self.assertEqual(str(res_b3), 'x_dddot2')

        res_b4 = st.perform_time_derivative(x2, xx, order=4)
        self.assertEqual(str(res_b4), 'x_ddddot2')

        # FIXME
        res_b5 = st.perform_time_derivative(x2, xx, order=5)
        #self.assertEqual(str(res_b5), 'x_2_d5')

    @unittest.expectedFailure
    def test_perform_time_deriv5f(self):
        # test numbered symbols

        x1, x2 = xx = sp.symbols("x1, x_2")

        # TODO: These two assertions should pass
        # Then the above FIXME-issues can be resolved and this test is obsolete

        res_a5 = st.perform_time_derivative(x1, xx, order=5)
        self.assertEqual(str(res_a5), 'x1_d5')

        res_b5 = st.perform_time_derivative(x2, xx, order=5)
        self.assertEqual(str(res_b5), 'x_2_d5')

    def test_perform_time_derivative7(self):
        a, b, t = sp.symbols("a, b, t", commutative=False)

        f1 = sp.Function('f1')(t)
        f2 = sp.Function('f2')(t)

        res1 = st.perform_time_derivative(f1, [a])
        self.assertEqual(res1, f1.diff(t))

        res2 = st.perform_time_derivative(f1, [])
        self.assertEqual(res2, f1.diff(t))

        res3 = st.perform_time_derivative(a*f1, [a, b])
        adot = st.perform_time_derivative(a, [a])
        self.assertEqual(res3, a*f1.diff(t) + adot*f1)

    def test_perform_time_derivative8(self):

        y1, y2 = yy = sp.Matrix( sp.symbols('y1, y2', commutative=False) )

        ydot1 = st.perform_time_derivative(y1, yy)
        ydot2 = st.perform_time_derivative(y2, yy)

        yddot1 = st.perform_time_derivative(y1, yy, order=2)
        ydddot1 = st.perform_time_derivative(y1, yy, order=3)

        res1 = st.perform_time_derivative(ydot1, yy)
        self.assertEqual(res1, yddot1)

        res2 = st.perform_time_derivative(ydot1, yy, order=2)
        self.assertEqual(res2, ydddot1)

        res3 = st.perform_time_derivative(yddot1, yy)
        self.assertEqual(res3, ydddot1)

    def test_match_symbols_by_name(self):
        a, b, c = abc0 = sp.symbols('a5, b, c', real=True)
        a1, b1, c1 = abc1 = sp.symbols('a5, b, c')

        self.assertFalse(a == a1 or b == b1 or c == c1)

        abc2 = st.match_symbols_by_name(abc0, abc1)
        self.assertEquals(abc0, tuple(abc2))

        input3 = [a1, b, "c", "x"]
        res = st.match_symbols_by_name(abc0, input3, strict=False)
        self.assertEquals(abc0, tuple(res))

        with self.assertRaises(ValueError) as cm:
            res = st.match_symbols_by_name(abc0, input3)  # implies strict=True

        self.assertTrue('symbol x' in cm.exception.message)

        self.assertEquals(abc0, tuple(res))

        r = st.match_symbols_by_name(abc0, 'a5')
        self.assertEquals(len(r), 1)
        self.assertEquals(r[0], a)

        # test expression as first argument

        expr = a*b**c + 5
        r3 = st.match_symbols_by_name(expr, ['c', 'a5'])
        self.assertEquals(r3, [c, a])

    def test_symbs_to_func(self):
        a, b, t = sp.symbols("a, b, t")
        x, y = sp.symbols("x, y")

        at = sp.Function('a')(t)
        bt = sp.Function('b')(t)

        xt = sp.Function('x')(t)
        yt = sp.Function('y')(t)

        M = sp.Matrix([x, y])
        Mt = sp.Matrix([xt, yt])

        expr1 = a**2 + b**2
        f1 = st.symbs_to_func(expr1, (a,b), t)
        self.assertEqual(f1, at**2 + bt**2)

        at2 = st.symbs_to_func(a, arg=t)
        self.assertEqual(at, at2)

        # Matrix
        Mt2 = st.symbs_to_func(M, arg=t)
        self.assertEqual(Mt, Mt2)

        # TODO: assert raises

    def test_poly_expr_coeffs(self):
        a, b, c, d = sp.symbols("a, b, c, d")
        x, y = sp.symbols("x, y")

        p1 = a*x + b*x**2 + c*y - d + 2*a*x*y
        sample_solution = [((0, 0), -d), ((1, 0), a), ((2, 0), b),
                           ((0, 1), c), ((1, 1), 2*a), ((0, 2), 0)]

        # coeff-dict
        cd1 = st.poly_expr_coeffs(p1, [x,y])
        self.assertEqual(cd1, dict(sample_solution))

    def test_get_diffterms(self):
        x1, x2, x3, = xx = sp.symbols('x1:4')
        res2 = st.get_diffterms(xx, 2)
        expected_res2 = [(x1, x1), (x1, x2), (x1, x3), (x2, x2),
                         (x2, x3), (x3, x3)]

        expected_indices2 = [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0),
                             (0, 1, 1), (0, 0, 2)]
        self.assertEqual(res2, expected_res2)

        res3 = st.get_diffterms(xx, 3)
        res23 = st.get_diffterms(xx, (2, 3))

        self.assertEqual(res23, res2 + res3)

        res2b, indices2 = st.get_diffterms(xx, 2, order_list=True)
        self.assertEqual(res2b, res2)
        self.assertEqual(expected_indices2, indices2)

        res3b, indices3 = st.get_diffterms(xx, 3, order_list=True)
        res23b, indices23 = st.get_diffterms(xx, (2, 3), order_list=True)

        self.assertEqual(res23b, res2b + res3b)
        self.assertEqual(indices23, indices2 + indices3)

    def test_get_diffterms_bug1(self):
        xx = sp.symbols('x, y')
        res, orders = st.get_diffterms(xx, 2, order_list=True)
        self.assertEqual(orders, [(2, 0), (1, 1), (0, 2)])

    def test_monomial_from_signature(self):
        x1, x2, x3, = xx = sp.symbols('x1:4')

        s1 = (0, 0, 0)
        p1 = st.monomial_from_signature(s1, xx)
        self.assertEqual(p1, 1)

        s2 = (3, 2, 1)
        p2 = st.monomial_from_signature(s2, xx)
        self.assertEqual(p2, x1**3 * x2**2 * x3 )

    def test_zip0(self):
        aa = sp.symbols('a1:5')
        bb = sp.symbols('b1:4')
        xx = sp.symbols('x1:3')

        s1 = sum(aa)
        self.assertEqual(s1.subs(st.zip0(aa)), 0)
        self.assertEqual(s1.subs(st.zip0(aa, arg=3.5)), 3.5*len(aa))
        self.assertEqual(s1.subs(st.zip0(aa, arg=xx[0])), xx[0]*len(aa))

        s2 = s1 + sum(bb) + sum(xx)
        self.assertEqual(s1.subs(st.zip0(aa, bb, xx)), 0)
        t2 = s2.subs(st.zip0(aa, bb, xx, arg=-2.1))
        self.assertEqual( t2, -2.1*len(aa + bb + xx) )

    def test_is_number(self):
        x1, x2, x3 = xx = sp.symbols('x1:4')

        self.assertTrue(st.is_number(x1/x1))
        self.assertTrue(st.is_number(1))
        self.assertTrue(st.is_number(3.4))
        self.assertTrue(st.is_number(-10.0000001))

        self.assertFalse(st.is_number(x1))
        self.assertFalse(st.is_number(x1/x2))
        self.assertFalse(st.is_number(float('nan')))
        self.assertFalse(st.is_number(float('inf')))
        self.assertFalse(st.is_number(-float('inf')))

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

        res_b1 = st.rnd_number_subs_tuples(ff)

        expct_b1_set = set([f, fdot, fddot, t, x1, x2])
        res_b1_atom_set = set( zip(*res_b1)[0] )

        self.assertEqual(expct_b1_set, res_b1_atom_set)
        self.assertEqual(res_b1[0][0], fddot)
        self.assertEqual(res_b1[1][0], fdot)
        self.assertTrue( all( [st.is_number(e[1]) for e in res_b1] ) )

    def test_rnd_number_tuples2(self):
        x1, x2, x3 = xx = sp.symbols('x1:4')

        s = sum(xx)
        res_a1 = st.rnd_number_subs_tuples(s, seed=1)
        res_a2 = st.rnd_number_subs_tuples(s, seed=2)
        self.assertNotEqual(res_a1, res_a2)

        res_b1 = st.rnd_number_subs_tuples(s, seed=2)
        self.assertEqual(res_b1, res_a2)

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

        udot1, udot2 = uudot = st.perform_time_derivative(uu, uu, order=1)
        uddot1, uddot2 = st.perform_time_derivative(uu, uu, order=2)

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

    def test_deriv_2nd_order_chain_rule(self):
        a, b, x = sp.symbols('a, b, x')

        f1 = a**3 + a*b**2 + 7*a*b
        f2 = -2*a**2 + b*a*b**2/(2+a**2 * b**2) + 12*a*b
        f3 = -3*a**2 + b*a*b**2 + 7*a*b/(2+a**2 * b**2)


        f = f1
        aa = sp.cos(3*x)
        bb = sp.exp(5*x)

        ab_subs = [(a, aa), (b, bb)]
        fab = f.subs(ab_subs)
        fab_d2 = fab.diff(x, 2)

        res1 = st.deriv_2nd_order_chain_rule(f, (a,b), [aa, bb], x)

        res1a = sp.simplify(res1 - fab_d2)
        self.assertEqual(res1a, 0)

        # multiple functions

        ff = sp.Matrix([f1, f2, f3])

        ff_ab_d2 = ff.subs(ab_subs).diff(x, 2)

        res2 = st.deriv_2nd_order_chain_rule(ff, (a, b), [aa, bb], x)
        res2a = sp.expand(res2 - ff_ab_d2)
        self.assertEqual(res2a, res2a*0)

    def test_update_cse(self):
        a, b, c, x1, x2, x3 = sp.symbols('a, b, c, x1, x2, x3')

        L1 = [(x1, a + b), (x2, x1*b**2)]
        new_subs1 = [(b, 2)]
        new_subs2 = [(a, b + 5), (b, 3),]

        res1_exp = [(x1, a + 2), (x2, (a + 2)*4)]
        res2_exp = [(x1, 11), (x2, 99)]

        res1 = st.update_cse(L1, new_subs1)
        res2 = st.update_cse(L1, new_subs2)

        self.assertEqual(res1, res1_exp)
        self.assertEqual(res2, res2_exp)

    def test_solve_linear_system1(self):
        M = sp.randMatrix(3, 8, seed=1131)
        xx = sp.Matrix(sp.symbols('x1:9'))
        bb = sp.Matrix(sp.symbols('b1:4'))

        eqns = M*xx + bb

        sol1 = sp.solve(eqns, xx)
        sol2 = st.solve_linear_system(eqns, xx)

        self.assertEqual(xx.subs(sol1) - xx.subs(sol2), xx*0)

        # symbolic coefficient matrix
        # this is (currently) not possible with sp.solve
        # (additionally add a zero line)
        M2 = st.symbMatrix(4, 8)
        bb2 = sp.Matrix(sp.symbols('b1:5'))
        eqns = M2*xx + bb2

        eqns[-1, :] *= 0
        sol3 = st.solve_linear_system(eqns, xx)

        res = eqns.subs(sol3)
        res.simplify()

        self.assertEqual(res, res*0)

    def test_solve_linear_system2(self):
        par = sp.symbols('phi0, phi1, phi2, phidot0, phidot1, phidot2, phiddot0, phiddot1,'
                         'phiddot2, m0, m1, m2, l0, l1, l2, J0, J1, J2, L1, L2')

        phi0, phi1, phi2, phidot0, phidot1, phidot2, phiddot0, phiddot1, \
        phiddot2, m0, m1, m2, l0, l1, l2, J0, J1, J2, L1, L2 = par

        k1, k2, k3, k4, k5, k6 = kk = sp.symbols('k1:7')

        eqns = sp.Matrix([[L1*l0*m2*sin(phi1) + k1*(-2*l2*m2*phidot2*(L1*sin(phi1 - phi2) -
                           l0*sin(phi2))*(J1 + L1**2*m2 + L1*l0*m2*cos(phi1) +
                           L1*l2*m2*cos(phi1 - phi2) + l0*l1*m1*cos(phi1) + l1**2*m1) -
                           2*phidot1*(L1*l0*m2*sin(phi1) + L1*l2*m2*sin(phi1 - phi2) +
                           l0*l1*m1*sin(phi1))*(J2 + L1*l2*m2*cos(phi1 - phi2) + l0*l2*m2*cos(phi2) +
                           l2**2*m2) - 2*(-L1*l0*m2*phidot1*sin(phi1) - L1*l2*m2*(phidot1 -
                           phidot2)*sin(phi1 - phi2) - l0*l1*m1*phidot1*sin(phi1))*(J2 + L1*l2*m2*cos(phi1 - phi2) +
                           l0*l2*m2*cos(phi2) + l2**2*m2)) - k2*(-2*l2*m2*phidot2*(L1*sin(phi1 - phi2) -
                           l0*sin(phi2))*(J0 + l0**2*m0 + l0*m1*(l0 + l1*cos(phi1)) + l0*m2*(L1*cos(phi1) +
                           l0 + l2*cos(phi2))) - 2*(-l0*l1*m1*phidot1*sin(phi1) + l0*m2*(-L1*phidot1*sin(phi1) - l2*phidot2*sin(phi2)))*(J2 + L1*l2*m2*cos(phi1 - phi2) + l0*l2*m2*cos(phi2) + l2**2*m2)) + k4*(J1 + L1**2*m2 + L1*l0*m2*cos(phi1) + L1*l2*m2*cos(phi1 - phi2) + l0*l1*m1*cos(phi1) + l1**2*m1) - k5*(J0 + l0**2*m0 + l0*m1*(l0 + l1*cos(phi1)) + l0*m2*(L1*cos(phi1) + l0 + l2*cos(phi2))) + l0*l1*m1*sin(phi1)],
                          [-2*k1*(-L1*l2*m2*(phidot1 - phidot2)*sin(phi1 - phi2) - l0*l2*m2*phidot2*sin(phi2))*(J2 + L1*l2*m2*cos(phi1 - phi2) + l0*l2*m2*cos(phi2) + l2**2*m2) - k3*(-2*l2*m2*phidot2*(L1*sin(phi1 - phi2) - l0*sin(phi2))*(J0 + l0**2*m0 + l0*m1*(l0 + l1*cos(phi1)) + l0*m2*(L1*cos(phi1) + l0 + l2*cos(phi2))) - 2*(-l0*l1*m1*phidot1*sin(phi1) + l0*m2*(-L1*phidot1*sin(phi1) - l2*phidot2*sin(phi2)))*(J2 + L1*l2*m2*cos(phi1 - phi2) + l0*l2*m2*cos(phi2) + l2**2*m2)) + k4*(J2 + L1*l2*m2*cos(phi1 - phi2) + l0*l2*m2*cos(phi2) + l2**2*m2) - k6*(J0 + l0**2*m0 + l0*m1*(l0 + l1*cos(phi1)) + l0*m2*(L1*cos(phi1) + l0 + l2*cos(phi2))) + l0*l2*m2*sin(phi2)]])

        sol_subs = st.solve_linear_system(eqns, kk)

        res = eqns.subs(sol_subs)
        res.simplify()

        self.assertEqual(res, res*0)
        #IPS()


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
        self.assertEquals(res1.diff(t), rhs1.subs(x1, res1))

        res2 = st.solve_scalar_ode_1sto(rhs2, x1, t)
        self.assertEquals(res2.diff(t), rhs2.subs(x1, res2))

        res3, iv3 = st.solve_scalar_ode_1sto(rhs3, x1, t, return_iv=True)
        self.assertEquals(res3.diff(t), rhs3.subs(x1, res3))
        self.assertEquals(res3, iv3*exp(t))

        res5 = st.solve_scalar_ode_1sto(rhs5, x1, t)
        test_difference5 = res5.diff(t) - rhs5.subs(x1, res5)
        self.assertEquals(test_difference5.expand(), 0)

        res6 = st.solve_scalar_ode_1sto(rhs6, x1, t)
        self.assertEquals(res6.diff(t), rhs6.subs(x1, res6).expand())

    @skip_slow
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
        vf1_sol = vf1.subs(zip(xx[:-1], res1))
        self.assertEqual(fp, t)
        self.assertEqual(res1.diff(t), vf1_sol)

        res2, fp, iv2 = st.calc_flow_from_vectorfield(vf2, xx, flow_parameter=t)
        vf2_sol = vf2.subs(zip(xx[:-1], res2))
        self.assertEqual(fp, t)
        self.assertEqual(res2.diff(t), vf2_sol)

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

        par_vals = zip(pp, [v1, v2, v3, v4])

        f = A*xx
        G = B

        fxu = (f + G*uu).subs(par_vals)

        # some random initial values
        x0 = st.to_np( sp.randMatrix(len(xx), 1, -10, 10, seed=706) ).squeeze()

        # create the model and the rhs-function
        mod = st.SimulationModel(f, G, xx, par_vals)
        rhs0 = mod.create_simfunction()

        res0_1 = rhs0(x0, 0)
        dres0_1 = st.to_np(fxu.subs(zip(xx, x0) + st.zip0(uu))).squeeze()

        bin_res01 = np.isclose(res0_1, dres0_1)  # binary array
        self.assertTrue( np.all(bin_res01) )

        # difference should be [0, 0, ..., 0]
        self.assertFalse( np.any(rhs0(x0, 0) - rhs0(x0, 3.7) ) )

        # simulate
        tt = np.linspace(0, 0.5, 100)  # simulation should be short due to instability
        res1 = sc.integrate.odeint(rhs0, x0, tt)

        # proof calculation
        # x(t) = x0*exp(A*t)
        Anum = st.to_np(A.subs(par_vals))
        Bnum = st.to_np(G.subs(par_vals))
        xt = [ np.dot( sc.linalg.expm(Anum*T), x0 ) for T in tt ]
        xt = np.array(xt)

        bin_res1 = np.isclose(res1, xt)  # binary array
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

        with self.assertRaises(TypeError) as cm:
            mod2.create_simfunction(input_function=des_input_func_scalar)

        rhs3 = mod2.create_simfunction(input_function=des_input_func_vec)
        res3_0 = rhs3(x0, 0)

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
        res3 = f3(-3.1, 4) == r_[-6.2, 9, 4]
        self.assertTrue(res3.all())

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


    def test_reformulate_Integral(self):
        t = sp.Symbol('t')
        c = sp.Symbol('c')
        F = sp.Function('F')
        x = sp.Function('x')(t)
        a = sp.Function('a')

        i1 = sp.Integral(F(t), t)
        j1 = st.reformulate_integral_args(i1)
        self.assertEquals(j1.subs(t, 0).doit(), 0)

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

class SymbToolsTest3(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_symbols_by_name(self):
        c1, C1, x, a, t, Y = sp.symbols('c1, C1, x, a, t, Y')
        F = sp.Function('F')

        expr1 = c1*(C1+x**x)/(sp.sin(a*t))
        expr2 = sp.Matrix([sp.Integral(F(x), x)*sp.sin(a*t) - \
                           1/F(x).diff(x)*C1*Y])

        res1 = st.get_symbols_by_name(expr1, 'c1')
        self.assertEquals(res1, c1)
        res2 = st.get_symbols_by_name(expr1, 'C1')
        self.assertEquals(res2, C1)
        res3 = st.get_symbols_by_name(expr1, *'c1 x a'.split())
        self.assertEquals(res3, [c1, x, a])


        self.assertRaises(ValueError, st.get_symbols_by_name, expr1, 'Y')
        self.assertRaises(ValueError, st.get_symbols_by_name, expr1, 'c1', 'Y')

        res4 = st.get_symbols_by_name(expr2, 'Y')
        self.assertEquals(res4, Y)
        res5 = st.get_symbols_by_name(expr2, 'C1')
        self.assertEquals(res5, C1)
        res6 = st.get_symbols_by_name(expr2, *'C1 x a'.split())
        self.assertEquals(res6, [C1, x, a])

    def test_difforder_attribute(self):
        x1 = sp.Symbol('x1')
        xdot1 = st.perform_time_derivative(x1, [x1], order=4)
        self.assertEquals(xdot1.difforder, 4)

    def test_user_attributes(self):
        x1 = sp.Symbol('x1')
        x1b = sp.Symbol('x1')
        # These symbols are equal and should share any new attribute

        self.assertTrue(x1 is x1b)

        def tmp():
            return x1.abc_xyz
        self.assertRaises(AttributeError, tmp)

        x1.abc_xyz = 85
        self.assertEqual(tmp(), x1b.abc_xyz)


def main():
    unittest.main()

# see also the skip_slow logic at the beginning of the file
if __name__ == '__main__':
    main()
