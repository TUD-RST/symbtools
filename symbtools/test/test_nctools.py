# -*- coding: utf-8 -*-
"""
Created on Fri 2015-03-20

@author: Carsten Knoll
"""

import unittest, sys
import inspect, os

import sympy as sp


import symbtools as st
import symbtools.noncommutativetools as nct
import pickle

from ipydex import IPS


if 'all' in sys.argv:
    FLAG_all = True
else:
    FLAG_all = False

# s, t from Noncommutative tools
t = nct.t
s = nct.s


# own decorator for skipping slow tests
def skip_slow(func):
    return unittest.skipUnless(FLAG_all, 'skipping slow test')(func)


def make_abspath(*args):
    """
    returns new absolute path, basing on the path of this module
    """
    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return os.path.join(current_dir, *args)


class NCTTest(unittest.TestCase):

    def setUp(self):
        st.init_attribute_store(reinit=True)

    def test_apply_deriv1(self):
        a, b = sp.symbols("a, b")
        f1 = sp.Function('f1')(t)
        F1 = a*f1

        res1 = nct.apply_deriv(F1, 1, s, t)
        self.assertEqual(res1, F1.diff(t) + F1*s)

        res2 = nct.apply_deriv(F1, 3, s, t)
        self.assertEqual(res2, F1.diff(t, 3) + 3*F1.diff(t, 2)*s + 3*F1.diff(t)*s**2 + F1*s**3)

    def test_apply_deriv2(self):
        y1, y2 = yy = sp.Matrix( sp.symbols('y1, y2', commutative=False) )

        ydot1 = st.time_deriv(y1, yy)
        ydot2 = st.time_deriv(y2, yy)
        yddot1 = st.time_deriv(y1, yy, order=2)
        ydddot1 = st.time_deriv(y1, yy, order=3)

        res1 = nct.apply_deriv(y1, 1, s, t, func_symbols=yy)
        self.assertEqual(res1, ydot1 + y1*s)

        res3 = nct.apply_deriv(y1*s, 1, s, t, func_symbols=yy)
        self.assertEqual(res3, ydot1*s + y1*s**2)

        res4 = nct.apply_deriv(y2 + y1, 1, s, t, func_symbols=yy)
        self.assertEqual(res4, ydot1 + ydot2 + y1*s + y2*s)

        res5 = nct.apply_deriv(ydot1 + y1*s, 1, s, t, func_symbols=yy)
        self.assertEqual(res5, yddot1 + 2*ydot1*s + y1*s**2)

        res6 = nct.apply_deriv(y1, 2, s, t, func_symbols=yy)
        self.assertEqual(res5, res6)

        res2 = nct.apply_deriv(y1, 3, s, t, func_symbols=yy)
        self.assertEqual(res2, ydddot1 + 3*yddot1*s + 3*ydot1*s**2 + y1*s**3)

    def test_apply_deriv3(self):
        a, b = sp.symbols("a, b", commutative=False)

        res1 = nct.apply_deriv(a, 1, s, t, func_symbols=[a, b])
        adot = res1.subs(s,0)
        self.assertFalse(adot.is_commutative)

    def test_right_shift(self):
        a, b = sp.symbols("a, b")
        f1 = sp.Function('f1')(t)
        f1d = f1.diff(t)
        f2 = sp.Function('f2')(t)

        res1 = nct.right_shift(s*f1, s, t)
        ex1 = f1.diff(t) + f1*s

        self.assertEqual(res1, ex1)

        res2 = nct.right_shift(f2*s*f1, s, t)
        ex2= f2*f1.diff(t) + f2*f1*s

        self.assertEqual(res2, ex2)

        res3 = nct.right_shift(a*f2*s*f1d, s, t)
        ex3= a*f2*f1.diff(t, 2) + a*f2*f1d*s

        self.assertEqual(res3, ex3)

        res4 = nct.right_shift(s*f1*f2, s, t)
        ex4 = f1.diff(t)*f2 + f1*f2*s + f1*f2.diff(t)

        self.assertEqual(res4, ex4)

        self.assertRaises( ValueError, nct.right_shift, s*f1*(f2+1), s, t)

        res = nct.right_shift(s, s, t)
        self.assertEqual(res, s)

        res = nct.right_shift(s**2, s, t)
        self.assertEqual(res, s**2)

        res = nct.right_shift(a, s, t)
        self.assertEqual(res, a)

        self.assertRaises( ValueError, nct.right_shift, sp.sin(s), s, t)
        self.assertRaises( ValueError, nct.right_shift, s*sp.sin(s), s, t)

    def test_right_shift2(self):
        a, b = sp.symbols("a, b", commutative=False)
        f1 = sp.Function('f1')(t)
        f1d = f1.diff(t)
        f2 = sp.Function('f2')(t)

        res1 = nct.right_shift(s*t, s, t)
        self.assertEqual(res1, 1 + t*s)

        res2 = nct.right_shift(s, s, t)
        self.assertEqual(res2, s)

        res3 = nct.right_shift(s**4, s, t)
        self.assertEqual(res3, s**4)

        res4 = nct.right_shift(s**4*a*b, s, t)
        self.assertEqual(res4, a*b*s**4)

        res5 = nct.right_shift(s**2*a*s*b*s, s, t)
        ex5 = a*b*s**4
        self.assertEqual(res5, ex5)

        res6 = nct.right_shift(s**2*(a*t**3), s, t)
        ex6 = a*(6*t + 6*t**2*s + t**3*s**2)
        self.assertEqual(res6, ex6)

        res7 = nct.right_shift(f1*s*a*s*b, s, t)
        self.assertEqual(res7, f1*a*b*s**2)

    def test_right_shift3(self):
        a, b = sp.symbols("a, b", commutative = False)
        f1 = sp.Function('f1')(t)
        f2 = sp.Function('y2')(t)
        f1d = f1.diff(t)
        f1dd = f1.diff(t, 2)
        f2d = f2.diff(t)
        f2dd = f2.diff(t, 2)

        res1 = nct.right_shift(s*f1d*f2d, s, t)
        ex1 = f1dd*f2d + f1d*f2dd + f1d*f2d*s

        self.assertEqual(res1, ex1)

        test = s*f2*f2d
        res2 = nct.right_shift(test, s, t)
        ex2 = f2d**2 + f2*f2dd + f2*f2d*s

        self.assertEqual(res2, ex2)

    def test_right_shift4(self):

        y1, y2 = yy = sp.Matrix( sp.symbols('y1, y2', commutative=False) )

        ydot1, ydot2 = st.time_deriv(yy, yy)
        res1 = nct.right_shift(s*y1, s, t, yy)

        self.assertEqual(res1, ydot1 + y1*s)

    def test_right_shift5(self):
        a, b = sp.symbols("a, b", commutative = False)
        f1 = sp.Function('f1')(t)
        f2 = sp.Function('y2')(t)

        res1 = nct.right_shift(f1**-1, s, t)
        self.assertEqual(res1, 1/f1)

        res2 = nct.right_shift((f1 + f2)**-1, s, t)
        self.assertEqual(res2, 1/(f1 + f2))

        ff = (f1 + f2)**-1
        res3 = nct.right_shift(s*ff, s, t) - (ff.diff(t) + ff*s)
        res3 = res3.expand()
        self.assertEqual(res3, 0)

    def test_right_shift_all(self):
        a, b = sp.symbols("a, b", commutative=False)
        f1 = sp.Function('f1', commutative=False)(t)
        f2 = sp.Function('f2', commutative=False)(t)
        f1d = f1.diff(t)
        f2d = f2.diff(t)

        p1 = s*(f1 + f2)

        ab = sp.Matrix([a, b])
        adot, bdot = st.time_deriv(ab, ab)

        res1 = nct.right_shift_all(p1)
        self.assertEqual(res1, f1d + f1*s + f2d + f2*s)

        res2 = nct.right_shift_all(f1**-1, s, t)
        self.assertEqual(res2, 1/f1)

        res3 = nct.right_shift_all(s*a + s*a*b, s, t, [])
        self.assertEqual(res3, a*s + a*b*s)

        res4 = nct.right_shift_all(s*a + s*a*b, s, t, [a, b])
        self.assertEqual(res4, a*s + a*b*s + adot + a*bdot + adot*b)

    def test_right_shift_all2(self):
        a, b = sp.symbols("a, b", commutative=False)

        ab = sp.Matrix([a, b])
        adot, bdot = ab_dot = st.time_deriv(ab, ab)

        sab = sp.Matrix([s*a, s*b])

        res1 = nct.right_shift_all(sab, func_symbols=ab)
        res2 = ab_dot + nct.nc_mul(ab, s)

        self.assertEqual(res1, res2)

        res = nct.right_shift_all(s, s, t)
        self.assertEqual(res, s)

        res = nct.right_shift_all(s**2, s, t)
        self.assertEqual(res, s**2)

        res = nct.right_shift_all(a, s, t)
        self.assertEqual(res, a)

        res = nct.right_shift_all(a**(sp.S(1)/2), s, t)
        self.assertEqual(res, a**(sp.S(1)/2))

        res = nct.right_shift_all(s*1/a, s, func_symbols=ab)
        self.assertEqual(res, 1/a*s -1/a**2*adot)

        res = nct.right_shift_all(s + sp.sin(a), s, func_symbols=ab)
        self.assertEqual(res, s + sp.sin(a))

        self.assertRaises( ValueError, nct.right_shift_all, s**(sp.S(1)/2), s, t)

        self.assertRaises( ValueError, nct.right_shift_all, sp.sin(s), s, t)

    def test_right_shift_all_naive(self):
        a, b = sp.symbols("a, b", commutative=False)

        ab = sp.Matrix([a, b])
        adot, bdot = ab_dot = st.time_deriv(ab, ab)
        addot, bddot = ab_ddot = st.time_deriv(ab, ab, order=2)

        sab = sp.Matrix([s*a, s*b])
        abs = sp.Matrix([a*s, b*s])

        res1 = nct.right_shift_all(sab, func_symbols=None)
        self.assertEqual(res1, abs)

        # normally derivatives are recognized as time dependent automatically
        res2 = nct.right_shift_all(s*adot)
        self.assertEqual(res2, addot + adot*s)

        # if func_symbols=None derivatives are like constants
        res3 = nct.right_shift_all(s*adot, func_symbols=None)
        self.assertEqual(res3, adot*s)


    @unittest.expectedFailure
    def test_nc_sympy_multiplication_bug(self):
    # This seems to be a sympy bug
        a, b = sp.symbols("a, b", commutative=False)
        E = sp.eye(2)

        Mb = b*E
        Mab = a*b*E

        res = a*Mb - Mab

        self.assertEqual(res, 0*E)

    def test_nc_multiplication(self):
        a, b = sp.symbols("a, b", commutative=False)
        E = sp.eye(2)

        Mb = b*E
        Mab = a*b*E

        res = nct.nc_mul(a, Mb) - Mab
        self.assertEqual(res, 0*E)

        res2 = nct.nc_mul(a*E, b*E)
        self.assertEqual(res2, Mab)

        res3 = nct.nc_mul(Mb, Mab)
        self.assertEqual(res3, b*a*b*E)

        # this was a bug 2019-02-08 10:18:36
        Mb2 = sp.ImmutableDenseMatrix(Mb)
        self.assertEqual(nct.nc_mul(a, Mb2), Mb*a)
        self.assertEqual(nct.nc_mul(Mb2, a), a*Mb)
        self.assertFalse(Mb*a == a*Mb)

    def test_make_all_symbols_commutative(self):

        a, b, c = sp.symbols("a, b, c", commutative=False)
        x, y = sp.symbols("x, y")

        exp1 = a*b*x + b*c*y
        exp2 = b*a*x + c*y*b

        diff = exp1 - exp2

        self.assertFalse(diff == 0)
        diff_c, subs_tuples = nct.make_all_symbols_commutative(diff)
        exp1_c, subs_tuples = nct.make_all_symbols_commutative(exp1)

        self.assertTrue( all([r.is_commutative for r in exp1_c.atoms()]) )

    def test_make_all_symbols_commutative2(self):
        import pickle
        path = make_abspath('test_data', 'Q_matrix_cart_pendulum.pcl')
        with open(path, 'rb') as pfile:
            Q = pickle.load(pfile)

        Qc, stl = nct.make_all_symbols_commutative(Q, '')

    def test_make_all_symbols_commutative3(self):
        x1, x2, x3 = xx = st.symb_vector('x1, x2, x3', commutative=False)

        xxd = st.time_deriv(xx, xx)

        xxd_c = nct.make_all_symbols_commutative(xxd)[0]

        self.assertEqual(xxd_c[0].difforder, 1)

    def test_make_all_symbols_noncommutative(self):

        a, b, c = abc = sp.symbols("a, b, c", commutative=True)
        x, y = xy = sp.symbols("x, y", commutative=False)

        adddot = st.time_deriv(a, abc, order=3)

        exp1 = a*b*x + b*c*y

        exp1_nc, subs_tuples = nct.make_all_symbols_noncommutative(exp1)

        self.assertTrue( all([ not r.is_commutative for r in exp1_nc.atoms()]) )

        adddot_nc = nct.make_all_symbols_noncommutative(adddot)[0]

        self.assertEqual(adddot.difforder, adddot_nc.difforder)

    def test_nc_coeffs(self):

        a, b, c, s = sp.symbols("a, b, c, s", commutative=False)

        p0 = a
        p1 = a + b*s + c*s
        p2 = a + (b**2 - c)*s - a*b*a*s**2 - s
        p8 = a + (b**2 - c)*s - a*b*a*s**2 - c*s**8

        c0 = nct.nc_coeffs(p0, s)
        c1 = nct.nc_coeffs(p1, s)
        c2 = nct.nc_coeffs(p2, s)
        c8 = nct.nc_coeffs(p8, s)

        self.assertEqual(c0, [a] + [0]*10)
        self.assertEqual(c1, [a, b + c] + [0]*9)
        self.assertEqual(c2, [a, b**2 - c - 1, -a*b*a] + [0]*8)
        self.assertEqual(c8, [a, b**2 - c, -a*b*a, ] + [0]*5 + [-c] + [0]*2)

        d01 = nct.nc_coeffs(p0, s, 3)
        d02 = nct.nc_coeffs(p0, s, 1)
        d03 = nct.nc_coeffs(p0, s, 0)

        self.assertEqual(d01, [p0] + [0]*3)
        self.assertEqual(d02, [p0] + [0])
        self.assertEqual(d03, [p0])

        d11 = nct.nc_coeffs(0, s, 5)
        d12 = nct.nc_coeffs(0, s, 0)
        self.assertEqual(d11, [0]*6)
        self.assertEqual(d12, [0])

    def test_nc_degree(self):
        a, b, c, s = sp.symbols("a, b, c, s", commutative=False)

        p1 = a + 5 + b*a*s - s**3
        p2 = c
        p3 = a + b*s + c*s**20

        M1 = sp.Matrix([p1, p2, p1, p3])

        d1 = nct.nc_degree(p1, s)
        d2 = nct.nc_degree(p2, s)
        d3 = nct.nc_degree(p3, s)

        d4 = nct.nc_degree(M1, s)

        self.assertEqual(d1, 3)
        self.assertEqual(d2, 0)
        self.assertEqual(d3, 20)
        self.assertEqual(d4, 20)

    def test_unimod_inv(self):
        y1, y2 = yy = st.symb_vector('y1, y2', commutative=False)
        s = sp.Symbol('s', commutative=False)
        ydot1, ydot2 = yyd1 = st.time_deriv(yy, yy, order=1, commutative=False)
        yddot1, yddot2 = yyd2 = st.time_deriv(yy, yy, order=2, commutative=False)
        yyd3 = st.time_deriv(yy, yy, order=3, commutative=False)
        yyd4 = st.time_deriv(yy, yy, order=4, commutative=False)
        yya = st.row_stack(yy, yyd1, yyd2, yyd3, yyd4)

        M1 = sp.Matrix([yy[0]])
        M1inv = nct.unimod_inv(M1, s, time_dep_symbs=yy)
        self.assertEqual(M1inv, M1.inv())

        M2 = sp.Matrix([[y1, y1*s], [0, y2]])
        M2inv = nct.unimod_inv(M2, s, time_dep_symbs=yy)

        product2a = nct.right_shift_all( nct.nc_mul(M2, M2inv), s, func_symbols=yya)
        product2b = nct.right_shift_all( nct.nc_mul(M2inv, M2), s, func_symbols=yya)

        res2a = nct.make_all_symbols_commutative( product2a)[0]
        res2b = nct.make_all_symbols_commutative( product2b)[0]
        self.assertEqual(res2a, sp.eye(2))
        self.assertEqual(res2b, sp.eye(2))

    def test_unimod_inv2(self):
        y1, y2 = yy = st.symb_vector('y1, y2', commutative=False)
        s = sp.Symbol('s', commutative=False)
        ydot1, ydot2 = yyd1 = st.time_deriv(yy, yy, order=1, commutative=False)
        yddot1, yddot2 = yyd2 = st.time_deriv(yy, yy, order=2, commutative=False)
        yyd3 = st.time_deriv(yy, yy, order=3, commutative=False)
        yyd4 = st.time_deriv(yy, yy, order=4, commutative=False)
        yya = st.row_stack(yy, yyd1, yyd2, yyd3, yyd4)

        # this Matrix is not unimodular due to factor 13 (should be 1)
        M3 = sp.Matrix([[ydot2,                                              13*y1*s],
                       [y2*yddot2 + y2*ydot2*s, y1*yddot2 + y2*y1*s**2 + y2*ydot1*s + ydot2*ydot1]])

        with self.assertRaises(ValueError) as cm:
            res = nct.unimod_inv(M3, s, time_dep_symbs=yya)


    @skip_slow
    def test_unimod_inv3(self):
        y1, y2 = yy = st.symb_vector('y1, y2', commutative=False)
        s = sp.Symbol('s', commutative=False)
        ydot1, ydot2 = yyd1 = st.time_deriv(yy, yy, order=1, commutative=False)
        yddot1, yddot2 = yyd2 = st.time_deriv(yy, yy, order=2, commutative=False)
        yyd3 = st.time_deriv(yy, yy, order=3, commutative=False)
        yyd4 = st.time_deriv(yy, yy, order=4, commutative=False)
        yya = st.row_stack(yy, yyd1, yyd2, yyd3, yyd4)

        M3 = sp.Matrix([[ydot2,                                              y1*s],
                       [y2*yddot2 + y2*ydot2*s, y1*yddot2 + y2*y1*s**2 + y2*ydot1*s + ydot2*ydot1]])

        M3inv = nct.unimod_inv(M3, s, time_dep_symbs=yya)

        product3a = nct.right_shift_all( nct.nc_mul(M3, M3inv), s, func_symbols=yya)
        product3b = nct.right_shift_all( nct.nc_mul(M3inv, M3), s, func_symbols=yya)
        res3a = nct.make_all_symbols_commutative(product3a)[0]
        res3b = nct.make_all_symbols_commutative(product3b)[0]
        res3a.simplify()
        res3b.simplify()

        self.assertEqual(res3a, sp.eye(2))
        self.assertEqual(res3b, sp.eye(2))

    @skip_slow
    def test_unimod_inv4(self):
        path = make_abspath('test_data', 'unimod_matrix_unicycle.pcl')
        with open(path, 'rb') as pfile:
            pdict = pickle.load(pfile)

        PQ = pdict['PQ']
        s = [ symb for symb in PQ.s if str(symb) == "s"][0]
        self.assertTrue(s in PQ.s)

        abc = pdict['abc']
        #kk = pdict['kk']
        #JEh = pdict['JEh']

        inv = nct.unimod_inv(PQ, s, None, abc, max_deg=2)
        res = nct.nc_mul(inv, PQ)
        res2 = nct.right_shift_all(res, s, None, abc)
        res3, tmp = nct.make_all_symbols_commutative(res2)
        res4 = st.subs_random_numbers(res3, prime=True)
        self.assertEqual(res4, sp.eye(3))


class NCTTest2(unittest.TestCase):

    def setUp(self):
        st.init_attribute_store(reinit=True)

    def test_commutative_simplification(self):

        x1, x2 = xx = st.symb_vector('x1, x2', commutative=False)
        y1, y2 = yy = st.symb_vector('y1, y2', commutative=False)
        s, z, t = sz = st.symb_vector('s, z, t', commutative=False)

        a, b = ab = st.symb_vector('a, b', commutative=True)

        F = sp.Function('F')(t)

        e1 = x1*y1 - y1*x1
        e2 = e1*s + x2
        e3 = e1*s + x2*s

        M1 = sp.Matrix([[e1, 1], [e2, e3]])

        r1 = nct.commutative_simplification(e1, s)
        self.assertEqual(r1, 0)

        r2 = nct.commutative_simplification(e2, s)
        self.assertEqual(r2, x2)

        r3 = nct.commutative_simplification(e3, s)
        self.assertEqual(r3, x2*s)

        r4 = nct.commutative_simplification(M1, s)
        r4_expected = sp.Matrix([[0, 1], [x2, x2*s]])
        self.assertEqual(r4, r4_expected)

        f1 = x1*s*x2*s
        f2 = s**2*x1*x2
        f3 = a*x1*s**2
        f4 = F*s

        with self.assertRaises(ValueError) as cm:
            nct.commutative_simplification(f1, s)

        with self.assertRaises(ValueError) as cm:
            nct.commutative_simplification(f2, s)

        with self.assertRaises(ValueError) as cm:
            nct.commutative_simplification(e1, [s, z])

        with self.assertRaises(ValueError) as cm:
            nct.commutative_simplification(f3, s)

        with self.assertRaises(NotImplementedError) as cm:
            nct.commutative_simplification(f4, s)


def main():
    # remove command line args which should not be passed to the testframework
    if 'all' in sys.argv:
        sys.argv.remove('all')

    unittest.main()


if __name__ == '__main__':
    main()