# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 11:25:00 2014

@author: Carsten Knoll
"""

import unittest
import sympy as sp
from sympy import sin, cos
import symb_tools as st
import model_tools as mt


from IPython import embed as IPS


class ModelToolsTest(unittest.TestCase):

    def setUp(self):
        pass


    def test_simple1(self):
        q1, = qq  = sp.Matrix(sp.symbols('q1,'))
        F1, = FF  = sp.Matrix(sp.symbols('F1,'))

        m = sp.Symbol('m')

        q1d = st.perform_time_derivative(q1, qq)
        q1dd = st.perform_time_derivative(q1, qq, order=2)

        T = q1d**2*m/2
        V = 0

        mod = mt.generate_symbolic_model(T, V, qq, FF)

        eq = m*q1dd - F1

        self.assertEqual(mod.eq_list[0], eq)

    # TODO: add some more test-Systems


def main():
    unittest.main()

if __name__ == '__main__':
    main()