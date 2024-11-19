"""
This module contains tests that trigger bugs that once occurred (and should be fixed now)
"""

import sympy as sp
import symbtools as st


def test_issue19():

    x,y,z = st.symb_vector('x,y,z')
    M = sp.Matrix([
        [x**2,y*x],
        [y*z,x**2]])
    st.generic_rank(M)