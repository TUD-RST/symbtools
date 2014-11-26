
# coding: utf-8

# In[2]:

import sympy as sp
import symb_tools as st
from IPython import embed as IPS

import ipHelp


# Hier versuche ich den Wikipedia-Algorithmus zur Bestimmung der Smith-Normalform zu implementieren.
# http://en.wikipedia.org/wiki/Smith_normal_form
#

# In[6]:

# Beispliel von Prof. Röbenack:
m0, m1, l, g, s= sp.symbols('m0, m1, l, g, s')
M = sp.Matrix([(m0+m1)*s**2,-l*m1*s**2,-1,0,-l*m1*s**2,l*m1*(l*s**2-g),0,-1]).reshape(2,4)
M


# In[11]:

def first_nonzero_element(seq):
    for i, elt in enumerate(seq):
        if not elt == 0:
            return i, elt
    raise ValueError("sequence must not vanish identically")


# In[23]:


def coeff_list_to_poly(coeffs, var):
    res = sp.sympify(0)
    for i, c in enumerate(coeffs):
        res += c*var**i
    return res

def solve_bezout_eq(p1, p2, var):
    """
    solving the bezout equation
    c1*p1 + c2*p2 = 1
    for monovariate polynomials p1, p2
    by ansatz-polynomials and equating
    coefficients
    """
    g1 = st.poly_degree(p1, var)
    g2 = st.poly_degree(p2, var)

    if (not sp.gcd(p1, p2) == 1) and (not p1*p2==0):
#        pass
        errmsg = "p1, p2 need to be coprime "\
                 "(condition for Bezout identity to be solveble).\n"\
                 "p1 = {p1}\n"\
                 "p2 = {p2}"
        raise ValueError(errmsg.format(p1=p1, p2=p2))

    if p1 == p2 == 0:
        raise ValueError("invalid: p1==p2==0")
    if p1 == 0 and g2 > 0:
        raise ValueError("invalid: p1==0 and not p2==const ")
    if p2 == 0 and g1 > 0:
        raise ValueError("invalid: p2==0 and not p1==const ")


    # Note: degree(0) = -sp.oo = -inf

    k1 = g2 - 1
    k2 = g1 - 1

    if k1<0 and k2 <0:
        if p1 == 0:
            k2 = 0
        else:
            k1 = 0

    if k1 == -sp.oo:
        k1 = -1
    if k2 == -sp.oo:
        k2 = -1

    cc1 = sp.symbols("c1_0:%i" % (k1 + 1))
    cc2 = sp.symbols("c2_0:%i" % (k2 + 1))

    c1 = coeff_list_to_poly(cc1, var)
    c2 = coeff_list_to_poly(cc2, var)

    # Bezout equation:
    eq = c1*p1 + c2*p2 - 1

    # solve it w.r.t. the unknown coeffs
    sol = sp.solve(eq, cc1+cc2, dict=True)

    if len(sol) == 0:
        pass
        IPS()
        errmsg = "No solution found.\n"\
                 "p1 = {p1}\n"\
                 "p2 = {p2}"

        raise ValueError(errmsg.format(p1=p1, p2=p2))
    sol = sol[0]

    return c1.subs(sol), c2.subs(sol)



def smith_column_step(col, t, var):

    nr = len(col)
    L0 = sp.eye(nr)
    new_col = col*1
    at = col[t]

    for k, ak in enumerate(col):
        if k == t or ak == 0:
            continue
        GCD = sp.gcd(at, ak)
        alpha_t = sp.simplify(at/GCD)
        gamma_k = sp.simplify(ak/GCD)

        sigma, tau = solve_bezout_eq(alpha_t, gamma_k, var)

        L0[t, t] = sigma
        L0[t, k] = tau
        L0[k, t] = -gamma_k
        L0[k, k] = alpha_t

        new_col = L0*col
        # Linksmultiplikation der Spalte mit L0 liefert eine neue Spalte
        # mit Einträgen beta bei t und 0 bei k
        break

    return new_col, L0



def smith_step(A, t, var):
    # erste Spalte (Index: j), die nicht komplett 0 ist
    # j soll größer als Schrittzähler sein
    nr, nc = A.shape

    row_op_list = []

    cols = st.col_split(A)
    for j, c in enumerate(cols):
        if j < t:
            continue
        if not c == c*0:
            break
    # Eintrag mit Index t soll ungleich 0 sein, ggf. Zeilen tauschen
    if c[t] == 0:
        i, elt = first_nonzero_element(c)
        row_op = sp.eye(nr)
        tmp1 = row_op[t, :]
        tmp2 = row_op[i, :]
        row_op[i, :] = tmp1
        row_op[t, :] = tmp2
        A = row_op*A
        c = row_op*c

        row_op_list.append(row_op)

    col = c
    while True:
        new_col, L0 = smith_column_step(col, t, var)
        if L0 == sp.eye(nr):
            # nothing has changed
            break
        row_op.append(L0)
        col = new_col


    IPS()






if __name__ == "__main__":

    #solve_bezout_eq(0, 3, s)
    smith_step(M, 0, s)




