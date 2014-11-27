
# coding: utf-8

# In[2]:

import sympy as sp
import symb_tools as st
from IPython import embed as IPS

import ipHelp


# Hier versuche ich den Wikipedia-Algorithmus zur Bestimmung der Smith-Normalform zu implementieren.
# http://en.wikipedia.org/wiki/Smith_normal_form
#

"""
Aktueller Stand: Schritte 1-3 sind umgesetzt. Schritt 4 (Normierung) fehlt noch

Weiteres Vorgehen:
    * Normierung,
    * Umgang mit Rechteckigen Matrizen,
    * Erweiterung auf andere algebraische Strukturen als Polynome
"""


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
        errmsg = "No solution found.\n"\
                 "p1 = {p1}\n"\
                 "p2 = {p2}"

        raise ValueError(errmsg.format(p1=p1, p2=p2))

    sol = sol[0]
    sol_symbols = st.atoms(sp.Matrix(sol.values()), sp.Symbol)

    # there might be some superflous coeffs
    free_c_symbols = set(cc1+cc2).intersection(sol_symbols)
    if free_c_symbols:
        # set them to zero
        fcs0 = st.zip0(free_c_symbols)
        keys, values = zip(*sol.items())
        new_values = [v.subs(fcs0) for v in values]

        sol = dict(zip(keys, new_values)+fcs0)


    return c1.subs(sol), c2.subs(sol)



def smith_column_step(col, t, var):

    nr = len(col)
    L0 = sp.eye(nr)
    col = col.expand()
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

        new_col = sp.expand(L0*col)
        # Linksmultiplikation der Spalte mit L0 liefert eine neue Spalte
        # mit Einträgen beta bei t und 0 bei k
        break

    return new_col, L0

def row_swap(n, i1, i2):
    row_op = sp.eye(n)
    tmp1 = row_op[i1, :]
    tmp2 = row_op[i2, :]
    row_op[i2, :] = tmp1
    row_op[i1, :] = tmp2

    return row_op

def row_op(n, i1, i2, c1, c2):
    """
    new <row i1> is <old i1>*c1 + <old i2>*c2
    """
    assert not c1 == 0
    row_op = sp.eye(n)
    row_op[i1, i1] = c1
    row_op[i1, i2] = c2

    return row_op


def smith_step(A, t, var):
    # erste Spalte (Index: j), die nicht komplett 0 ist
    # j soll größer als Schrittzähler sein
    nr, nc = A.shape

    row_op_list = []

    cols = st.col_split(A)

    # erste nicht verschwindende Spalte finden
    for j, c in enumerate(cols):
        if j < t:
            continue
        if not c == c*0:
            break
    # Eintrag mit Index t soll ungleich 0 sein, ggf. Zeilen tauschen
    if c[t] == 0:
        i, elt = first_nonzero_element(c)
        ro = row_swap(nr, t, i)
        c = ro*c

        row_op_list.append(ro)

    col = c.expand()
    while True:
        new_col, L0 = smith_column_step(col, t, var)
        if L0 == sp.eye(nr):
            # nothing has changed
            break
        row_op_list.append(L0)
        col = new_col

    # jetzt teilt col[t] alle Einträge in col
    # Probe in der nächsten Schleife

    col.simplify()
    col = col.expand()
    for i,a in enumerate(col):
        if i == t:
            continue
        if not sp.simplify(sp.gcd(a, col[t]) - col[t]) == 0:
            IPS()
            raise ValueError, "col[t] should divide all entries in col"
        quotient = sp.simplify(a/col[t])
        if a == 0:
            continue

        # eliminiere a
        ro = row_op(nr, i, t, -1, quotient)
        row_op_list.append(ro)


    return row_op_list

def smith_form(A, var):

    nr, nc = A.shape
    row_op_list = []
    total_ro = sp.eye(nr)

    col_op_list = []
    total_co = sp.eye(nc)

    A_tmp = A*1
    for t in xrange(nr):
        while True:
            print "t=", t
            step_ro = sp.eye(nr)
            step_co = sp.eye(nc)

            new_ro_list = smith_step(A_tmp, t, var)
            for ro in new_ro_list:
                A_tmp = ro*A_tmp
                A_tmp.simplify()
                total_ro = ro*total_ro
                step_ro = ro*step_ro

            row_op_list.extend(new_ro_list)

            # after processing column t, now process row t

            new_co_list = smith_step(A_tmp.T, t, var)
            # Transpose the column operations
            new_co_list = [co.T for co in new_co_list]
            for co in new_co_list:
                A_tmp = A_tmp*co
                A_tmp.simplify()
                total_co = total_co*co
                step_co = step_co*co

            if step_ro == sp.eye(nr) and step_co == sp.eye(nc):
                break
        print "fertig mit t=", t



    total_ro.simplify()
    total_co.simplify()
    return total_ro.expand(), total_co.expand()



if __name__ == "__main__":

    #solve_bezout_eq(0, 3, s)
    #smith_step(M, 0, s)

    if 0:
        M1 = sp.Matrix([[s-2, 0, 0, 0],
                        [1, s-1, 0, 0],
                        [0, 1, s  , 1],
                        [-1, -1, -1, s-2]])

        ro = smith_form(M1, s)

        M1_1 = sp.simplify(ro*M1)
        IPS()

        # - - - -

    M1_1 = sp.Matrix([
    [s - 2,     0,             0, 0],
    [    1, s - 1,             0, 0],
    [    0,     0, s*(s - 2) + 1, 0],
    [    0,     1,             s, 1]])

    ro2, co = smith_form(M1_1.T, s)

    M1_2 = sp.simplify(ro2*M1_1).T
    ro3 = smith_form(M1_2, s)


#    M = sp.Matrix([
#    [s - 2, -(s - 2)*(s - 1) + 1, s*((s - 2)*(s - 1) - 1), s**4 - 5*s**3 + 8*s**2 - 5*s + 1],
#    [    0,                s - 1,              s*(-s + 1),    -(s - 1)*(s**2 - 2*s + 1) + 1],
#    [    0,                    0,           s*(s - 2) + 1,      s + (s - 2)*(s*(s - 2) + 1)],
#    [    0,                    0,                       0,                                1]])
#
#    M.simplify()
#    ro, co = smith_form(M, s)
    IPS()






