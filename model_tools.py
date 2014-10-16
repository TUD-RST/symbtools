# -*- coding: utf-8 -*-
"""
functions to generate/analyse a model based on Lagrange formalism 2nd kind
with minimal coordinates:

Authors: Thomas Mutzke, Carsten Knoll, 2013
"""

import sympy as sp
import symb_tools as st


def Rz(phi):
    """
    Rotation Matrix in the xy plane
    """
    c=sp.cos(phi)
    s=sp.sin(phi)
    return sp.Matrix([[c, -s], [s, c]])


# helper function for velocity generation


def point_velocity(point, coord_symbs, velo_symbs, t):

    coord_funcs = []
    for c in coord_symbs:
        coord_funcs.append(sp.Function(c.name+"f")(t))

    coord_funcs = sp.Matrix(coord_funcs)
    p1 = point.subs(zip(coord_symbs, coord_funcs))

    v1_f = p1.diff(t)

    backsubs = zip(coord_funcs, coord_symbs) + zip(coord_funcs.diff(t),velo_symbs)

    return v1_f.subs(backsubs)


def symbColVector(n, s='a'):
    """
    returns a column vector with n symbols s, index 1..n
    >>> symbColVector(3)
    [a1]
    [a2]
    [a3]
    >>> symbColVector(2,'k')
    [k1]
    [k2]
    >>> symbRowVector(-1)
    Traceback (most recent call last):
        ...
    ValueError: Positive integer required
    >>> symbColVector(1,3)
    Traceback (most recent call last):
        ...
    TypeError: unsupported operand type(s) for +: 'int' and 'str'
    """
    if n <= 0 or int(n) != n:
        raise ValueError, "Positive integer required"

    A = sp.Matrix(n,1, lambda i,j:sp.Symbol( s+'%i'%(i+1) ) )
    return A





class SymbolicModel(object):
    """ model class """
    def __init__(self):#, eq_list, var_list, extforce_list, disforce_list):
        self.eq_list = None#eq_list
        self.qs = None#var_list
        self.extforce_list = None#extforce_list
        self.disforce_list = None#disforce_list
        self.f = None
        self.u = None
        self.g = None
        self.solved_eq = None
        self.zero_equilibrium = None
        self.state_eq = None
        self.A = None
        self.b = None
        self.x = None
        self.xd = None
        self.h = None

    def substitute_ext_forces(self, old_F, new_F, new_F_symbols):
        """
        Backround: sometimes modeling is easier with forces which are not
        the real control inputs. Then it is necessary to introduce the real
        forces later.
        """

        #nothing fancy has be done yet with the model
        assert self.eq_list != None
        assert (self.f, self.solved_eq, self.state_eq) == (None,)*3

        subslist = zip(old_F, new_F)
        self.eq_list = self.eq_list.subs(subslist)
        self.extforce_list = new_F_symbols

# TODO: convert the other functions into methods

def generate_model(T, U, q, F):
    """
    T kinetic co-energy
    U potential energy
    q independend deflection variables
    F external forces
    D dissipation function
    """

    #time variable
    t = sp.symbols('t')

    # ensure that the same Symbol for t is used
    for tmp in q:
        assert tmp.is_Function and tmp.args == (t,)

    #derivative of configuration variables
    qd = q.diff(t)
    qdd = qd.diff(t)

    #lagrange function
    L = T-U

    #substitute functions with symbols for partial derivatives
    n = len(q)
    qs = symbColVector(n, 'qs')
    qds = symbColVector(n, 'qds')
    qdds = symbColVector(n, 'qdds')
    #highest derivative first
    subslist = zip(qdd, qdds) + zip(qd, qds) + zip(q, qs)
    L = L.subs(subslist)


    # partial derivatives of L
    Ldq = st.jac(L, qs)
    Ldqd = st.jac(L, qds)

    # generalised external force
    f = sp.Matrix(F)

    # substitute symbols with functions for time derivative
    subslistrev = st.rev_tuple(subslist)
    Ldq = Ldq.subs(subslistrev)
    Ldqd = Ldqd.subs(subslistrev)
    Ldqd = Ldqd.diff(t)

    #lagrange equation 2nd kind
    model_eq = q*0
    for i in range(n):
        model_eq[i] = Ldqd[i]-Ldq[i] - f[i]

    # model equations with symbols
    #model_eq = model_eq.subs(subslist)

    # create object of model
    model1 = SymbolicModel()#model_eq, qs, f, D)
    model1.eq_list = model_eq
    model1.qs = qs
    model1.extforce_list = f
    model1.qds = qds
    model1.qdds = qdds

    # analyse the model

    return model1

def solve_eq(model):
    """
    solve model equations for accelerations
    xdd = f(x,xd)+g(x)*u
    """
    eq2_list = model.qdds*0
    for i in range(len(model.qdds)):
        eq2_list[i] = sp.solve([model.eq_list[i]], model.qdds[i])
        eq2_list[i] = sp.Eq(model.qdds[i], (eq2_list[i][model.qdds[i]]))
    model.solved_eq = eq2_list

def is_zero_equilibrium(model):
    """ checks model for equilibrium point zero """
    #substitution of state variables and their derivatives to zero
    #substitution of input to zero --> stand-alone system
    subslist = st.zip0(model.qs) + st.zip0(model.qds) + st.zip0(model.qdds) + st.zip0(model.extforce_list)
    eq_1 = model.eq_list.subs(subslist)
    model.zero_equilibrium = all(eq_1)

def state_eq(model):
    """
    from n equation order m to m*n equation 1st order
    xd = f(x)+g(x)*u
    """
    model.xd = symbColVector(2*len(model.qs), 'xd')
    model.x = symbColVector(2*len(model.qs), 'x')
    eq1 = 0*model.x
    #equations by definition (xd1 = x2)
    for i in range(len(model.qs)):
        eq1[i*2] = sp.Eq(model.xd[i*2], model.x[i*2+1])
    #model equations solved for acceleration
    for i in range(len(model.qs)):
        eq1[i*2+1] = model.solved_eq[i]
    #substitute "q" vars with "x" vars
    for i in range(len(model.qs)): #delete elements with odd index
        model.xd.row_del(i)

    qsubs = model.qs.col_insert(1, model.qds) #matrix with cols [qs,qds]
    subslist = zip(model.qdds, model.xd) + zip(qsubs, model.x)

    #state equations
    eq1 = eq1.subs(subslist)

    #rhs of state equations
    eq1_rhs = 0*eq1
    for i in range(len(eq1)):
        eq1_rhs[i] = eq1[i].rhs

    # overwrite xd because some rows xd were deleted above
    model.xd = symbColVector(2*len(model.qs), 'xd')


    #get input u from list
    model.u = st.sortedUniqueSymbList(model.extforce_list)
    #vectorfield f(x)
    subslist = st.zip0(model.u)
    model.f = eq1_rhs.subs(subslist)

    #vectorfield g(x)
    model.g = eq1_rhs.jacobian(model.u)

    #model state equation
    model.state_eq = eq1


def linearise(model):
    """ linearise model at equilibrium point zero """
    if model.zero_equilibrium:
        J = model.f.jacobian(model.x)
        # substitute state varibles with equilibrium point zero
        model.A = J.subs(st.zip0(model.x))
        model.b = model.g.subs(st.zip0(model.x))
