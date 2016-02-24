from distutils.core import setup

setup(
    name='symbtools',
    version='0.1.8',
    author='Carsten Knoll, Klemens Fritzsche',
    author_email='Carsten.Knoll@tu-dresden.de',
    packages=['symbtools'],
    url='https://github.com/cknoll/rst_symbtools',
    license='BSD3',
    description='Symbolic calculations related to dynamical systems.',
    long_description="""
    A collection of functions to facilitate the (symbolic) calculations
    associated with the investigation of nonlinear dynamical systems in
    the field of control theory.
    """,
    requires=[
        "sympy (>= 0.7.6)",
        "numpy (>= 1.10.4)",
        "scipy (>= 0.17.0)",
        "ipython (>= 3.1.0)",
    ],
)
