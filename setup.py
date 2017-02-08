from distutils.core import setup
import os

# This directory
dir_setup = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_setup, 'symbtools', 'release.py')) as f:
    # Defines __version__
    exec(f.read())

setup(
    name='symbtools',
    version=__version__,
    author='Carsten Knoll, Klemens Fritzsche',
    author_email='Carsten.Knoll@tu-dresden.de',
    packages=['symbtools'],
    url='https://github.com/cknoll/rst_symbtools',
    license='BSD3',
    description='Symbolic calculations related to dynamical systems.',
    long_description="""
    A collection of functions to facilitate the (symbolic) calculations
    associated with the investigation of nonlinear dynamical systems in
    the field of control theory (0.1.10+ has python3 support).
    """,
    requires=[
        "sympy (>= 0.7.6)",
        "numpy (>= 1.10.4)",
        "scipy (>= 0.17.0)",
        "ipython (>= 3.1.0)",
    ],
)
