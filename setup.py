from distutils.core import setup
import os

# This directory
from symbtools import __version__

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read()

with open("extra_requirements_mpc.txt") as requirements_file:
    mpc_requirements = requirements_file.read()


# hack to ensure that IDE pycharm recognizes the additional requirements
# but also consistently specify them in extra file
mpc_requirements2 = "casadi"
assert mpc_requirements.strip() == mpc_requirements2

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
    install_requires=requirements,
    extras_require={
        'mpc':  mpc_requirements2}
)
