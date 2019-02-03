"""
This modules serves to handle command line args which are passed to the test_* scripts.

It should be loaded before the `untitest` module as it alters sys.argv which seems to be evaluated by that module
"""

import sys

if 'all' in sys.argv:
    FLAG_all = True
else:
    FLAG_all = False

if 'optdep' in sys.argv:
    FLAG_optdep = True
else:
    FLAG_optdep = False

# remove command line args which should not be passed to the test framework
# the flags have already been set
custom_args = ["all", "optdep"]
for carg in custom_args:
    if carg in sys.argv:
        sys.argv.remove(carg)

import unittest

import inspect
from ipydex import IPS


def get_all_tests_from_this_module():
    """
    inspects the callers namespace
    :param internal_modname:
    :return:
    """
    relevant_frame = inspect.currentframe().f_back

    test_suite = unittest.TestSuite()
    for k, v in relevant_frame.f_locals.items():
        if isinstance(v, unittest.TestCase):
            test_suite.addTest(v)

    return test_suite







