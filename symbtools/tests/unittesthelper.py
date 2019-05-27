"""
This modules serves to handle command line args which are passed to the test_* scripts.

It should be loaded before the `untitest` module as it alters sys.argv which seems to be evaluated by that module
"""

import sys
import os


class Container(object):
    pass


def set_flags():
    """
    set some flags based on command line args or environment variables

    :param FLAGS:   Container where to store the flags
    :return:
    """

    flags = Container()

    flags.all = bool(os.getenv('test_all', False))
    flags.optdep = bool(os.getenv('test_optdep', False))

    # allow command line args to override envvars
    if 'all' in sys.argv:
        flags.all = True

    if 'optdep' in sys.argv:
        flags.optdep = True

    # now remove command line args which should not be passed to the test framework
    custom_args = ["all", "optdep"]
    for carg in custom_args:
        if carg in sys.argv:
            sys.argv.remove(carg)

    return flags

FLAGS = set_flags()

import unittest


# own decorator for skipping slow tests
def skip_slow(func):
    return unittest.skipUnless(FLAGS.all, 'skipping slow test')(func)


tests_with_optional_deps = []


def optional_dependency(func):
    msg = 'skipping optional dependency test: {}'.format(func.__name__)
    wrapped_func = unittest.skipUnless(FLAGS.optdep, msg)(func)

    if sys.version_info[0] >= 3:
        name = func.__qualname__
    else:
        name = func.__name__

    tests_with_optional_deps.append(name)
    return wrapped_func


# TODO: remove this function (planned use-case (separately run tests for different tags) was not implemented)
def gen_suite_from_ns_and_list(ns, list_of_testnames):
    """
    assume list_list_of_testnames = ["MyTestClass2.test_something1", "MyTestClassABC.test_XYZ"]
    -> generate a suite for them.

    :param ns:                  namespace
    :param list_of_testnames:   list of testnames
    :param internal_modname:
    :return:
    """

    test_suite = unittest.TestSuite()
    for testname in list_of_testnames:
        classname, methname = testname.split(".")
        theclass = ns[classname]
        assert issubclass(theclass, unittest.TestCase)

        test_suite.addTest(theclass(methname))

    return test_suite


def inject_tests_into_namespace(target_ns, list_of_modules):
    """
    :param target_ns:
    :param list_of_modules:
    :return:
    """

    name_mod_dict = {}

    for mod in list_of_modules:
        for k, v in vars(mod).items():
            if isinstance(v, type):
                if issubclass(v, unittest.TestCase):
                    # name = "{}{}".format(mod.__name__, v.__name__)
                    name = v.__name__
                    # keep track of in which module a name was defined first
                    if name in target_ns:
                        first_mod_name = name_mod_dict[name]
                        msg = "Name-Conflict: '{}' is defined in both {} and {} ".format(name, mod.__name__,
                                                                                         first_mod_name)
                        raise NameError(msg)
                    assert name not in target_ns
                    name_mod_dict[name] = mod.__name__
                    target_ns[name] = v


def smart_run_tests_in_ns(ns):
    """
    run tests in namspace (typically ns = globals() of caller module)
    :param ns:
    :return:
    """

    if FLAGS.optdep:
        # run only those test with optional dependencies
        runner = unittest.TextTestRunner()
        runner.run(gen_suite_from_ns_and_list(ns, tests_with_optional_deps))
    else:
        globals().update(ns)
        unittest.main()

