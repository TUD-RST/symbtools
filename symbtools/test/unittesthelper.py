"""
This modules provides some auxiliary functions for test execution.
1. Handle command line args which are passed to the test_* scripts.
2. Enable the command: python -c "from symbtools.test import run_all; run_all()".

It should be loaded before the `untitest` module as it alters sys.argv which seems to be evaluated by that module
"""

import sys
import os
import types
import inspect
import importlib


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


# this has to be executed before `import unittest`
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
    Iterate trough modules and insert all subclasses of TestCase into target namespace.

    :param target_ns:           dict (usually `globals()`)
    :param list_of_modules:     sequence of modules
    :return:                    None
    """

    if isinstance(list_of_modules, types.ModuleType):
        list_of_modules = [list_of_modules]

    for m in list_of_modules:
        assert isinstance(m, types.ModuleType)

    # book-keeping where which test comes from
    # (maybe injected by an earlier call)
    name_mod_dict = target_ns.get("__name_mod_dict", {})

    for mod in list_of_modules:
        for k, v in vars(mod).items():
            if isinstance(v, type):
                if issubclass(v, unittest.TestCase):
                    # name = "{}{}".format(mod.__name__, v.__name__)
                    name = v.__name__
                    # keep track of in which module a name was defined first
                    if name in target_ns:
                        first_mod_name = name_mod_dict.get(name, "<target_ns>")
                        msg = "Name-Conflict: '{}' is defined in both {} and {} ".format(name, mod.__name__,
                                                                                         first_mod_name)
                        raise NameError(msg)
                    assert name not in target_ns
                    name_mod_dict[name] = mod.__name__
                    target_ns[name] = v

    target_ns["__name_mod_dict"] = name_mod_dict


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


def run_all():
    """
    This function enables the command: python -c "from symbtools.test import run_all; run_all()"
    (see also test/__init__.py)

    :return: None
    """

    mod_name = __name__.split('.')[0]
    release = importlib.import_module(mod_name+".release")

    print("Running all tests for module `{}` {}.".format(mod_name, release.__version__))

    current_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

    loader = unittest.TestLoader()
    suite = loader.discover(current_path)

    runner = unittest.TextTestRunner()
    res = runner.run(suite)

    # cause CI to fail if tests have failed (otherwise this script returns 0 despite of failing tests)
    assert res.wasSuccessful()
