# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 11:35:00 2014

@author: Carsten Knoll
"""

import unittesthelper as uth

import test_core
import test_modeltools
import test_nctools
import test_quick
import test_meshtools


def main():
    modules = [test_core, test_modeltools, test_nctools, test_quick, test_meshtools]

    if uth.FLAGS.optdep:
        import test_mpctools
        modules.append(test_mpctools)

    uth.inject_tests_into_namespace(globals(), modules)
    uth.smart_run_tests_in_ns(globals())


if __name__ == '__main__':
    main()
