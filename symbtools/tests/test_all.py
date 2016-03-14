# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 11:35:00 2014

@author: Carsten Knoll
"""

import sys

if 'all' in sys.argv:
    FLAG_all = True
else:
    FLAG_all = False

from test_core import *
from test_modeltools import *
from test_nctools import *
from test_quick import *

import test_core, test_modeltools, test_nctools, test_quick

    
def main():
    # remove command line args which should not be passed to the testframework
    if 'all' in sys.argv:
        sys.argv.remove('all')
    
    unittest.main()

# see also the skip_slow logic at the beginning of the file
if __name__ == '__main__':
    main()
