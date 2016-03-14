# -*- coding: utf-8 -*-
"""
Created on Mo 2016-03-14

@author: Carsten Knoll

This file contains just dummy tests to quickly test the test-infrastructure
"""

import unittest, sys



if 'all' in sys.argv:
    FLAG_all = True
else:
    FLAG_all = False

# own decorator for skipping slow tests
def skip_slow(func):
    return unittest.skipUnless(FLAG_all, 'skipping slow test')(func)


class NCTTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_func1(self):
        self.assertTrue(1)
        
    @skip_slow
    def test_func2(self):
        self.assertTrue(1)
        

def main():
    # remove command line args which should not be passed to the testframework
    if 'all' in sys.argv:
        sys.argv.remove('all')
    
    unittest.main()

if __name__ == '__main__':
    main()