import os
import unittest

def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    this_dir = os.path.dirname(__file__)
    package_tests = loader.discover(start_dir=this_dir, pattern='*_test.py')
    suite.addTests(package_tests)
    return suite

if __name__ == '__main__':
    unittest.main()
