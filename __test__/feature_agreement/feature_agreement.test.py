import unittest
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from FeatureAgreement import FeatureAgreement

class TestFeatureAgreement(unittest.TestCase):
  
    def test_identical_k6(self):
        '''Testing that feature agreement of identical cases is 1 with all features considered'''
        self.fa = FeatureAgreement()
        T1 = [1,2,3,4,5,6]
        T2 = [1,2,3,4,5,6]
        K = 6
        result = self.fa.feature_agreement(T1, T2, K)
        expected = 1
        self.assertEqual(result, expected)
    
    def test_identical_k3(self):
        '''Testing that feature agreement of identical cases is 1 with half of features considered'''
        self.fa = FeatureAgreement()
        T1 = [1,2,3,4,5,6]
        T2 = [1,2,3,4,5,6]
        K = 3
        result = self.fa.feature_agreement(T1, T2, K)
        expected = 1
        self.assertEqual(result, expected)

    def test_non_identical_k6(self):
        '''Testing that feature agreement of non-identical cases is 1 with all features considered'''
        self.fa = FeatureAgreement()
        T1 = [0.1, 0.5, 0.4, 0.6]
        T2 = [0.2, 0.5, 0.4, 0.3]
        K = 6
        result = self.fa.feature_agreement(T1, T2, K)
        expected = 1
        self.assertEqual(result, expected)
    
    def test_non_identical_k2(self):
        '''Testing that feature agreement of non-identical cases is 0.5 with half of features considered'''
        self.fa = FeatureAgreement()
        T1 = [0.1, 0.5, 0.4, 0.6]
        T2 = [0.2, 0.5, 0.4, 0.3]
        K = 2
        result = self.fa.feature_agreement(T1, T2, K)
        expected = 0.5
        self.assertEqual(result, expected)

    def test_sign_k6(self):
        '''Testing that feature agreement of identical cases is 1 with all features considered and irrespective of sign'''
        self.fa = FeatureAgreement()
        T1 = [1,2,3,4,5,6]
        T2 = [-1,-2,-3,-4,-5,-6]
        K = 6
        result = self.fa.feature_agreement(T1, T2, K)
        expected = 1
        self.assertEqual(result, expected)
    
    def test_sign_k3(self):
        '''Testing that feature agreement of identical cases is 1 with half of features considered and irrespective of sign'''
        self.fa = FeatureAgreement()
        T1 = [1,2,3,4,5,6]
        T2 = [-1,-2,-3,-4,-5,-6]
        K = 3
        result = self.fa.feature_agreement(T1, T2, K)
        expected = 1
        self.assertEqual(result, expected)

    def test_k0(self):
        '''Testing that an error is thrown when k=0'''
        self.fa = FeatureAgreement()
        T1 = [1, 2, 3, 4, 5, 6]
        T2 = [1, 2, 3, 4, 5, 6]
        K = 0

        with self.assertRaises(Exception) as context:
            result = self.fa.feature_agreement(T1, T2, K)
            self.assertTrue('Error: k must be >=1 ' in context.exception)

    def test_unequal_length(self):
        '''Testing that feature agreement still works with unequal length vectors'''
        self.fa = FeatureAgreement()
        T1 = [1, 2, 3]
        T2 = [1, 2, 3, 4, 5, 6]
        K = 3
        result = self.fa.feature_agreement(T1, T2, K)
        expected = 0
        self.assertEqual(result, expected)

    def test_unequal_length_k6(self):
        '''Testing that feature agreement still works with unequal length vectors, with k > possible features of smallest vector'''
        self.fa = FeatureAgreement()
        T1 = [1, 2, 3]
        T2 = [1, 2, 3, 4, 5, 6]
        K = 6
        result = self.fa.feature_agreement(T1, T2, K)
        expected = 0
        self.assertEqual(result, expected)

unittest.main(argv=[''], verbosity=2, exit=False)