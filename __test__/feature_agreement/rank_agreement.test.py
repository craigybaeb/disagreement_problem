import unittest
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from FeatureAgreement import FeatureAgreement

class TestRankAgreement(unittest.TestCase):
  
    def test_identical_k6(self):
        '''Testing that rank agreement of identical cases is 1 with all features considered'''
        self.fa = FeatureAgreement()
        T1 = [1,2,3,4,5,6]
        T2 = [1,2,3,4,5,6]
        K = 6
        result = self.fa.rank_agreement(T1, T2, K)
        expected = 1
        self.assertEqual(result, expected)
    
    def test_identical_k3(self):
        '''Testing that rank agreement of identical cases is 1 with half of features considered'''
        self.fa = FeatureAgreement()
        T1 = [1,2,3,4,5,6]
        T2 = [1,2,3,4,5,6]
        K = 3
        result = self.fa.rank_agreement(T1, T2, K)
        expected = 1
        self.assertEqual(result, expected)

    def test_unranked_k6(self):
        '''Testing that rank agreement of identical cases is 0 when all features are considered but ranking is entirely different'''
        self.fa = FeatureAgreement()
        T1 = [1,2,3,4,5,6]
        T2 = [6,5,4,3,2,1]
        K = 6
        result = self.fa.rank_agreement(T1, T2, K)
        expected = 0
        self.assertEqual(result, expected)

    def test_unranked_k3(self):
        '''Testing that rank agreement of identical cases is 0 with half of features are considered but ranking is entirely different'''
        self.fa = FeatureAgreement()
        T1 = [1,2,3,4,5,6]
        T2 = [6,5,4,3,2,1]
        K = 3
        result = self.fa.rank_agreement(T1, T2, K)
        expected = 0
        self.assertEqual(result, expected)

    def test_unequal_length(self):
        '''Testing that rank agreement still works with unequal length vectors'''
        self.fa = FeatureAgreement()
        T1 = [1, 2, 3]
        T2 = [1, 2, 3, 4, 5, 6]
        K = 3
        result = self.fa.rank_agreement(T1, T2, K)
        expected = 0
        self.assertEqual(result, expected)

    def test_unequal_length_k6(self):
        '''Testing that rank agreement still works with unequal length vectors, with k > possible features of smallest vector'''
        self.fa = FeatureAgreement()
        T1 = [1, 2, 3]
        T2 = [1, 2, 3, 4, 5, 6]
        K = 6
        result = self.fa.rank_agreement(T1, T2, K)
        expected = 0.5
        self.assertEqual(result, expected)

    def test_k0(self):
        '''Testing that an error is thrown when k=0'''
        self.fa = FeatureAgreement()
        T1 = [1, 2, 3, 4, 5, 6]
        T2 = [1, 2, 3, 4, 5, 6]
        K = 0

        with self.assertRaises(Exception) as context:
            result = self.fa.rank_agreement(T1, T2, K)
            self.assertTrue('Error: k must be >=1 ' in context.exception)



unittest.main(argv=[''], verbosity=2, exit=False)