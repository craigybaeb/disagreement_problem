import unittest
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from FeatureAgreement import FeatureAgreement

class TestSignAgreement(unittest.TestCase):
  
    def test_identical_positive_k6(self):
        '''Testing that sign agreement of identical positive cases is 1 with all features considered'''
        self.fa = FeatureAgreement()
        T1 = [1,2,3,4,5,6]
        T2 = [1,2,3,4,5,6]
        K = 6
        result = self.fa.sign_agreement(T1, T2, K)
        expected = 1
        self.assertEqual(result, expected)
    
    def test_identical__positive_k3(self):
        '''Testing that sign agreement of identical positive cases is 1 with half of features considered'''
        self.fa = FeatureAgreement()
        T1 = [1,2,3,4,5,6]
        T2 = [1,2,3,4,5,6]
        K = 3
        result = self.fa.sign_agreement(T1, T2, K)
        expected = 1
        self.assertEqual(result, expected)

    def test_identical_negative_k6(self):
        '''Testing that sign agreement of identical negative cases is 1 with all features considered'''
        self.fa = FeatureAgreement()
        T1 = [-1,-2,-3,-4,-5,-6]
        T2 = [-1,-2,-3,-4,-5,-6]
        K = 6
        result = self.fa.sign_agreement(T1, T2, K)
        expected = 1
        self.assertEqual(result, expected)
    
    def test_identical__negative_k3(self):
        '''Testing that sign agreement of identical negative cases is 1 with half of features considered'''
        self.fa = FeatureAgreement()
        T1 = [-1,-2,-3,-4,-5,-6]
        T2 = [-1,-2,-3,-4,-5,-6]
        K = 3
        result = self.fa.sign_agreement(T1, T2, K)
        expected = 1
        self.assertEqual(result, expected)

    def test_identical_mirrored_signs_k6(self):
        '''Testing that sign agreement of identical cases but opposite signs is 0 with all features considered'''
        self.fa = FeatureAgreement()
        T1 = [-1,-2,-3,-4,-5,-6]
        T2 = [1,2,3,4,5,6]
        K = 6
        result = self.fa.sign_agreement(T1, T2, K)
        expected = 0
        self.assertEqual(result, expected)
    
    def test_identical__mirrored_signs_k3(self):
        '''Testing that sign agreement of identical cases but opposite signs is 0  with half of features considered'''
        self.fa = FeatureAgreement()
        T1 = [-1,-2,-3,-4,-5,-6]
        T2 = [1,2,3,4,5,6]
        K = 3
        result = self.fa.sign_agreement(T1, T2, K)
        expected = 0
        self.assertEqual(result, expected)

    def test_mixed_k6(self):
        '''Testing that sign agreement of identical mixed cases is 1 when all features are considered'''
        self.fa = FeatureAgreement()
        T1 = [-1,-2,3,4,-5,6]
        T2 = [-1,-2,3,4,-5,6]
        K = 6
        result = self.fa.sign_agreement(T1, T2, K)
        expected = 1
        self.assertEqual(result, expected)

    def test_mixed_k3(self):
        '''Testing that sign agreement of identical mixed cases is 1 with half of features are considered'''
        self.fa = FeatureAgreement()
        T1 = [-1,-2,3,4,-5,6]
        T2 = [-1,-2,3,4,-5,6]
        K = 3
        result = self.fa.sign_agreement(T1, T2, K)
        expected = 1
        self.assertEqual(result, expected)

    def test_mixed_unequal_k6(self):
        '''Testing that sign agreement of completely different mixed cases is 0 when all features are considered'''
        self.fa = FeatureAgreement()
        T1 = [1,-2,3,4,-5,6]
        T2 = [-600, 500, -400, -300, 200, -100]
        K = 6
        result = self.fa.sign_agreement(T1, T2, K)
        expected = 0
        self.assertEqual(result, expected)

    def test_mixed_unequal_k3(self):
        '''Testing that sign agreement of completely different mixed cases is 0 with half of features are considered'''
        self.fa = FeatureAgreement()
        T1 = [1,-2,3,4,-5,6]
        T2 = [-600, 500, -400, -300, 200, -100]
        K = 3
        result = self.fa.sign_agreement(T1, T2, K)
        expected = 0
        self.assertEqual(result, expected)

    def test_unequal_length(self):
        '''Testing that sign agreement still works with unequal length vectors'''
        self.fa = FeatureAgreement()
        T1 = [-1, 2, 3]
        T2 = [-1, 2, 3, 4, 5, 6]
        K = 3
        result = self.fa.sign_agreement(T1, T2, K)
        expected = 0
        self.assertEqual(result, expected)

    def test_unequal_length_k6(self):
        '''Testing that sign agreement still works with unequal length vectors, with k > possible features of smallest vector'''
        self.fa = FeatureAgreement()
        T1 = [-1, 2, 3]
        T2 = [-1, 2, 3, 4, 5, 6]
        K = 6
        result = self.fa.sign_agreement(T1, T2, K)
        expected = 0.5
        self.assertEqual(result, expected)

    def test_k0(self):
        '''Testing that an error is thrown when k=0'''
        self.fa = FeatureAgreement()
        T1 = [1, 2, 3, 4, 5, 6]
        T2 = [1, 2, 3, 4, 5, 6]
        K = 0

        with self.assertRaises(Exception) as context:
            result = self.fa.sign_agreement(T1, T2, K)
            self.assertTrue('Error: k must be >=1 ' in context.exception)



unittest.main(argv=[''], verbosity=2, exit=False)