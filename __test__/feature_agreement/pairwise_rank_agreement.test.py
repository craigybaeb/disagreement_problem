import unittest
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from FeatureAgreement import FeatureAgreement

class TestPairwiseRankAgreement(unittest.TestCase):
  
    def test_identical_f6(self):
        '''Testing that pairwise rank agreement of identical cases is 1 with all features considered'''
        self.fa = FeatureAgreement()
        T1 = [1,2,3,4,5,6]
        T2 = [1,2,3,4,5,6]
        F = [0,1,2,3,4,5]
        result = self.fa.pairwise_rank_agreement(T1, T2, F)
        expected = 1
        self.assertEqual(result, expected)

    def test_identical_f3(self):
        '''Testing that pairwise rank agreement of identical cases is 1 with half features considered'''
        self.fa = FeatureAgreement()
        T1 = [1,2,3,4,5,6]
        T2 = [1,2,3,4,5,6]
        F = [0,1,2]
        result = self.fa.pairwise_rank_agreement(T1, T2, F)
        expected = 1
        self.assertEqual(result, expected)

    def test_identical_f3_2(self):
        '''Testing that pairwise rank agreement of identical cases is 1 with half features considered'''
        self.fa = FeatureAgreement()
        T1 = [1,2,3,4,5,6]
        T2 = [3,2,1,4,5,6]
        F = [0,1,2]
        result = self.fa.pairwise_rank_agreement(T1, T2, F)
        expected = 0
        self.assertEqual(result, expected)

    def test_identical_f3_3(self):
        '''Testing that pairwise rank agreement of identical cases is 1 with half features considered'''
        self.fa = FeatureAgreement()
        T2 = [1,2,3,4,5,6]
        T1 = [3,2,1,4,5,6]
        F = [0,1,2]
        result = self.fa.pairwise_rank_agreement(T1, T2, F)
        expected = 0
        self.assertEqual(result, expected)

    def test_rank_f6(self):
        '''Testing that pairwise rank agreement of identical cases is 1 with all features considered (checking rank is important)'''
        self.fa = FeatureAgreement()
        T1 = [1,2,3,4,5,6]
        T2 = [6,5,4,3,2,1]
        F = [0,1,2,3,4,5]
        result = self.fa.pairwise_rank_agreement(T1, T2, F)
        expected = 0
        self.assertEqual(result, expected)

    def test_rank_f3(self):
        '''Testing that pairwise rank agreement of identical cases is 1 with all features considered (checking rank is important)'''
        self.fa = FeatureAgreement()
        T1 = [1,2,3,4,5,6]
        T2 = [6,5,4,3,2,1]
        F = [0,1,2]
        result = self.fa.pairwise_rank_agreement(T1, T2, F)
        expected = 0
        self.assertEqual(result, expected)

    def test_unequal_length_f6(self):
        '''Testing that pairwise rank agreement of identical cases is 1 with all features considered (checking unequal lengths)'''
        self.fa = FeatureAgreement()
        T1 = [1,2,3,4,5,6]
        T2 = [1,2,3,4,5,6,7]
        F = [0,1,2,3,4,5]
        result = self.fa.pairwise_rank_agreement(T1, T2, F)
        expected = 1
        self.assertEqual(result, expected)

    def test_f0(self):
        '''Testing that an error is thrown when len(F) =0'''
        self.fa = FeatureAgreement()
        T1 = [1, 2, 3, 4, 5, 6]
        T2 = [1, 2, 3, 4, 5, 6]
        F = []

        with self.assertRaises(Exception) as context:
            result = self.fa.pairwise_rank_agreement(T1, T2, F)
            self.assertTrue('Error: Must contain at least one feature.' in context.exception)

unittest.main(argv=[''], verbosity=2, exit=False)