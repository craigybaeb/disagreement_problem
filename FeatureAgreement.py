from scipy.stats import spearmanr as rank
from scipy.stats import rankdata
import math
import numpy as np

binom = math.comb
class FeatureAgreement:
    def mostImportantFeatures(self, features, k):
        MAX = sorted(range(len(features)), key = lambda sub: abs(features[sub]))[-k:]
        return MAX

    def feature_agreement(self, a1, a2, k):
        if(k == 0):
            raise Exception('Error: k must be >= 1')

        if(k > len(a1) or k > len(a1)):
            k = min([len(a1), len(a2)])

        A1_TOP = self.mostImportantFeatures(a1, k)
        A2_TOP = self.mostImportantFeatures(a2, k)
        
        FEATURE_ALIGNMENT = len(set(A1_TOP[:k]).intersection(A2_TOP[:k])) / k
        
        return FEATURE_ALIGNMENT

    def rank_agreement(self, a1, a2, k):

        if(k == 0):
            raise Exception('Error: k must be >= 1')

        A1_TOP = self.mostImportantFeatures(a1, k)
        A2_TOP = self.mostImportantFeatures(a2, k)

        UNION = []

        FEATURES = set(A1_TOP[:k]).intersection(A2_TOP[:k])
        
        for feature in FEATURES:
            A1_RANK = A1_TOP.index(feature)
            A2_RANK = A2_TOP.index(feature)

            if(A1_RANK == A2_RANK):
                UNION.append(feature)

        RANK_AGREEMENT = len(UNION) / k
        return RANK_AGREEMENT

    def sign_agreement(self,a1, a2, k):
        if(k == 0):
            raise Exception('Error: k must be >= 1')

        A1_TOP = self.mostImportantFeatures(a1, k)
        A2_TOP = self.mostImportantFeatures(a2, k)

        UNION = []
        FEATURES = set(A1_TOP[:k]).intersection(A2_TOP[:k])

        for feature in FEATURES:
            A1_POSITIVE = a1[feature] > 0
            A2_POSITIVE = a2[feature] > 0

            signs_match = False

            if(A1_POSITIVE and A2_POSITIVE):
                signs_match = True
            elif(not A1_POSITIVE and not A2_POSITIVE):
                signs_match = True

            if(signs_match):
                UNION.append(feature)

        SIGNED_AGREEMENT = len(UNION) / k
        return SIGNED_AGREEMENT

    def sign_rank_agreement(self,a1, a2, k):
        if(k == 0):
            raise Exception('Error: k must be >= 1')

        A1_TOP = self.mostImportantFeatures(a1, k)
        A2_TOP = self.mostImportantFeatures(a2, k)

        UNION = []

        FEATURES = set(A1_TOP[:k]).intersection(A2_TOP[:k])
        
        for feature in FEATURES:
            A1_POSITIVE = a1[feature] > 0
            A2_POSITIVE = a2[feature] > 0

            A1_RANK = A1_TOP.index(feature)
            A2_RANK = A2_TOP.index(feature)

            signs_match = False

            if(A1_POSITIVE and A2_POSITIVE):
                signs_match = True
            elif(not A1_POSITIVE and not A2_POSITIVE):
                signs_match = True

            if(signs_match and A1_RANK == A2_RANK):
                UNION.append(feature)

        SIGNED_RANK_AGREEMENT = len(UNION) / k
        return SIGNED_RANK_AGREEMENT

    #A1: First attribution set
    #A2: Second attribution set
    #F: Features of interest
    def rank_correlation(self,A1,A2,F):
        if(len(F) == 0):
            raise Exception('Error: Must include at least one feature.')

        RANKING_A1 = np.array(A1)[F] #Get only features of interest for first attribution method
        RANKING_A2 = np.array(A2)[F] #Get only features of interest for second attribution method

        #Get Spearman's rank correlation of two attribution sets
        RANK_CORRELATION = rank(RANKING_A1, RANKING_A2)[0]
        
        return RANK_CORRELATION

    #A1: First attribution set
    #A2: Second attribution set
    #F: Features of interest
    def pairwise_rank_agreement(self,A1, A2, F):
        if(len(F) == 0):
            raise Exception('Error: Must include at least one feature.')

        A1_RANKED = rankdata(A1)[F] #Get only features of interest for first attribution method
        A2_RANKED = rankdata(A2)[F] #Get only features of interest for second attribution method

        #Initialise summation
        sigma = 0

        #Do indicator function for each pair of features of interest
        for i in range(len(F)):
            for j in range(len(F)):
            
                #Initialise indicator function values
                relativeRankingA1 = 0 
                relativeRankingA2 = 0
                
                #If features are the same, skip
                if(i == j):
                    continue

                #First attribution method indicator function check
                if(A1_RANKED[i] > A1_RANKED[j]):
                    relativeRankingA1 += 1

                #Second attribution method indicator function check
                if(A2_RANKED[i] > A2_RANKED[j]):
                    relativeRankingA2 += 1

                #Combined indicator function check
                if(relativeRankingA1 == relativeRankingA2):
                    sigma += 1

        BINOM = binom (len(F), 2) #Binomial co-efficient of number of features of interest, choose 2
        PAIRWISE_RANK_AGREEMENT = sigma / (BINOM * 2) #Calculate pairwise rank agreement

        return PAIRWISE_RANK_AGREEMENT

    def calcMeanAgreement(self, agreementMethod, data_prob, data_sol, k, f=[]):
        agreementTotal = 0

        if(len(f) > 0):
            for sample in range(len(data_prob)):
                if(np.array_equal(data_prob, data_sol)):
                    agreement = 1
                else:
                     agreement = agreementMethod(data_prob[sample], data_sol[sample], f)
               
                agreementTotal += agreement
        else:
            for sample in range(len(data_prob) - 1):
                agreement = agreementMethod(data_prob[sample], data_sol[sample], k)
                agreementTotal += agreement
        
        return agreementTotal / (len(data_prob) - 1)