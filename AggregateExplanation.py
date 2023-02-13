import numpy as np

class AggregateExplanation:
    def __init__(self, caseAlignments, importances, explanationMethods):
        self.caseAlignments = caseAlignments
        self.aggregateExplanation = []
        self.confidence = 0.00
        self.explanationMethods = explanationMethods
        self.importances = importances
        self.explainerConfidences = {}

        return

    def displayExplanation(self, showConfidence=True):
        print(f'Aggregate Explanation: {self.aggregateExplanation}')

        if(showConfidence==True):
            print(f'Confidence: {round(self.confidence * 100, 2)}%')

        return 

    def __str__(self):
        displayExplanation()
        return
    
    #Sorts into ordinal list of numbers, irrespective of sign. 
    #1 is most important, 0 is least important.
    def rank(self, attributions):
        RANKED = sorted(range(len(attributions)), key = lambda sub: abs(attributions[sub]))
        return RANKED

    def getAggregateExplanation(self, mode='median'): 
        if(mode == 'median'):
            WEIGHTED_EXPLAINERS = list(map(lambda explainer: self.importances[explainer] * self.explainerConfidences[explainer], self.explanationMethods))

            #Gets median ranks for each column in a vector of feature attributions
            medianRanks = []
            for attribution in range(len(self.importances[next(iter(self.importances))])):
                WEIGHTED_EXPLAINER_COLUMNS = list(map(lambda weightedExplainer: weightedExplainer[attribution], WEIGHTED_EXPLAINERS))
                MEDIAN_RANK = np.median(WEIGHTED_EXPLAINER_COLUMNS)
                medianRanks.append(MEDIAN_RANK)

            #Re-ranks the features by order of importance
            self.aggregateExplanation = self.rank(medianRanks)
        elif(mode == 'weighted_average'):
            #Gets weighted average ranks for each column in a vector of feature attributions
            averageRanks = []
            for attribution in range(len(self.importances[next(iter(self.importances))])):
                EXPLAINER_COLUMNS = list(map(lambda explainer: self.importances[explainer][attribution], self.explanationMethods))
                
                weightedExplanationSum = 0
                weightSum = 0
                for explainer in self.explanationMethods:
                    weightedExplanationSum += (self.importances[explainer][attribution] * self.explainerConfidences[explainer])
                    weightSum += self.explainerConfidences[explainer]
                
                WEIGHTED_AVERAGE = weightedExplanationSum / weightSum
                averageRanks.append(WEIGHTED_AVERAGE)

            #Re-ranks the features by order of importance
            self.aggregateExplanation = self.rank(averageRanks)
        else:
            raise Exception('Invalid mode received. Choose between median or weighted_average.')
        return

    def calculateExplainerConfidence(self):
        explainerConfidences = {}

        for explainer in self.explanationMethods:
            explainerConfidences[explainer] = 0
        
        for firstExplainer in self.explanationMethods:
            for secondExplainer in self.explanationMethods:
                explainerConfidences[firstExplainer] += self.caseAlignments[firstExplainer][secondExplainer]

                if(not firstExplainer == secondExplainer):
                  explainerConfidences[secondExplainer] += self.caseAlignments[firstExplainer][secondExplainer]

        explainerConfidences.update((x, y / ((len(self.importances) * 2) - 1)) for x, y in explainerConfidences.items())
        self.explainerConfidences = explainerConfidences
        return
    
    def calculateTotalConfidence(self):
        self.confidence = sum(self.explainerConfidences.values()) / len(self.explanationMethods)
        return