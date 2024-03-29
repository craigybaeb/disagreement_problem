import numpy as np
import pickle 
from tqdm import tqdm

class CaseAlignment:
    def __init__(self, num_neighbours):
        self.caseAlignments = {}
        self.num_neighbours = num_neighbours

    #Get the distance between a query and case in the neighbourhood space
    def getDistance(self, query, neighbourCase, cbr):
        NEIGHBOURS = np.squeeze(cbr.retrieve(query, cbr.inputDict[query], len(cbr.data)))
        DISTANCE = NEIGHBOURS[NEIGHBOURS[:,0] == neighbourCase][0][1]
        
        return DISTANCE

    #Get the max and min distances in the neighbourhood space
    def getMaxMinDistances(self, query, cbr):
        NEIGHBOURS = cbr.retrieve(query, cbr.inputDict[query], len(cbr.data)).squeeze()
        MAX_DISTANCE = max(NEIGHBOURS[:,1])
        MIN_DISTANCE = min(NEIGHBOURS[:,1])

        return MAX_DISTANCE, MIN_DISTANCE

    #Get the alignment score for an attribution method
    def alignmentScore(self, query, neighbourCase, cbr):
        MAX_DISTANCE, MIN_DISTANCE = self.getMaxMinDistances(query, cbr)
        CASE_DISTANCE = self.getDistance(query, neighbourCase, cbr)

        #Normalised alignment
        ALIGNMENT = 1 - ((CASE_DISTANCE - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE))
        
        return ALIGNMENT

    #Get the case alignment score for an attribution method
    def caseAlignmentScore(self, query, cbr_a1, cbr_a2, show_progress=False):
        weighted_problem_distance_total = 0 #Numerator
        problem_distance_total = 0 #Denominator

        cases = cbr_a1.retrieve(query, cbr_a1.inputDict[query], self.num_neighbours + 1)[1:] 

        if(show_progress):
            cases = tqdm(cases)

        #Calculate distance for each neighbour
        for neighbourCase in cases:
            CASE = neighbourCase[:, 0][0]

            DISTANCE = self.getDistance(query, CASE, cbr_a1)
            ALIGNMENT = self.alignmentScore(query, CASE, cbr_a2)
            WEIGHTED_PROBLEM_DISTANCE = (1 - DISTANCE) * ALIGNMENT
            PROBLEM_SIMILARITY = 1 - DISTANCE

            weighted_problem_distance_total += WEIGHTED_PROBLEM_DISTANCE
            problem_distance_total += PROBLEM_SIMILARITY

        CASE_ALIGNMENT = weighted_problem_distance_total / problem_distance_total
        return CASE_ALIGNMENT
  
    #Get the casebase alignment score for an attribution method
    def caseBaseAlignment(self, cbr_a1, cbr_a2, show_progress=True):
        case_base_alignment = 0
        cases = range(len(cbr_a1.data))

        if(show_progress):
            cases = tqdm(cases)

        #Calculate distance for each neighbour
        for neighbourCase in cases:
            CASE_ALIGNMENT = self.caseAlignmentScore(neighbourCase, cbr_a1, cbr_a2)
            case_base_alignment += CASE_ALIGNMENT
        return case_base_alignment / len(cbr_a1.data)

    def saveAlignments(self, filepath, alignments):
        file = open(filepath, 'wb') 
        pickle.dump(alignments, file)

    def findKey(self, dataDict, valueToFind):
        found = -1

        for key, value in dataDict.items():
            if(np.array_equal(value, valueToFind)):
                found = key

        return found