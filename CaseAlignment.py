import numpy as np
import pickle 
from tqdm import tqdm

class CaseAlignment:
    def __init__(self, num_neighbours):
        self.caseAlignments = {}
        self.num_neighbours = num_neighbours

    #Get the distance between a query and case in the neighbourhood space
    def getDistance(self, query, neighbourCase, cbr):
        if(np.array_equal(query,neighbourCase)):
            return 0
            
        NEIGHBOURS = np.squeeze(cbr.retrieve(query, len(cbr.data)))
        NEIGHBOURS_WITHOUT_SELF = NEIGHBOURS[1:] #Exclude self from calculation
        DISTANCE = NEIGHBOURS_WITHOUT_SELF[NEIGHBOURS_WITHOUT_SELF[:,0] == self.findKey(cbr.inputDict, neighbourCase)][0][1]
        
        return DISTANCE

    #Get the max distance in the neighbourhood space
    def getMaxDistance(self, query, cbr):
        NEIGHBOURS = cbr.retrieve(query, len(cbr.data)).reshape(len(cbr.data),2)
        NEIGHBOURS_WITHOUT_SELF = NEIGHBOURS[1:]
        MAX_DISTANCE = max(NEIGHBOURS_WITHOUT_SELF[:,1])

        return MAX_DISTANCE

    #Get the minimum distance in the neighbourhood space
    def getMinDistance(self, query, cbr):
        NEIGHBOURS = cbr.retrieve(query, len(cbr.data)).reshape(len(cbr.data),2)
        NEIGHBOURS_WITHOUT_SELF = NEIGHBOURS[1:]
        MIN_DISTANCE = min(NEIGHBOURS_WITHOUT_SELF[:,1])

        return MIN_DISTANCE

    #Get the alignment score for an attribution method
    def alignmentScore(self, query, neighbourCase, cbr):
        MAX_DISTANCE = self.getMaxDistance(query, cbr)
        MIN_DISTANCE = self.getMinDistance(query, cbr)

        if(np.array_equal(query, neighbourCase)):
            MIN_DISTANCE = 0
            
            
        CASE_DISTANCE = self.getDistance(query, neighbourCase, cbr)

        #Normalised alignment
        ALIGNMENT = 1 - ((CASE_DISTANCE - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE))
        
        return ALIGNMENT

    #Get the case alignment score for an attribution method
    def caseAlignmentScore(self, query, cbr_a1, cbr_a2, show_progress=False):
        weighted_problem_distance_total = 0 #Numerator
        problem_distance_total = 0 #Denominator

        cases = cbr_a1.retrieve(cbr_a1.inputDict[query], self.num_neighbors + 1)[1:] 

        if(show_progress):
            cases = tqdm(cases)

        #Calculate distance for each neighbour
        for neighbourCase in cases:
            CASE = neighbourCase[:, 0][0]

            WEIGHTED_PROBLEM_DISTANCE = (1 - self.getDistance(cbr_a1.inputDict[query], cbr_a1.inputDict[CASE], cbr_a1)) * self.alignmentScore(cbr_a2.inputDict[query], cbr_a2.inputDict[CASE], cbr_a2) #Weight by alignment of comparison attribution method
            PROBLEM_DISTANCE = 1 - self.getDistance(cbr_a1.inputDict[query], cbr_a1.inputDict[CASE], cbr_a1)

            weighted_problem_distance_total += WEIGHTED_PROBLEM_DISTANCE
            problem_distance_total += PROBLEM_DISTANCE

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