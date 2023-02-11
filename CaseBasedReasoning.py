from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

#Contain the case-base within a class
class CBR:
  #Initialise the case-base data and k-NN model
  def __init__(self, categorical, neighbours, filepath='', raw_data=[], inputDict={}, limit=1):
    self.data = self.load_data(categorical, filepath, raw_data)
    self.model = self.fit_model(neighbours)
    self.inputDict = inputDict
    self.limit = limit

  #Fit the k-NN case-base
  def fit_model(self, neighbours):
    knn = NearestNeighbors(n_neighbors=len(self.data))
    knn.fit(self.data)

    return knn

  #Read the data from the CSV file into a Pandas dataframe
  def load_data(self, categorical, filepath='', raw_data=[]):
    if(len(raw_data) > 0):
      return raw_data
    else: 
      data = pd.read_csv(filepath)
    
    #One-hot encode all categorical data as k-NN is numeric
    ohe = OneHotEncoder()
    transformed = ohe.fit_transform(data[categorical])

    return transformed

  #Retrieve the neighbours for a query along with its distance
  def retrieve(self, index, num_neighbours):
    NEIGHBOURS_AND_DISTANCES = self.model.kneighbors(index.reshape(1,-1), n_neighbors=round(num_neighbours * self.limit), return_distance=True)
    DISTANCES = NEIGHBOURS_AND_DISTANCES[0]
    NEIGHBOURS = NEIGHBOURS_AND_DISTANCES[1]

    #Convert the neighbour-distances 2-D array into a 1D array of tuples
    # e.g (neighbour, distance)
    TRANSPOSED = np.array((NEIGHBOURS,DISTANCES)).T
    return TRANSPOSED