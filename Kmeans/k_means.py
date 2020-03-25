import numpy as np
import matplotlib.pyplot as plt
from random import randrange

def scale(data, min_data, max_data):
  difference = max_data - min_data
  data -= min_data
  data = float(data)/float(difference)
  return data

def get_distance(node, cluster):
  return (pow(node.data[0] - cluster.mean[0], 2)) + (pow(node.data[1] - cluster.mean[1], 2)) + (pow(node.data[2] - cluster.mean[2], 2))\
         + (pow(node.data[3] - cluster.mean[3], 2)) + (pow(node.data[4] - cluster.mean[4], 2)) + (pow(node.data[5] - cluster.mean[5], 2))\
         + (pow(node.data[6] - cluster.mean[6], 2))  

data = np.genfromtxt("dataset1.csv", delimiter=";", usecols=[1,2,3,4,5,6,7], converters={5:
lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

dates = np.genfromtxt("dataset1.csv", delimiter=";", usecols=[0])
labels = []
for label in dates:
  if label < 20000301:
    labels.append("winter")
  elif 20000301 <= label < 20000601:
    labels.append("lente")
  elif 20000601 <= label < 20000901:
    labels.append("zomer")
  elif 20000901 <= label < 20001201:
    labels.append("herfst")
  else: # from 01-12 to end of year
    labels.append("winter")

class node:
  def __init__(self, index, data):
    self.index = index
    self.data = data

class centroid:
  def __init__(self, mean):
    self.mean = mean
    self.label = None
    self.node_indexes = []

  def calculate_mean(self):
    if(len(self.node_indexes) >= 1):
      self.mean = [0,0,0,0,0,0,0]
      for i in range(len(self.node_indexes)):
        for j in range(len(data[self.node_indexes[i]])):
          self.mean[j] += data[self.node_indexes[i]][j]
      
      for i in range(len(self.mean)):
        self.mean[i] /= len(self.node_indexes)
  
  def get_intra_cluster_distance(self):
    distance = 0
    for node_ in self.node_indexes:
      distance += get_distance(node(node_, data[node_]), self)
    return distance

  def calculate_label(self):
    amount_winter = 0
    amount_herfst = 0
    amount_zomer = 0
    amount_lente = 0
    for i in range(len(self.node_indexes)):
      current_label = labels[self.node_indexes[i]]
      if current_label == "winter":
        amount_winter += 1
      elif current_label == "herfst":
        amount_herfst += 1
      elif current_label == "zomer":
        amount_zomer += 1
      elif current_label == "lente":
        amount_lente += 1
    
    if amount_winter > amount_lente and amount_winter > amount_zomer and amount_winter > amount_herfst:
      self.label = "winter"
    elif amount_lente > amount_winter and amount_lente > amount_zomer and amount_lente > amount_herfst:
      self.label = "lente"
    elif amount_herfst > amount_lente and amount_herfst > amount_zomer and amount_herfst > amount_winter:
      self.label = "herfst"
    elif amount_zomer > amount_lente and amount_zomer > amount_winter and amount_zomer > amount_herfst:
      self.label = "zomer"

FG = []
TG = []
TN = []
TX = []
SQ = []
DR = []
RH = []

for dataset in data:
    FG.append(dataset[0])
    TG.append(dataset[1])
    TN.append(dataset[2])
    TX.append(dataset[3])
    SQ.append(dataset[4])
    DR.append(dataset[5])
    RH.append(dataset[6])

FG_min = min(FG)
FG_max = max(FG)
TG_min = min(TG)
TG_max = max(TG)
TN_min = min(TN)
TN_max = max(TN)
TX_min = min(TX)
TX_max = max(TX)
SQ_min = min(SQ)
SQ_max = max(SQ)
DR_min = min(DR)
DR_max = max(DR)
RH_min = min(RH)
RH_max = max(RH)

for dataset in data:
  dataset[0] = scale(dataset[0], FG_min, FG_max)
  dataset[1] = scale(dataset[1], TG_min, TG_max)
  dataset[2] = scale(dataset[2], TN_min, TN_max)
  dataset[3] = scale(dataset[3], TX_min, TX_max)
  dataset[4] = scale(dataset[4], SQ_min, SQ_max)
  dataset[5] = scale(dataset[5], DR_min, DR_max)
  dataset[6] = scale(dataset[6], RH_min, RH_max)



def cluster_nodes(k):
  nodes = []
  for i in range(len(data)):
    nodes.append(node(i, data[i]))

  centroids = []
  random_points = []
  for i in range(k):
    random_index = randrange(len(data))
    while(random_index in random_points):
      random_index = randrange(len(data))
    random_points.append(random_index)
    centroids.append(centroid(nodes[random_points[i]].data))
    
  done_clustering = False

  while not done_clustering:
    new_clusters = []
    for i in range(k):
      new_clusters.append([])

    for node_ in nodes:
      distances = []
      for i in range(len(centroids)):
        distances.append([get_distance(node_, centroids[i]), i])
      new_clusters[min(distances)[1]].append(node_.index)

    old_clusters = []
    for centroid_ in centroids:
      old_clusters.append(centroid_.node_indexes)

    if new_clusters == old_clusters:
      done_clustering = True
    else:
      for i in range(len(centroids)):
        centroids[i].node_indexes = new_clusters[i]
        centroids[i].calculate_mean()
  
  average_intra_cluster_distance = 0

  for centroid_ in centroids:
    centroid_.calculate_label()
    average_intra_cluster_distance += centroid_.get_intra_cluster_distance()
  
  return (average_intra_cluster_distance / k)
  

intra_cluster_distances = []
k_values = []
for k in range(1, 13):
  intra_cluster_distances.append(cluster_nodes(k))
  k_values.append(k)

plt.plot(k_values, intra_cluster_distances)
plt.show()