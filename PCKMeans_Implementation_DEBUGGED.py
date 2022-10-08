#!/usr/bin/env python
"""
=== DOCUMENTATION ===
Step1: Instantiate class with k=const.
Step2: Fit training data: self.fit_training_data(X), input X as numpy array.
Step3: Fit constraint matrix (defined in excel as seen in template): self.read_constraint_matrix(excel_path)
Step4: Define weight w to instantiate weight matrix: self.weight_matrix(w)
Step5: Run self.fit() to train model, returning X and y (joinable dataframes).
	   Also possible to access self.general_loss_per_epoch, self.cluster_list_per_epoch.
"""


# === IMPORTS ===
import random
import math
import numpy as np
import pandas as pd


# === CLASS DEFINITION ===
class PCK():

	def __init__(self, k, distance_measure="Euclidian Distance"):
		"""
		Instantiates the PCK-Means algorithm.
		Please specify k and the distance_measure (set to Euclidian Distance by default).
		"""
		self.k = k
		self.distance_measure = distance_measure
		self.con = None
		self.w = None
		self.cluster_list_per_epoch = None
		self.X = None
		self.feature_list = None
		self.learned_labels = {}


	def fit_training_data(self, X, feature_list):
		"""
		Reads training data X and assigns datapoints indices for handling constraints.
		"""
		X_keys = range(X.shape[0]) # Initialize keys with length of dataset X
		X_dict = dict(zip(X_keys, X))
		self.X = X_dict # Assign to instance
		self.feature_list = feature_list # List of feature names


	def read_constraint_matrix(self, excel_path):
		"""
		Reads the constraint matrix defined in an Excel-file.

		Format of Excel:
		Datapoint1,Datapoint2,Constraints
		0,1,-1
		0,2,1
		...
		(with -1 as CannotLink and 1 as MustLink)
		"""
		self.con = np.full((len(self.X.keys()), len(self.X.keys())), 0, dtype=float)
		excel = pd.read_excel(excel_path)
		for i in list(excel.index): # Place constraints in matrix
			self.con[excel.loc[i, "Datapoint1"]][excel.loc[i, "Datapoint2"]]= excel.loc[i, "Constraints"]
			self.con[excel.loc[i, "Datapoint2"]][excel.loc[i, "Datapoint1"]]= excel.loc[i, "Constraints"]


	def weight_matrix(self, weight):
		"""
		Creates a weight-matrix filled with cell values weight
		and with same shape as constraint_matrix for element-wise multiplication.
		The variable weight regulates the constraint enforcing factor.
		"""
		self.w = np.full((self.con.shape[0], self.con.shape[1]), weight, dtype=float)
		for i in range(self.w.shape[0]):
			self.w[i][i] = np.nan # Cell where instance is regarded with itself set to None


	def compute_euclidian_distance(self, datapoint, cluster_center):
		"""
		Computes the euclidian distance between to datapoints.
		"""
		"""
		# For 1D-data (development phase)
		delta=math.sqrt((datapoint2-datapoint1)**2)
		return delta
		"""
		# For nD-data
		delta=self.X[datapoint]-cluster_center
		print(f"datapoint: {datapoint}")
		print(delta, self.X[datapoint], cluster_center)
		print(delta.shape, self.X[datapoint].shape, cluster_center.shape)
		return np.linalg.norm(delta, 2)


	def compute_penalties(self, datapoint, cluster):
		"""
		Computes penalties for violating must-link and cannot-links when assigned datapoint to cluster.
		"""
		ml_penalties = 0
		cl_penalties = 0

		for j in range(self.con.shape[0]): # Iterate over columns of constraint matrix
			if self.con[datapoint][j]==1: # Check for must-link-constraints
				if j not in cluster[1]["Datapoints"]: # If Constraint is violated
					temp_ml_pen = self.con[datapoint][j]*self.w[datapoint][j] # Add cost for must-link-constraint violation
					ml_penalties += temp_ml_pen

			elif self.con[datapoint][j]==-1: # Check for cannot-link-constraints
				if j in cluster[1]["Datapoints"]: # If constraint is violated
					temp_cl_pen = self.con[datapoint][j]*self.w[datapoint][j]*(-1) # Add cost for cannot-link-constraint violation
					cl_penalties += temp_cl_pen

			else: # Else do not add penalization cost
				pass

		return ml_penalties+cl_penalties # return the cost of both together


	def compute_losses_for_assignment(self, cluster_list, datapoint):
		"""
		Computes loss for a datapoint for every possible cluster-assignment.
		"""
		not_append_tracker = False # Tracker to check if we'd go beneath k clusters with removals
		datapoint_list = [] # List to track datapoints as presented (in case sequence differs)
		loss_list = [] # List to track losses for each possible cluster assignment

		# Compute losses
		for cluster in cluster_list:
			for cluster_datapoints in cluster[1]["Datapoints"]:
				datapoint_list.append(cluster_datapoints) # Keep track of order of datapoints
				loss = (1/2)*self.compute_euclidian_distance(datapoint, cluster[0]["Center"])+self.compute_penalties(datapoint, cluster)
				loss_list.append(loss)

		# Remove datapoint from old cluster (if necessary), so it is not there twice
		if not datapoint_list[np.argmin(loss_list)]==datapoint: # unless its the same point
			for cluster in cluster_list:
				for cluster_datapoints in cluster[1]["Datapoints"]:
					if cluster_datapoints == datapoint and not len(cluster[1]["Datapoints"])<=1:
						cluster[1]["Datapoints"].remove(datapoint) # remove datapoint from cluster
					elif cluster_datapoints == datapoint and len(cluster[1]["Datapoints"])<=1: # unless it is the only remaining datapoint in cluster
						not_append_tracker = True
		else:
			return np.argmin(loss_list), cluster_list # if loss for datapoint is already min in its cluster, return current loss and cluster_list

		if not not_append_tracker==True:
			# Append datapoint to cluster with lowest loss
			cluster_update_point = datapoint_list[np.argmin(loss_list)] # Get datapoint where loss is minimum
			cluster_counter=0
			for cluster in cluster_list: # Figure out in which cluster datapoint is
				if cluster_update_point in cluster[1]["Datapoints"]:
					break
				else:
					cluster_counter+=1
			cluster_list[cluster_counter][1]["Datapoints"].append(datapoint) # Update cluster
			temp_list = []
			for sample in cluster_list[cluster_counter][1]["Datapoints"]: # Get feature vector of samples
				temp_list.append(self.X[sample])
			cluster_list[cluster_counter][0]["Center"] = np.mean(temp_list, axis=0) # Update cluster center
			return np.argmin(loss_list), cluster_list

		else:
			cluster_counter=0
			for cluster in cluster_list: # Figure out in which cluster datapoint is
				if datapoint in cluster[1]["Datapoints"]:
					break
				else:
					cluster_counter+=1
			loss = self.compute_penalties(datapoint, cluster_list[cluster_counter])
			return loss, cluster_list




	def iterate_through_dataset(self, cluster_list):
		"""
		Iterates through dataset until convergence.
		"""
		convergence_statement = False
		general_loss_per_epoch = []
		cluster_list_per_epoch = []

		while convergence_statement == False:
			general_loss = 0
			# Iterate through dataset
			for datapoint in range(self.con.shape[0]):
				datapoint_loss, cluster_list = self.compute_losses_for_assignment(cluster_list, datapoint)
				general_loss += datapoint_loss
			general_loss_per_epoch.append(general_loss) # Keep track of loss
			cluster_list_per_epoch.append(cluster_list)
			# Check convergence statement
			if len(general_loss_per_epoch)>=2:
				if general_loss_per_epoch[-1]>=general_loss_per_epoch[-2]:
					convergence_statement == True
					print("Model converged.")
					return general_loss_per_epoch, cluster_list_per_epoch


	def initialize_cluster_centers(self):
		"""
		Initializes cluster centers according to neighborhoods.
		"""

		# === Step1: Find all neighborhoods. ===
		neighborhood_collection = [] # Collection of neighborhoods

		# Iterate through constraint_matrix
		for i in range(self.con.shape[0]): # Iterate through rows of constraint_matrix
			neighborhood = [] # Temporary variable for neighborhood

			for j in range(self.con.shape[1]): # Iterate through columns of constraint_matrix
				if self.con[i][j] == -1: # If constraint-relationship is cannot-link, continue
					continue

				elif self.con[i][j] == 1: # If constraint-relationship is must-link, assign them to same neighborhood

					# Except there is a cannot-link to a datapoint already assigned in neighborhood
					cannot_link_tracker=False # Initialize a tracker for cannot-links

					for datapoint in neighborhood:
						if self.con[j][datapoint]==-1: # Cannot-link to a datapoint already assigned
							cannot_link_tracker=True # Save information

					if cannot_link_tracker==True: # Skip if cannot-link exists
						continue

					else: # If not, assign them to same neighborhood
						neighborhood.append(i)
						neighborhood.append(j)
						neighborhood = list(set(neighborhood)) # filter duplicates

				else: # If no constraint-relationship to a datapoint
					continue

			# Append neighborhood in collection
			# Check whether an already assigned datapoint is included in new created neighborhood
			sample_assigned_tracker=False # Track with this variable
			for c_neighborhood in neighborhood_collection:
				for sample in c_neighborhood:
					if sample in neighborhood: # if so, continue with next column
						sample_assigned_tracker=True
			# Else, append new created neighborhood to collection
			if not (neighborhood==[] or sample_assigned_tracker==True): # but only if neighborhood is not still in initial form
				neighborhood_collection.append(list(neighborhood))
			else:
				continue

		# === Step2: Create cluster centers. ===
		if self.k > len(neighborhood_collection):
			# List all datapoints that are assigned to a neighborhood
			assigned_datapoints = []
			for neighborhood in neighborhood_collection:
				for dp in neighborhood:
					assigned_datapoints.append(dp)

			# Find out which datapoints are not assigned yet
			not_assigned_datapoints = [sample for sample in range(self.con.shape[0]) if sample not in assigned_datapoints]

			# Randomly initialize some of these as the missing cluster centers
			random_clusters = [[cluster] for cluster in random.sample(not_assigned_datapoints, (self.k-len(neighborhood_collection)))]
			clusters = neighborhood_collection + random_clusters

		elif self.k < len(neighborhood_collection):
			neighborhood_collection_desc = sorted(neighborhood_collection, reverse=True) # Sorts neighborhood collection decreasingly according to members
			clusters = neighborhood_collection_desc[:self.k] # Take k biggest neighborhoods as cluster centers

		else:
			clusters = neighborhood_collection # Take neighborhoods as cluster centers

		# === Step3: Compute the cluster-centers and return cluster information ===
		cluster_list = []
		for cluster in clusters:
			# Get feature vectors of samples
			dp_in_cluster = []
			for datapoint in cluster:
				dp_in_cluster.append(self.X[datapoint])
			temp_cluster_dict = [{"Center": np.mean(np.array(dp_in_cluster), axis=0)}, # calculate mean w.r.T. feature vectors
								 {"Datapoints": cluster}]
			cluster_list.append(temp_cluster_dict)

		return cluster_list


	def get_inertia(self):
		"""
		Returns the (summed) inertia (=Euclidian Distance) from all datapoints to its cluster centers.
		"""
		inertia = 0
		for cluster in self.cluster_list_per_epoch[-1]: # Take cluster assignments with minimum loss and iterate through them
			# print(f'Debugging Output: Datapoints: {cluster[1]["Datapoints"]}, Cluster center: {cluster[0]["Center"]}')
			for datapoint in cluster[1]["Datapoints"]:
				inertia += self.compute_euclidian_distance(datapoint, cluster[0]["Center"])
				# print(f'Datapoint: {datapoint}, ED to center: {self.compute_euclidian_distance(datapoint, cluster[0]["Center"])}')
				# print(f'Inertia: {inertia}')
		self.inertia_ = inertia


	def fit(self):
		"""
		Trains the PCK-model based on training data X, the constraint matrix and weight matrix.
		"""

		# Step1: Initialize cluster centers according to neighborhoods.
		cluster_list = self.initialize_cluster_centers()

		# Step2: Iterate through data until convergence.
		general_loss_per_epoch, cluster_list_per_epoch = self.iterate_through_dataset(cluster_list)

		# Step3: Save cluster-centers.
		self.general_loss_per_epoch = general_loss_per_epoch
		self.cluster_list_per_epoch = cluster_list_per_epoch

		# Step4: Save labels for datapoints in learned data.
		cluster_count = 1
		for cluster in cluster_list_per_epoch[-1]: # Take cluster assignments with minimum loss and iterate through them
			for datapoint in cluster[1]["Datapoints"]: # Assign clusters numbers
				self.learned_labels[datapoint] = cluster_count
			cluster_count += 1

		# Compute Inertia
		self.get_inertia()

		y = pd.DataFrame(self.learned_labels.values(), index=self.learned_labels.keys(), columns=["Class"])
		X = pd.DataFrame(self.X.values(), index=self.X.keys(), columns=self.feature_list)
		return X,y

