#!/usr/bin/env python3
#encoding: utf-8

'''
This script is part of the program to simulate a navigation experiment where a robot
has to discover the rewarded state of the arena in which it evolves using a meta-control
decision algorithm : "Dromnelle, R., Renaudo, E., Chetouani, M., Maragos, P., Chatila, R.,
Girard, B., & Khamassi, M. (2022). Reducing Computational Cost During Robot Navigation and 
Human–Robot Interaction with a Human-Inspired Reinforcement Learning Architecture. 
International Journal of Social Robotics, 1-27."

This script contains some functions used by the modelFree.py, the 
modelBased.py and the metaController.py scripts. The names of the functions
are explicit : 
set_* = edit the contents of a dictionary
compute_* = do a computation on an element of a data structure and return the result
get_* = collect a value in a data structure
'''

__author__ = "Rémi Dromnelle"
__version__ = "1.0"
__maintainer__ = "Rémi Dromnelle"
__email__ = "remi.dromnelle@gmail.com"
__status__ = "Production"

# ---------------------------------------------------------------------------
# IMPORT
# ---------------------------------------------------------------------------
from optparse import OptionParser
import re
import json
import sys
import os
import datetime
import time
import random
import numpy as np
import collections as col
import math
import copy
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
INF = 1000000000000000000000000
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def softmax_actions_prob(qvalues, beta):
	print(f"Qvalues : {qvalues}")
	actions_prob = dict()
	sum_probs = 0
	# -----------------------------------------------------------------------
	for key, value in qvalues.items():
		# -------------------------------------------------------------------
		actions_prob[str(key)] = np.exp(value*beta)
		# -------------------------------------------------------------------
		sum_probs += actions_prob[str(key)]
	# -----------------------------------------------------------------------
	for key, value in qvalues.items():
		actions_prob[str(key)] = actions_prob[str(key)]/sum_probs
	# -----------------------------------------------------------------------
	print(f"Prob actions : {actions_prob}")
	return actions_prob
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def softmax_decision(actions_prob, actions):
	cum_actions_prob = list()
	previous_value = 0
	# -----------------------------------------------------------------------
	for key, value in actions_prob.items():
		cum_actions_prob.append([key,value + previous_value])
		#print(cum_actions_prob)
		previous_value = cum_actions_prob[-1][1]
	# -----------------------------------------------------------------------
	randval = np.random.rand()
	decision = dict()
	# -----------------------------------------------------------------------
	for key_value in cum_actions_prob:
		if randval < key_value[1]:
			decision = key_value[0]
			action = actions[key_value[0]]
			break
	return decision, action
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def set_visit(dict_qvalues, state):
	for dictStateValues in dict_qvalues["values"]:
		if dictStateValues["state"] == state:
			dictStateValues["visits"] += 1
			break

def get_visit(dict_qvalues, state):
	for dictStateValues in dict_qvalues["values"]:
		if dictStateValues["state"] == state:
			visit = dictStateValues["visits"]
			return visit
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def set_deltaQ(dict_qvalues, state, deltaQ):
	for dictStateValues in dict_qvalues["values"]:
		if dictStateValues["state"] == state:
			dictStateValues["deltaQ"] = deltaQ
			break

def get_deltaQ(dict_qvalues, state):
	for dictStateValues in dict_qvalues["values"]:
		if dictStateValues["state"] == state:
			return dictStateValues["deltaQ"]
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def set_RPE(dict_qvalues, state, RPE):
	for dictStateValues in dict_qvalues["values"]:
		if dictStateValues["state"] == state:
			dictStateValues["RPE"] = RPE
			break

def get_RPE(dict_qvalues, state):
	for dictStateValues in dict_qvalues["values"]:
		if dictStateValues["state"] == state:
			return dictStateValues["RPE"]
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def set_delta_prob(dict_delta_prob, state, delta_prob):
	for dictStateValues in dict_delta_prob["values"]:
		if dictStateValues["state"] == state:
			dictStateValues["delta_prob"] = delta_prob
			break

def get_delta_prob(dict_delta_prob, state):
	for dictStateValues in dict_delta_prob["values"]:
		if dictStateValues["state"] == state:
			return dictStateValues["delta_prob"]
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def set_actions_prob(dict_probs, state, list_probs):
	for dictStateValues in dict_probs["values"]:
		if dictStateValues["state"] == state:
			dictStateValues["actions_prob"] = list_probs
			break

def set_filtered_prob(dict_probs, state, list_probs):
	for dictStateValues in dict_probs["values"]:
		if dictStateValues["state"] == state:
			dictStateValues["filtered_prob"] = list_probs
			break

def get_filtered_prob(dict_probs, state):
	for dictStateValues in dict_probs["values"]:
		if dictStateValues["state"] == state:
			list_probs = dictStateValues["filtered_prob"]
			return list_probs
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def set_decided_action(dict_decision, state, list_actions):
	for dictStateValues in dict_decision["values"]:
		if dictStateValues["state"] == state:
			dictStateValues["history_decisions"] = list_actions
			break

def get_decided_action(dict_decision, state):
	for dictStateValues in dict_decision["values"]:
		if dictStateValues["state"] == state:
			list_actions = dictStateValues["history_decisions"]
			return list_actions

def set_history_decision(dict_decision, state, decided_action, window_size):
	for dictStateValues in dict_decision["values"]:
		if dictStateValues["state"] == state:
			for action in range(0,len(dictStateValues["history_decisions"])):
				if action == decided_action:
					value_to_add = 1
				else:
					value_to_add = 0
				pointer = window_size-1
				while pointer >= 0:
					if pointer == 0:
						dictStateValues["history_decisions"][action][pointer] = value_to_add
					else:
						dictStateValues["history_decisions"][action][pointer] = dictStateValues["history_decisions"][action][pointer-1]
					pointer = pointer - 1
			break
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def set_duration(dict_duration, state, duration):
	for dictStateValues in dict_duration["values"]:
		if dictStateValues["state"] == state:
			dictStateValues["duration"] = duration
			break

def get_duration(dict_duration, state):
	for dictStateValues in dict_duration["values"]:
		if dictStateValues["state"] == state:
			return dictStateValues["duration"]
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def set_qval(dict_qvalues, state, action, value):
	for dictStateValues in dict_qvalues["values"]:
		if dictStateValues["state"] == state:
			for dictActionValue in dictStateValues["values"]:
				if dictActionValue["action"] == action:
					dictActionValue["value"] = value
					break
			break

def get_qval(dict_qvalues, state, action):
	for dictStateValues in dict_qvalues["values"]:
		if dictStateValues["state"] == state:
			for dictActionValue in dictStateValues["values"]:
				if dictActionValue["action"] == action:
					qvalue = dictActionValue["value"]
					return qvalue

def get_qvals(dict_qvalues, state):
	qvalues = dict()
	for dictStateValues in dict_qvalues["values"]:
		if dictStateValues["state"] == state:
			for dictActionValue in dictStateValues["values"]:
				action = dictActionValue["action"]
				value = dictActionValue["value"]
				qvalues[action] = value
			return qvalues

def get_vval(dict_qvalues, state):
	for dictStateValues in dict_qvalues["values"]:
		if dictStateValues["state"] == state:
			vVal = 0.0
			for dictActionValue in dictStateValues["values"]:
				qvalue = dictActionValue["value"]
				if qvalue > vVal:
					vVal = qvalue
			return vVal
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def get_reward(dict_rewards, state, action):
	for transition in dict_rewards["transitionActions"]:
		if transition["state"] == state and transition["action"] == action:
			currentReward = transition["reward"]
			return currentReward

def set_reward(dict_rewards, state, action, reward):
	for transition in dict_rewards["transitionActions"]:
		if transition["state"] == state and transition["action"] == action:
			transition["reward"] = reward
			break

def initialize_rewards(dict_rewards, state, ACTIONSPACE):
	for a in range(0,ACTIONSPACE):
		dicStateActionReward = dict()
		dicStateActionReward["state"] = state
		dicStateActionReward["action"] = a
		dicStateActionReward["reward"] = 0.0
		dict_rewards["transitionActions"].append(dicStateActionReward)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def get_transition_prob(dict_transitions, start, action, arrival):
	for state in dict_transitions["transitionActions"]:
		if state["state"] == start:
			for transition in state["transitions"]:
				if transition["action"] == action and transition["state"] == arrival:
					prob = transition["prob"]
					return prob

def get_transition_probs(dict_transitions, start, action):
	dictProbs = dict()
	for state in dict_transitions["transitionActions"]:
		if state["state"] == start:
			for transition in state["transitions"]:
				if transition["action"] == action:# and sum(transition["window"]) != 0:
					state = transition["state"]
					prob = transition["prob"]
					dictProbs[state] = prob
			return dictProbs

def set_transition_prob(dict_transitions, start, action, arrival, prob):
	for state in dict_transitions["transitionActions"]:
			if state["state"] == start:
				for transition in state["transitions"]:
					if transition["action"] == action and transition["state"] == arrival:
						transition["prob"] = prob
						break
				break
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def initialize_transition(dict_transitions, start, action, arrival, prob, window_size):
	dicStateTransitions = dict()
	dicStateTransitions["state"] = start
	dicStateTransitions["transitions"] = list()
	# -----------------------------------------------------------------------
	dictActionStateProb = dict()
	dictActionStateProb["action"] = action
	dictActionStateProb["state"] = arrival
	dictActionStateProb["prob"] = prob
	dictActionStateProb["window"] = [1]+(window_size-1)*[0]
	# -----------------------------------------------------------------------
	dicStateTransitions["transitions"].append(dictActionStateProb)
	dict_transitions["transitionActions"].append(dicStateTransitions)

def add_transition(dict_transitions, start, action, arrival, prob, window_size):
	for dicStateTransitions in dict_transitions["transitionActions"]:
		if dicStateTransitions["state"] == start:
			dictActionStateProb = dict()
			dictActionStateProb["action"] = action
			dictActionStateProb["state"] = arrival
			dictActionStateProb["prob"] = prob
			dictActionStateProb["window"] = [1]+(window_size-1)*[0]
			dicStateTransitions["transitions"].append(dictActionStateProb)
			break

def get_number_transitions(dict_transitions, start, action):
	dictTransitions = dict()
	for state in dict_transitions["transitionActions"]:
		if state["state"] == start:
			for transition in state["transitions"]:
				if transition["action"] == action:
					state = transition["state"]
					number_transitions = sum(transition["window"])
					dictTransitions[state] = number_transitions
			return dictTransitions

def set_transitions_window(dict_transitions, start, action, arrival, window_size):
	for state in dict_transitions["transitionActions"]:
		if state["state"] == start:
			for transition in state["transitions"]:
				if transition["action"] == action:
					pointer = window_size - 1
					while pointer >= 0:
						if pointer == 0:
							if transition["state"] == arrival:
								transition["window"][pointer] = 1
							else:
								transition["window"][pointer] = 0
						else:
							transition["window"][pointer] = transition["window"][pointer-1]
						pointer  = pointer - 1
			break
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def low_pass_filter(alpha, old_value, new_value):
	"""
	Apply a low-pass filter
	"""
	# ---------------------------------------------------------------------------
	filtered_value = (1.0 - alpha) * old_value + alpha * new_value
	return filtered_value
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def shanon_entropy(source_vector):
	entropy = 0
	for element in source_vector:
		if element == 0:
			element = 0.0000001
		entropy += element*np.log2(element)
	return -(entropy)
# ---------------------------------------------------------------------------












