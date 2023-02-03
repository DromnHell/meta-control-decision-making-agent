#!/usr/bin/env python3
#encoding: utf-8

'''
This script is part of the program to simulate a navigation experiment where a robot
has to discover the rewarded state of the arena in which it evolves using a meta-control
decision algorithm : "Dromnelle, R., Renaudo, E., Chetouani, M., Maragos, P., Chatila, R.,
Girard, B., & Khamassi, M. (2022). Reducing Computational Cost During Robot Navigation and 
Human–Robot Interaction with a Human-Inspired Reinforcement Learning Architecture. 
International Journal of Social Robotics, 1-27."

With this script, the simulated agent use a Value-Iteration algorithm (model-based learning
by planification) to learn to solve the task.
'''

__author__ = "Rémi Dromnelle"
__version__ = "1.0"
__maintainer__ = "Rémi Dromnelle"
__email__ = "remi.dromnelle@gmail.com"
__status__ = "Production"

from utility import *

VERSION = 1


class ModelBased:
	"""
	This class implements a model-based learning algorithm (value-iteration).
    """

	def __init__(self, experiment, map_file, initial_variables, action_space, boundaries_exp, parameters, options_log):
		"""
		Iinitialise values and models
		"""
		# -----------------------------------------------------------------------------
		# Initialise all the variables which will be used
		self.experiment = experiment
		self.max_reward = boundaries_exp["max_reward"]
		self.duration = boundaries_exp["duration"]
		self.window_size = boundaries_exp["window_size"]
		self.epsilon = boundaries_exp["epsilon"]
		self.alpha = parameters["alpha"]
		self.gamma = parameters["gamma"]
		self.beta = parameters["beta"]
		self.log = options_log["log"]
		self.summary = options_log["summary"]
		action_count = initial_variables["action_count"]
		decided_action = initial_variables["decided_action"]
		previous_state = initial_variables["previous_state"]
		current_state = initial_variables["current_state"]
		self.init_qvalue = initial_variables["qvalue"]
		init_reward = initial_variables["reward"]
		init_delta = initial_variables["delta"]
		init_plan_time = initial_variables["plan_time"]
		self.action_space = action_space
		init_actions_prob = initial_variables["actions_prob"]
		self.not_learn = False
		# ----------------------------------------------------------------------------
		# // List and dicts for store data //
		# Create the list of states
		self.list_states = list()
		# Create the list of known action
		self.list_actions = dict()
		# Create a dict that contains the qvalues of the expert
		self.dict_qvalues = dict()
		# Create the transitions dict according to the type of log used by Erwan
		self.dict_transitions = dict()
		self.dict_transitions["actioncount"] = action_count
		self.dict_transitions["transitionActions"] = list()
		# Create the rewards dict according to the type of log used by Erwan
		self.dict_rewards = dict()
		self.dict_rewards["actioncount"] = action_count
		self.dict_rewards["transitionActions"] = list()
		# Create a dict that contains the probability of actions for each states
		self.dict_actions_prob = dict()
		self.dict_actions_prob["actioncount"] = action_count
		self.dict_actions_prob["values"] = list()
		# Create the dict of neighbour reward states
		self.dict_goals = dict()
		self.dict_goals["values"] = list()
		# Create a dict that contains the delta prob for each state
		self.dict_delta_prob = dict()
		self.dict_delta_prob["actioncount"] = action_count
		self.dict_delta_prob["values"] = list()
		# Create a dict that contains the probability of actions for each states
		self.dict_decision = dict()
		self.dict_decision["actioncount"] = action_count
		self.dict_decision["values"] = list()
		# Create a dict that contains the time of planification for each states
		self.dict_duration = dict()
		self.dict_duration["actioncount"] = action_count
		self.dict_duration["values"] = list()
		# -----------------------------------------------------------------------------
		# Load the transition model which will be used as map
		with open(map_file,'r') as file2:
			self.map = json.load(file2)
		# For each state of the map : 
		for state in self.map["transitionActions"]:
			s = str(state["state"])
			t = state["transitions"]
			# -----------------------------------------------------------------------
			self.dict_qvalues[(s,"qvals")] = [self.init_qvalue]*self.action_space
			self.dict_qvalues[(s,"visits")] = 0
			# -----------------------------------------------------------------------
			# - initialise the probabilties of actions
			self.dict_actions_prob["values"].append({"state": s, "actions_prob": [init_actions_prob]*self.action_space, "filtered_prob": [init_actions_prob]*self.action_space})
			# -------------------------------------------------------------------------
			# - initialise the "identity of the selected action" dict
			self.dict_decision["values"].append({"state": s, "history_decisions": [[0]*self.window_size]*self.action_space})
			# -----------------------------------------------------------------------
			# - initialise the delta prob dict
			self.dict_delta_prob["values"].append({"state": s, "delta_prob": init_delta})
			# -------------------------------------------------------------------------
			# - initialise the duration dict
			self.dict_duration["values"].append({"state": s, "duration": 0.0})
		# -----------------------------------------------------------------------------


	def get_actions_prob(self, current_state):
		"""
		Get the probabilities of actions of the current state
		"""
		# ----------------------------------------------------------------------------
		return get_filtered_prob(self.dict_actions_prob, current_state)
		# ----------------------------------------------------------------------------

	def get_plan_time(self, current_state):
		"""
		Get the time of planification for the current state
		"""
		# ----------------------------------------------------------------------------
		return get_duration(self.dict_duration, current_state)
		# ----------------------------------------------------------------------------


	def decide(self, current_state, qvalues):
		"""
		Choose the next action using soft-max policy
		"""
		# ----------------------------------------------------------------------------
		actions = dict()
		qvals = dict()
		# ----------------------------------------------------------------------------
		for a in range(0,8):
			actions[str(a)] = a
			qvals[str(a)] = qvalues[a] #repassage en mode dico pour compatibilité avec les fonctions de Rémi
		# ----------------------------------------------------------------------------
		# Soft-max function
		actions_prob = softmax_actions_prob(qvals, self.beta)
		new_probs = list()
		for action, prob in actions_prob.items():
			new_probs.append(prob)
		set_actions_prob(self.dict_actions_prob, current_state, new_probs)
		# -------------------------------------------------------------------------
		# For each action, sum the probabilitie of selection with a low pass filter
		old_probs = get_filtered_prob(self.dict_actions_prob, current_state)
		filtered_actions_prob = list()
		for a in range(0,len(new_probs)):
			filtered_actions_prob.append(low_pass_filter(self.alpha, old_probs[a], new_probs[a]))
		set_filtered_prob(self.dict_actions_prob, current_state, filtered_actions_prob)
		# ----------------------------------------------------------------------------
		# The end of the soft-max function
		decision, choosen_action = softmax_decision(actions_prob, actions)
		# ---------------------------------------------------------------------------
		return choosen_action, actions_prob
		# ----------------------------------------------------------------------------


	def update_qvalues(self, this_state):
		"""
		Value iteration algorithm
		"""
		# -----------------------------------------------------------------------------
		sum_deltaQ = 0.0
		# -----------------------------------------------------------------------------
		for i in self.list_actions[this_state]:
			action = i 
			previous_qvalue = self.dict_qvalues[(str(this_state),"qvals")][action] 
			flag = False
			accu = 0.0
			reward = get_reward(self.dict_rewards, this_state, action)
			# ----------------------------------------------------------------
			# loop througth the transitions
			for state in self.dict_transitions["transitionActions"]:
				if str(state["state"]) == this_state:
					for transition in state["transitions"]:
						link = transition["action"]
						prob = transition["prob"]

						linked_state = str(transition["state"])
						vValue = max(self.dict_qvalues[(str(linked_state),"qvals")])
						if link == action:
							flag = True
							accu += (prob * (1.9*reward + self.gamma * vValue))
					break
			# ----------------------------------------------------------------
			if flag == True:
				self.dict_qvalues[(str(this_state),"qvals")][int(action)] = accu
			# ----------------------------------------------------------------
			new_deltaQ = abs(self.dict_qvalues[(str(this_state),"qvals")][int(action)] - previous_qvalue)
			sum_deltaQ += new_deltaQ
		# ----------------------------------------------------------------------------
		return sum_deltaQ
		# ----------------------------------------------------------------------------


	def infer(self, current_state):
		"""
		In the MB expert, the process of inference consists to do planification using
		models of the world.
		"""
		# ----------------------------------------------------------------------------
		cycle = 0
		while True:
			# ------------------------------------------------------------------------
			#print(f"Cycle of VI : {cycle}")
			convergence_indicator = 0.0
			# ------------------------------------------------------------------------
			for state in self.list_states:
				# Use low pass filter on the sum of deltaQ
				new_sum_deltaQ = self.update_qvalues(state)
				# --------------------------------------------------------------------
				convergence_indicator += new_sum_deltaQ
			# ------------------------------------------------------------------------
			#print(f"Convergence indicator : {convergence_indicator}")
			cycle += 1
			# ------------------------------------------------------------------------
			# Stop VI when convergence
			if convergence_indicator < self.epsilon:
				break
		# ----------------------------------------------------------------------------
		return self.dict_qvalues[(str(current_state),"qvals")]
		# ----------------------------------------------------------------------------


	def update_reward(self, current_state, reward_obtained):
		"""
		Update the the model of reward
		"""
		# ----------------------------------------------------------------------------
		#expected_reward = reward_obtained
		#prob = get_transition_prob(self.dict_transitions, previous_state, action, current_state)
		# ----------------------------------------------------------------------------
		for state in self.dict_goals["values"]:
			# ------------------------------------------------------------------------
			# Change potentially the reward of the rewarded state
			if state["state"] == current_state:
				state["reward"] = reward_obtained
			# ------------------------------------------------------------------------
			for link in state["links"]:
				action = link[0]
				previous_state = link[1]
				prob = get_transition_prob(self.dict_transitions, previous_state, action, state["state"])
				relative_reward = prob  * state["reward"]
				set_reward(self.dict_rewards, previous_state, action, relative_reward)
				# --------------------------------------------------------------------


	def update_prob(self, previous_state, action, current_state):
		"""
		Update the the model of transition
		"""
		# ----------------------------------------------------------------------------
		delta_prob = 0.0 
		nb_transitions = get_number_transitions(self.dict_transitions, previous_state, action)
		sum_nb_transitions = sum(nb_transitions.values())
		probs = get_transition_probs(self.dict_transitions, previous_state, action)
		# ----------------------------------------------------------------------------
		for arrival, old_prob in probs.items():
			new_prob = nb_transitions[arrival]/sum_nb_transitions
			set_transition_prob(self.dict_transitions, previous_state, action, arrival, new_prob)
			delta_prob += abs(new_prob - old_prob)
		probs = get_transition_probs(self.dict_transitions, previous_state, action)
		# ----------------------------------------------------------------------------


	def learn(self, previous_state, action, current_state, reward_obtained):
		"""
		Update the contents of the rewards and the transitions model
		"""
		# ----------------------------------------------------------------------------
		# // Update the transition model //
		self.update_prob(previous_state, action, current_state)
		# ----------------------------------------------------------------------------
		# // Update the reward model //
		self.update_reward(current_state, reward_obtained)
		# ----------------------------------------------------------------------------


	def update_data_structure(self, action_count, previous_state, action, current_state, reward_obtained):
		"""
		Update the data structure of the rewards and the transitions models
		"""
		# ----------------------------------------------------------------------------
		self.dict_qvalues[(str(current_state),"actioncount")] = action_count
		self.dict_transitions["actioncount"] = action_count
		self.dict_rewards["actioncount"] = action_count
		# ----------------------------------------------------------------------------
		# For the modelof qvalues, update only the numer of visit
		self.dict_qvalues[(str(previous_state),"visits")] += 1 #CHECK IF GOOD
		# ----------------------------------------------------------------------------
		# If the previous state is unknown, add it in the states model, in the reward model and in the transition model
		# (not needful for the model of q-values because it has already is final size)
		if previous_state not in self.list_states:
			# do not do for rewarded state
			if self.not_learn == False:
				self.list_states.append(previous_state)
				self.list_actions[previous_state] = [action]
				initialize_rewards(self.dict_rewards, previous_state, self.action_space)
				initialize_transition(self.dict_transitions, previous_state, action, current_state, 1, self.window_size)
		else:
			# do not do for rewarded state
			if self.not_learn == False:
				if action not in self.list_actions[previous_state]:
					self.list_actions[previous_state].append(action)
					add_transition(self.dict_transitions, previous_state, action, current_state, 1, self.window_size)
				else:
					# Check if the transition "previous state -> action -> current state" has already been experimented
					transition = False
					for dicStateTransitions in self.dict_transitions["transitionActions"]:
						if dicStateTransitions["state"] == previous_state:
							for dictActionStateProb in dicStateTransitions["transitions"]:
								if dictActionStateProb["action"] == action and dictActionStateProb["state"] == current_state:
									transition = True
									break
							break
					# If the transition doesn't exist, add it
					if transition == False:
						add_transition(self.dict_transitions, previous_state, action, current_state, 0, self.window_size)
					# If it exist, update the window of transitions
					else:
						set_transitions_window(self.dict_transitions, previous_state, action, current_state, self.window_size)
		# # Else delete from the list of state
		# else:
		# 	while previous_state in self.list_states:
		# 		del self.list_states[self.list_states.index(previous_state)]
		# ----------------------------------------------------------------------------
		# Check if the agent already known goals and neighbours
		if reward_obtained != 0.0:
			# ------------------------------------------------------------------------
			# If none goal is known, add it in the dict with this neighbour
			if not self.dict_goals["values"]:
				self.dict_goals["values"].append({"state": current_state, "reward": reward_obtained, "links": [(action, previous_state)]})
			# ------------------------------------------------------------------------
			# Check if this goal is already known
			known_goal = False
			for state in self.dict_goals["values"]:
				if state["state"] == current_state:
					known_goal = True
					known_link = False
					for link in state["links"]:
						if link[0] == action and link[1] == previous_state:
							known_link = True
							break
					# ----------------------------------------------------------------
					if known_link == False:
						state["links"].append((action, previous_state))
					break
			# ------------------------------------------------------------------------
			if known_goal == False:
				self.dict_goals["values"].append({"state": current_state, "reward": reward_obtained, "links": [(action, previous_state)]})
				# delete transitions and possible actions for current_state
				for transitions in self.dict_transitions["transitionActions"]:
					if transitions["state"] == current_state:
						transitions["transitions"] = []
				for i in range (8):
					self.list_actions[current_state] = []
		# ----------------------------------------------------------------------------


	def run(self, action_count, cumulated_reward, reward_obtained, previous_state, decided_action, current_state, do_we_plan): 
		"""
		Run the model-based system
		"""
		# ----------------------------------------------------------------------------
		print("------------------------ MB --------------------------------")
		# ----------------------------------------------------------------------------
		# Update the actioncount and the number of the visits for the previous state
		self.dict_duration["actioncount"] = action_count
		self.dict_delta_prob["actioncount"] = action_count
		self.dict_qvalues[(str(current_state),"actioncount")] = action_count
		self.dict_qvalues[(str(previous_state),"visits")] += 1
		# ----------------------------------------------------------------------------
		# Update the data structure of the models (states, rewards, transitions, qvalues)
		self.update_data_structure(action_count, previous_state, decided_action, current_state, reward_obtained)
		# ----------------------------------------------------------------------------
		if self.not_learn == False:
		# Update the transition model and the reward model according to the learning.
			self.learn(previous_state, decided_action, current_state, reward_obtained)
		# ----------------------------------------------------------------------------
		# If the expert was choosen to plan, update all the q-values using planification
		if do_we_plan:
			# ------------------------------------------------------------------------
			old_time = datetime.datetime.now()
			# ------------------------------------------------------------------------
			# Run the planification process
			qvalues = self.infer(current_state)
			# ------------------------------------------------------------------------
			# Sum the duration of planification with a low pass filter
			current_time = datetime.datetime.now()
			new_plan_time = (current_time - old_time).total_seconds()
			old_plan_time = get_duration(self.dict_duration, current_state)
			filtered_time = low_pass_filter(self.alpha, old_plan_time, new_plan_time)
			set_duration(self.dict_duration, current_state, filtered_time)
		else:
			qvalues = self.dict_qvalues[(str(current_state),"qvals")]
		# ----------------------------------------------------------------------------
		# Choose the next action to do from the current state using soft-max policy.
		decided_action, actions_prob = self.decide(current_state, qvalues)
		# -------------------------------------------------------------------------
		# Maj the history of the decisions
		set_history_decision(self.dict_decision, current_state, decided_action, self.window_size)
		prefered_action = [0]*self.action_space
		for action in range(0,len(prefered_action)):
			for dictStateValues in self.dict_decision["values"]:
				if dictStateValues["state"] == current_state:
					prefered_action[action] = sum(dictStateValues["history_decisions"][action])
		# ---------------------------------------------------------------------------
		plan_time = get_duration(self.dict_duration, current_state)
		selection_prob = get_filtered_prob(self.dict_actions_prob, current_state)
		# ----------------------------------------------------------------------------
		if reward_obtained > 0.0:
			self.not_learn = True
			for a in range(0,8):
				self.dict_qvalues[(current_state,"qvals")] = [0.0]*self.action_space
		else:
			self.not_learn = False
		# ----------------------------------------------------------------------------
		if (action_count == self.duration) or (cumulated_reward == self.max_reward):
			# ------------------------------------------------------------------------
			# Build the summary file 
			if self.summary == True:
				if self.directory_flag == True:
					os.chdir("../")
				# --------------------------------------------------------------------
				prefixe = 'v%d_TBMB_'%(VERSION)
				self.summary_log = open(prefixe+'summary_log.dat', 'a')
				self.summary_log.write(f"{self.gamma} {self.beta} {cumulated_reward}\n")
		# ---------------------------------------------------------------------------
		# print("Qvalues : ")
		# for action in range(0,8):
		# 	print(self.dict_qvalues[str(current_state),"qvals"][int(action)])
		# ----------------------------------------------------------------------------
		return decided_action
		# ----------------------------------------------------------------------------



