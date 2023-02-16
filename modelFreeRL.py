#!/usr/bin/env python3
#encoding: utf-8

'''
This script is part of the program to simulate a navigation experiment where a robot
has to discover the rewarded state of the arena in which it evolves using a meta-control
decision algorithm : "Dromnelle, R., Renaudo, E., Chetouani, M., Maragos, P., Chatila, R.,
Girard, B., & Khamassi, M. (2022). Reducing Computational Cost During Robot Navigation and 
Human–Robot Interaction with a Human-Inspired Reinforcement Learning Architecture. 
International Journal of Social Robotics, 1-27."

With this script, the simulated agent use a Q-learning algorithm (model-free reinforcement learning) 
to learn to solve the task.
'''

__author__ = "Rémi Dromnelle"
__version__ = "2.0"
__maintainer__ = "Rémi Dromnelle"
__email__ = "remi.dromnelle@gmail.com"
__status__ = "Production"

from utility import *

VERSION = 2

class ModelFree:
	"""
	This class implements a tabular model-free reinforcement learning algorithm (Q-learning).
    """

	def __init__(self, expert, experiment, env_file, initial_variables, action_space, boundaries_exp, parameters, log):
		"""
		Initialize values and models.
		"""
		# ---------------------------------------------------------------------------
		# Initialize all the variables which will be used
		self.ID = expert
		self.experiment = experiment
		action_count = initial_variables["action_count"]
		self.init_qvalue = initial_variables["qvalue"]
		init_actions_prob = initial_variables["actions_prob"]
		self.action_space = action_space
		self.max_reward = boundaries_exp["max_reward"]
		self.duration = boundaries_exp["duration"]
		self.window_size = boundaries_exp["window_size"]
		self.alpha = parameters["alpha"]
		self.gamma = parameters["gamma"]
		self.beta = parameters["beta"]
		self.log = log["log"]
		self.summary = log["summary"]
		self.not_learn = False
		self.wait_new_goal = True
		self.rewarded_state = None
		# ---------------------------------------------------------------------------
		# // List and dicts to store data //
		# Create a dict that contains the qvalues of the expert
		self.dict_qvalues = dict()
		# Create a dict that contains the probability of actions for each states
		self.dict_actions_prob = dict()
		self.dict_actions_prob["actioncount"] = action_count
		self.dict_actions_prob["values"] = list()
		# Create a dict that contains the probability of actions for each states
		self.dict_decision = dict()
		self.dict_decision["actioncount"] = action_count
		self.dict_decision["values"] = list()
		# Create a dict that contains the time of planification for each states
		self.dict_duration = dict()
		self.dict_duration["actioncount"] = action_count
		self.dict_duration["values"] = list()
		# ---------------------------------------------------------------------------
		# Load the transition model which will be used as the environment representation
		with open(env_file,'r') as file2:
			self.env = json.load(file2)
		# For each state of the environment : 
		for state in self.env["transitionActions"]:
			s = str(state["state"])
			t = state["transitions"]
			# -----------------------------------------------------------------------
			self.dict_qvalues[(s,"qvals")] = [self.init_qvalue]*self.action_space
			self.dict_qvalues[(s,"visits")] = 0
			# -----------------------------------------------------------------------
			# - initialize the "probabilties of actions" dict
			self.dict_actions_prob["values"].append({"state": s, "actions_prob": [init_actions_prob]*self.action_space, "filtered_prob": [init_actions_prob]*self.action_space})
			# -----------------------------------------------------------------------
			# - initialize the "identity of the selected action" dict
			self.dict_decision["values"].append({"state": s, "history_decisions": [[0]*self.window_size]*self.action_space})
			# -----------------------------------------------------------------------
			# - initialize the duration dict
			self.dict_duration["values"].append({"state": s, "duration": 0.0})
		# ---------------------------------------------------------------------------


	def get_actions_prob(self, current_state):
		"""
		Get the probabilities of actions of the current state.
		"""
		# ----------------------------------------------------------------------------
		return get_filtered_prob(self.dict_actions_prob, current_state)
		# ----------------------------------------------------------------------------

	def get_plan_time(self, current_state):
		"""
		Get the time of planification for the current state.
		"""
		# ----------------------------------------------------------------------------
		return get_duration(self.dict_duration, current_state)
		# ----------------------------------------------------------------------------


	def decide(self, current_state, qvalues):
		"""
		Choose the next action using a soft-max policy.
		"""
		# ----------------------------------------------------------------------------
		actions = dict()
		qvals = dict()
		# ----------------------------------------------------------------------------
		for a in range(0,self.action_space):
			actions[str(a)] = a
			qvals[str(a)] = qvalues[a]
		# ----------------------------------------------------------------------------
		# Soft-max function
		actions_prob = softmax_actions_prob(qvals, self.beta)
		new_probs = list()
		for prob in actions_prob.values():
			new_probs.append(prob)
		set_actions_prob(self.dict_actions_prob, current_state, new_probs)
		# -------------------------------------------------------------------------
		# For each action, sum the probabilities of selection with a low pass filter
		old_probs = get_filtered_prob(self.dict_actions_prob, current_state)
		filtered_actions_prob = list()
		for a in range(0,len(new_probs)):
			filtered_actions_prob.append(low_pass_filter(self.alpha, old_probs[a], new_probs[a]))
		set_filtered_prob(self.dict_actions_prob, current_state, filtered_actions_prob)
		# ----------------------------------------------------------------------------
		# The end of the soft-max function
		decision, choosen_action = softmax_decision(actions_prob, actions)
		# ---------------------------------------------------------------------------
		return choosen_action
		# ----------------------------------------------------------------------------

	def infer(self, current_state):
		"""
		In the MF expert, the process of inference consists to read the qvalues table.
		(this function is useless. It's only to be symetric with MB expert).
		"""
		# ----------------------------------------------------------------------------
		return self.dict_qvalues[(str(current_state),"qvals")]
		# ----------------------------------------------------------------------------


	def learn(self, previous_state, action, current_state, reward_obtained):
		"""
		Update qvalues using Q-learning.
		"""
		# ---------------------------------------------------------------------------
		qvalue_previous_state = self.dict_qvalues[str(previous_state),"qvals"][int(action)]
		qvalues_current_state = self.dict_qvalues[str(current_state),"qvals"]
		max_qvalues_current_state = max(qvalues_current_state)
		# ---------------------------------------------------------------------------
		# Compute the qvalue.
		# The bias of 1.9 was added to be able to compare the results of the simulated
		# navigation experiment (produced by this code) to the old real navigation experiment.
		# Indeed, the real robot had a bug, and this bias allows to "mimic" the bug effect
		# (corrected in this code), and thus to compare the results and reproduce those of
		# the paper. This biais can of course be remooved for new experiments !
		new_RPE = 1.9*reward_obtained + self.gamma * max_qvalues_current_state - qvalue_previous_state
		new_qvalue = qvalue_previous_state + self.alpha * new_RPE
		self.dict_qvalues[str(previous_state),"qvals"][int(action)] = new_qvalue
		# ---------------------------------------------------------------------------


	def run(self, action_count, cumulated_reward, reward_obtained, previous_state, decided_action, current_state, do_we_plan, new_goal): 
		"""
		Run the model-free RL expert.
		"""
		# ---------------------------------------------------------------------------
		print("------------------------ MF --------------------------------")
		# ---------------------------------------------------------------------------
		# Update the actioncount and the number of the visits for the previous state
		self.dict_duration["actioncount"] = action_count
		self.dict_actions_prob["actioncount"] = action_count
		self.dict_qvalues[(str(current_state),"actioncount")] = action_count
		self.dict_qvalues[(str(previous_state),"visits")] += 1
		# ---------------------------------------------------------------------------
		# The qvalues of the rewarded state have to be null. So set them to 0 the
		# first time the rewarded state is met
		if reward_obtained == 1 and cumulated_reward == 1:
			self.rewarded_state = current_state
			for a in range(0,self.action_space):
				self.dict_qvalues[(self.rewarded_state,"qvals")] = [0.0]*self.action_space
		# ---------------------------------------------------------------------------
		# Same with the new rewarded state after the environmental change
		if reward_obtained == 1 and new_goal == self.wait_new_goal == True:
			self.rewarded_state = current_state
			for a in range(0,self.action_space):
				self.dict_qvalues[(self.rewarded_state,"qvals")] = [0.0]*self.action_space
			self.wait_new_goal = False
		# ---------------------------------------------------------------------------
		# If the previous state is the rewarded state, the agent must not run its
		# learning process
		if previous_state == self.rewarded_state:
			self.not_learn = True
		else:
			self.not_learn = False
		# ---------------------------------------------------------------------------
		if self.not_learn == False:
			# Update the qvalues of the previous state using Q-learning
			self.learn(previous_state, decided_action, current_state, reward_obtained)
		# ---------------------------------------------------------------------------
		# If the expert was choosen to plan, compute the news probabilities of actions
		if do_we_plan:
			# -----------------------------------------------------------------------
			old_time = datetime.datetime.now()
			# -----------------------------------------------------------------------
			# Run the process of inference
			qvalues = self.infer(current_state)
			# -----------------------------------------------------------------------
			# Sum the duration of the planification with a low pass filter
			current_time = datetime.datetime.now()
			new_plan_time = (current_time - old_time).total_seconds()
			old_plan_time = get_duration(self.dict_duration, current_state)
			filtered_time = low_pass_filter(self.alpha, old_plan_time, new_plan_time)
			set_duration(self.dict_duration, current_state, filtered_time)
		else:
			qvalues = self.dict_qvalues[(str(current_state),"qvals")]
			# -----------------------------------------------------------------------
		decided_action = self.decide(current_state, qvalues)
		# -------------------------------------------------------------------------
		# Maj the history of the decisions
		set_history_decision(self.dict_decision, current_state, decided_action, self.window_size)
		prefered_action = [0]*self.action_space
		for action in range(0,len(prefered_action)):
			for dictStateValues in self.dict_decision["values"]:
				if dictStateValues["state"] == current_state:
					prefered_action[action] = sum(dictStateValues["history_decisions"][action])
		# ---------------------------------------------------------------------------
		if (action_count == self.duration) or (cumulated_reward == self.max_reward):
			# Build the summary file 
			if self.summary == True:
				if self.directory_flag == True:
					os.chdir("../")
				# -------------------------------------------------------------------
				prefixe = 'v%d_TBMF_'%(VERSION)
				self.summary_log = open(prefixe+'summary_log.dat', 'a')
				self.summary_log.write(f"{self.alpha} {self.gamma} {self.beta} {cumulated_reward}\n")
		# ----------------------------------------------------------------------------
		return decided_action
		# ---------------------------------------------------------------------------



