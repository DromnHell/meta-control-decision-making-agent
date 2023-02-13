#!/usr/bin/env python3
#encoding: utf-8

'''
This script is part of the program to simulate a navigation experiment where a robot
has to discover the rewarded state of the arena in which it evolves using a meta-control
decision algorithm : "Dromnelle, R., Renaudo, E., Chetouani, M., Maragos, P., Chatila, R.,
Girard, B., & Khamassi, M. (2022). Reducing Computational Cost During Robot Navigation and 
Human–Robot Interaction with a Human-Inspired Reinforcement Learning Architecture. 
International Journal of Social Robotics, 1-27."

With this script, the simulated agent use a Deep Q-Network algorithm (deep reinforcement learning) 
to learn to solve the task.
'''

__author__ = "Rémi Dromnelle"
__version__ = "1.0"
__maintainer__ = "Rémi Dromnelle"
__email__ = "remi.dromnelle@gmail.com"
__status__ = "Devlopment"

from utility import *

VERSION = 1

class DQN:
	"""
	This class implements a deep model-free reinforcement learning algorithm (Deep Q-Network).
    """

	def __init__(self, expert, experiment, env_file, initial_variables, spaces, boundaries_exp, parameters, log):
		"""
		Iinitialize values and models
		"""
		# ---------------------------------------------------------------------------
		# initialize all the variables which will be used
		self.ID = expert
		self.experiment = experiment
		action_count = initial_variables["action_count"]
		self.init_qvalue = initial_variables["qvalue"]
		init_actions_prob = initial_variables["actions_prob"]
		self.action_space = spaces["actions"]
		self.state_space = spaces["states"]
		self.max_reward = boundaries_exp["max_reward"]
		self.duration = boundaries_exp["duration"]
		self.window_size = boundaries_exp["window_size"]
		self.alpha = parameters["alpha"]
		self.gamma = parameters["gamma"]
		self.beta = parameters["beta"]
		self.beta_max = parameters["beta_max"]
		self.beta_growth = parameters["beta_growth"]
		self.log = log["log"]
		self.summary = log["summary"]
		self.not_learn = False
		self.wait_new_goal = True
		self.rewarded_state = None
		# ---------------------------------------------------------------------------
		# Build DQN models
		self.model = self.build_model()
		model_weights = self.model.get_weights()
		self.target_model = self.build_model()
		self.target_model.set_weights(model_weights)
		# Inputs for previous (ps) and current (cs) states
		self.input_ps = np.array([0]*self.state_space)
		self.input_cs = np.array([0]*self.state_space)
		# How many last samples to keep for model training
		self.train_start = 256
		self.replay_memory_size = 1000
		self.replay_memory = col.deque(maxlen = self.replay_memory_size)
		# How many samples to use for training
		self.minibatch_size = 256
		# Update target 
		self.update_target_every = 20
		self.target_update_counter = 0
		# ---------------------------------------------------------------------------
		#self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/DQN-{int(time.time())}")
		# ---------------------------------------------------------------------------
		# // List and dicts for store data //
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

	def build_model(self):
		"""
		Neural Net for Deep-Q learning Model
		"""
		# ---------------------------------------------------------------------------
		model = Sequential()
		model.add(Dense(64, input_dim = self.state_space, activation = 'relu'))
		model.add(Dense(64, activation = 'relu'))
		model.add(Dense(self.action_space, activation = 'linear'))
		model.summary()
		model.compile(loss = 'mse', optimizer = Adam(lr = self.alpha), metrics = ["accuracy"])
		# ---------------------------------------------------------------------------
		return model
		# ---------------------------------------------------------------------------

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
		for a in range(0,self.action_space):
			actions[str(a)] = a
			qvals[str(a)] = qvalues[a]
		# ----------------------------------------------------------------------------
		# Soft-max function
		#actions_prob = {"0": 0.125, "1": 0.125, "2": 0.125, "3": 0.125, "4": 0.125, "5": 0.125, "6": 0.125, "7": 0.125}
		actions_prob = softmax_actions_prob(qvals, self.beta)
		new_probs = list()
		for prob in actions_prob.values():
			new_probs.append(prob)
		set_actions_prob(self.dict_actions_prob, current_state, new_probs)
		# ---------------------------------------------------------------------------
		# For each action, sum the probabilitie of selection with a low pass filter
		old_probs = get_filtered_prob(self.dict_actions_prob, current_state)
		filtered_actions_prob = list()
		for a in range(0,len(new_probs)):
			filtered_actions_prob.append(low_pass_filter(self.alpha, old_probs[a], new_probs[a]))
		set_filtered_prob(self.dict_actions_prob, current_state, filtered_actions_prob)
		# ----------------------------------------------------------------------------
		# The end of the soft-max function
		#choosen_action = rargmax(qvalues)
		decision, choosen_action = softmax_decision(actions_prob, actions)
		# ----------------------------------------------------------------------------
		return choosen_action
		# ----------------------------------------------------------------------------

	def infer(self, current_state):
		"""
		Predict the output of the neural network
		"""
		# ----------------------------------------------------------------------------
		qvalues = self.model.predict(np.array(self.input_cs).reshape(1,self.state_space))[0]
		for action in range(0,self.action_space):
			self.dict_qvalues[(str(current_state),"qvals")][int(action)] = qvalues[action]
		# ----------------------------------------------------------------------------
		return self.dict_qvalues[(str(current_state),"qvals")]
		# ----------------------------------------------------------------------------

	def learn_whith_replay(self):
		"""
		Train the neural network with replay
		"""
		# ----------------------------------------------------------------------------
		# Get a minibatch of random samples from memory replay table
		minibatch = random.sample(self.replay_memory, self.minibatch_size)
		# Get current stats from minibath
		previous_states = np.array([transition[0] for transition in minibatch])
		current_states = np.array([transition[3] for transition in minibatch])
		# Query neural network model for qvalues in one times instead in the loop
		qvalues_previous_states = self.model.predict(previous_states)
		qvalues_current_states = self.target_model.predict(current_states)
		# ----------------------------------------------------------------------------
		# Data structure for training
		x = list()
		y = list()
		# ----------------------------------------------------------------------------
		for index, (previous_state, action, reward, current_state) in enumerate(minibatch):
			# ------------------------------------------------------------------------
			if reward == 1:
				new_qvalue = reward
			else:
				max_qvalues_current_state = np.amax(qvalues_current_states[index])
				new_qvalue = reward + self.gamma * max_qvalues_current_state
			# ------------------------------------------------------------------------
			qvalues_previous_state = qvalues_previous_states[index]
			qvalues_previous_state[action] = new_qvalue
			# ------------------------------------------------------------------------
			# Feed the data for training
			x.append(previous_state)
			y.append(qvalues_previous_state)
		# ----------------------------------------------------------------------------
		self.model.fit(np.array(x), np.array(y), batch_size = self.minibatch_size, verbose = 1)
		# ----------------------------------------------------------------------------

	def learn(self, reward, action):
		"""
		Train the neural network
		"""
		# ---------------------------------------------------------------------------
		input_ps = np.reshape(self.input_ps, [1,self.state_space])
		input_cs = np.reshape(self.input_cs, [1,self.state_space])
		qvalues_current_state = self.target_model.predict(input_cs)
		max_qvalues_current_state = np.max(qvalues_current_state)
		if reward == 1:
			new_qvalue = reward
			print(str(reward))
		else:
			new_qvalue = reward + self.gamma * max_qvalues_current_state
			print(f"{reward} + {self.gamma} * {max_qvalues_current_state} = {new_qvalue}")
		# -------------------------c--------------------------------------------------
		targeted_qvalues = self.model.predict(input_ps)
		targeted_qvalues[0][action] = new_qvalue
		# ----------------------------------------------------------------------------
		self.model.fit(input_ps, targeted_qvalues, verbose = 1)
		# ----------------------------------------------------------------------------

	def update_memory(self, input_ps, decided_action, reward_obtained, input_cs):
		# ----------------------------------------------------------------------------
		self.replay_memory.append((input_ps, decided_action, reward_obtained, input_cs))
		# ----------------------------------------------------------------------------

	def run(self, action_count, cumulated_reward, reward_obtained, previous_state, decided_action, current_state, do_we_plan, new_goal): 
		"""
		Run the model-free Deep RL expert
		"""
		# ----------------------------------------------------------------------------
		print("------------------------ DQN --------------------------------")
		# ----------------------------------------------------------------------------
		# Update the actioncount and the number of the visits for the previous state
		self.dict_duration["actioncount"] = action_count
		self.dict_actions_prob["actioncount"] = action_count
		self.dict_qvalues[(str(current_state),"actioncount")] = action_count
		self.dict_qvalues[(str(previous_state),"visits")] += 1
		# ---------------------------------------------------------------------------
		# Update beta value
		self.beta = min(self.beta * self.beta_growth, self.beta_max)
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
		# If the previous state is the reward state, the agent must not run its
		# learning process
		if previous_state == self.rewarded_state:
			self.not_learn = True
		else:
			self.not_learn = False
		# ---------------------------------------------------------------------------
		# Update the input vectors for the DQN
		self.input_ps = np.array([0]*self.state_space)
		self.input_ps[int(previous_state)] = 1
		self.input_cs = np.array([0]*self.state_space)
		self.input_cs[int(current_state)] = 1
		# ----------------------------------------------------------------------------
		if self.not_learn == False:
			self.update_memory(self.input_ps, decided_action, reward_obtained, self.input_cs)
			# Save the transition on memory for replaying
			if len(self.replay_memory) > self.train_start and cumulated_reward > 0:
				self.learn_whith_replay()
			#if cumulated_reward > 0:
				#self.learn(reward_obtained, decided_action)
		# ----------------- ----------------------------------------------------------
		self.target_update_counter += 1
		if self.target_update_counter == self.update_target_every:
			print("Update target model !")
			self.target_model.set_weights(self.model.get_weights())
			self.target_update_counter = 0
		# ----------------------------------------------------------------------------
		# If the expert was choosen to plan, compute the news probabilities of actions
		if do_we_plan:
			# ------------------------------------------------------------------------
			old_time = datetime.datetime.now()
			# ------------------------------------------------------------------------
			# Run the process of inference
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
			# ------------------------------------------------------------------------
		decided_action = self.decide(current_state, qvalues)
		# ----------------------------------------------------------------------------
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
				prefixe = 'v%d_TBDQN_'%(VERSION)
				self.summary_log = open(prefixe+'summary_log.dat', 'a')
				self.summary_log.write(f"{self.alpha} {self.gamma} {self.beta} {cumulated_reward}\n")
		# ---------------------------------------------------------------------------
		return decided_action
		# ---------------------------------------------------------------------------









