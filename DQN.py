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
__version__ = "2.0"
__maintainer__ = "Rémi Dromnelle"
__email__ = "remi.dromnelle@gmail.com"
__status__ = "Development"


from utility import *

VERSION = 2

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
		self.learning_rate = parameters["alpha"]
		self.learning_rate_decay = parameters["alpha_decay"]
		self.learning_rate_min = parameters["alpha_min"]
		self.gamma = parameters["gamma"]
		self.epsilon = parameters["epsilon"]
		self.epsilon_min = parameters["epsilon_min"]
		self.epsilon_decay = parameters["epsilon_decay"]
		self.log = log["log"]
		self.summary = log["summary"]
		self.not_learn = False
		self.wait_new_goal = True
		self.rewarded_state = None
		# ---------------------------------------------------------------------------
		# Build DQN models
		self.model = self._build_model()
		model_weights = self.model.get_weights()
		self.target_model = self._build_model()
		self.target_model.set_weights(model_weights)
		self.epoch_number = 1
		# Build inputs
		self.inputs = tf.one_hot(range(self.state_space), self.state_space)
		# Replay memory parameters
		self.minibatch_size = 64
		self.memory = list()
		self.memory_size = 256
		self.priorities = np.array([])
		self.alpha = 0.6
		self.beta = 0.4
		self.pos = 0
		self.beta_increment_per_sampling = 0.001
		# Update target 
		self.update_target_threshold = 50
		self.update_target_counter = 0
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

	def _build_model(self):
		"""
		Neural Net for Deep-Q learning model
		"""
		# ---------------------------------------------------------------------------
		model = Sequential()
		model.add(Dense(20, input_shape = (self.state_space,), activation = 'relu'))
		model.add(Dense(20, activation = 'relu'))
		model.add(Dense(self.action_space, activation = 'linear'))
		model.summary()
		model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate = self.learning_rate))
		# ---------------------------------------------------------------------------
		return model
		# ---------------------------------------------------------------------------

	def _decide(self, current_state, qvalues):
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
		print(f"Beta = {1/self.epsilon}")
		actions_prob = softmax_actions_prob(qvals, 1/self.epsilon)
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
		_, choosen_action = softmax_decision(actions_prob, actions)
		# ----------------------------------------------------------------------------
		return choosen_action
		# ----------------------------------------------------------------------------

	def _infer(self, current_state):
		"""
		Predict the output of the neural network
		"""
		# ----------------------------------------------------------------------------
		X = tf.expand_dims(self.inputs[current_state], axis = 0)
		qvalues = self.model.predict(X)
		for action in range(0,self.action_space):
			self.dict_qvalues[(str(current_state),"qvals")][int(action)] = qvalues[0][action]
		# ----------------------------------------------------------------------------
		return self.dict_qvalues[(str(current_state),"qvals")]
		# ----------------------------------------------------------------------------

	def _update_target_model(self):
		"""
		Update the target model
		"""
		# ----------------------------------------------------------------------------
		self.update_target_counter += 1
		if self.update_target_counter == self.update_target_threshold:
			self.target_model.set_weights(self.model.get_weights())
			self.update_target_counter = 0
		# ----------------------------------------------------------------------------

	def _get_samples(self, indices):
		"""
		Get a memory sample according to priorities
		"""
		# ----------------------------------------------------------------------------
		samples = np.array(self.memory)[indices]
		# ----------------------------------------------------------------------------
		previous_states = np.array(samples[:, 0])
		actions = np.array(samples[:, 1])
		rewards = np.array(samples[:, 2])
		current_states = np.array(samples[:, 3])
		dones = np.array(samples[:, 4])
		# ----------------------------------------------------------------------------
		return previous_states, actions, rewards, current_states, dones
		# ----------------------------------------------------------------------------

	def _sample_indices(self):
		"""
		Sample indices
		"""
		# ----------------------------------------------------------------------------
		probabilities = self.priorities ** self.alpha
		probabilities = probabilities / probabilities.sum()
		# ----------------------------------------------------------------------------
		return np.random.choice(len(self.memory), size = self.minibatch_size, p = probabilities)
		# ----------------------------------------------------------------------------

	def _learn(self):
		"""
		Train the neural network with prioritized replay
		"""
		# ----------------------------------------------------------------------------
		# Get a minibatch of according to priorities
		indices = self._sample_indices()
		previous_states, actions, rewards, current_states, dones = self._get_samples(indices)
		# ----------------------------------------------------------------------------
		# Extract inputs
		inputs_ps = tf.gather(self.inputs, indices = previous_states)
		imputs_cs = tf.gather(self.inputs, indices = current_states)
		# Calculate TD error and weights
		qvalues_current_states = self.target_model.predict(imputs_cs)
		max_qvalues_current_state = np.max(qvalues_current_states, axis = 1)
		qvalues_previous_states = self.model.predict(inputs_ps)
		errors = np.abs(rewards + self.gamma * max_qvalues_current_state - np.squeeze(qvalues_previous_states[np.arange(len(actions)), actions]))
		self.priorities[indices] = errors + 0.000006
		importance_sampling_weights = np.power(len(self.memory) * self.priorities[indices], -self.beta)
		importance_sampling_weights =  importance_sampling_weights / importance_sampling_weights.max()
		# ----------------------------------------------------------------------------
		# Update the model
		self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
		qvalues_previous_states[np.arange(len(actions)), actions] = rewards + (1 - dones) * self.gamma * max_qvalues_current_state
		self.model.fit(inputs_ps, qvalues_previous_states, sample_weight = importance_sampling_weights, verbose = 1)
		# ----------------------------------------------------------------------------

	def _add_memory(self, previous_state, decided_action, reward_obtained, current_state):
		"""
		Add or replace the current transition in the memory buffer
		"""
		# ----------------------------------------------------------------------------
		if reward_obtained == 1:
			done = 1
		else:
			done = 0
		# ----------------------------------------------------------------------------
		if len(self.memory) < self.memory_size :
			self.memory.append((previous_state, decided_action, reward_obtained, current_state, done))
			self.priorities = np.append(self.priorities, max(self.priorities, default = 1))
		else:
			self.memory[self.pos] = (previous_state, decided_action, reward_obtained, current_state, done)
			self.priorities[self.pos] = max(self.priorities, default = 1)
		self.pos = (self.pos + 1) % self.memory_size
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
		# Memorize the transition and the priorities (except for the fake transition 
		# final state -> start state)
		if previous_state != self.rewarded_state:
			self._add_memory(int(previous_state), decided_action, reward_obtained, int(current_state))
		# ----------------------------------------------------------------------------
		# Run the learning process and update eventually the target model
		if len(self.memory) >= self.minibatch_size:
			self._learn()
			self._update_target_model()
			# Update learning rate
			self.learning_rate = max(self.learning_rate * self.learning_rate_decay, self.learning_rate_min)
			print(f"Alpha = {self.learning_rate}")
		# ----------------------------------------------------------------------------
		# If the expert was choosen to plan, compute the news probabilities of actions
		if do_we_plan:
			# ------------------------------------------------------------------------
			old_time = datetime.datetime.now()
			# ------------------------------------------------------------------------
			# Run the process of inference
			qvalues = self._infer(int(current_state))
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
		# Choose the next action to do from the current state using a soft-max policy
		decided_action = self._decide(current_state, qvalues)
		# ----------------------------------------------------------------------------
		# Update epsilon value
		self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
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
				self.summary_log.write(f"{self.alpha} {self.gamma} {self.epsilon} {cumulated_reward}\n")
		# ---------------------------------------------------------------------------
		return decided_action
		# ---------------------------------------------------------------------------









