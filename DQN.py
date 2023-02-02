#!/usr/bin/env python3
#encoding: utf-8

'''
This script is part of the program to simulate a navigation experiment where a robot
has to discover the rewarded state of the arena in which it evolves using a meta-control
decision algorithm : "Dromnelle, R., Renaudo, E., Chetouani, M., Maragos, P., Chatila, R.,
Girard, B., & Khamassi, M. (2022). Reducing Computational Cost During Robot Navigation and 
Human–Robot Interaction with a Human-Inspired Reinforcement Learning Architecture. 
International Journal of Social Robotics, 1-27."

With this script, the simulated agent use DQN algorithm (model-free behavior) 
to learn the task and decide.
'''

__author__ = "Rémi Dromnelle"
__version__ = "1"
__maintainer__ = "Rémi Dromnelle"
__email__ = "remi.dromnelle@gmail.com"
__status__ = "Production"

from utility import *

VERSION = 1


class DQN:
	"""
	This class implements a model-free learning algorithm (q-learning).
    """

	def __init__(self, experiment, map_file, initial_variables, action_space, state_space, boundaries_exp, parameters, options_log):
		"""
		Iinitialise values and models
		"""
		# ---------------------------------------------------------------------------
		# Initialise all the variables which will be used
		self.experiment = experiment
		self.max_reward = boundaries_exp["max_reward"]
		self.duration = boundaries_exp["duration"]
		self.window_size = boundaries_exp["window_size"]
		self.gamma = 0.95
		self.alpha = 0.1
		self.alpha_min = 0.01
		self.alpha_decay = 1.001
		self.beta = 40
		self.beta_max = 50
		self.beta_decay = 1.005
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
		init_actions_prob = initial_variables["actions_prob"]
		self.not_learn = False
		self.action_space = action_space
		self.state_space = state_space
		# ---------------------------------------------------------------------------
		# Build DQN models
		self.model = self._build_model()
		self.target_model = self._build_model()
		self.target_model.set_weights(self.model.get_weights())
		# Inputs for previous (ps) and current (cs) states
		self.input_ps = np.array([0]*self.state_space)
		self.input_cs = np.array([0]*self.state_space)
		# How many last samples to keep for model training
		self.train_start = 128
		self.replay_memory_size = 1000
		self.replay_memory = col.deque(maxlen = self.replay_memory_size)
		# How many samples to use for training
		self.minibatch_size = 128
		# Update target 
		self.update_target_every = 20
		self.target_update_counter = 0
		# ---------------------------------------------------------------------------
		self.final_state = -1 
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
		# Load the transition model which will be used as map
		with open(map_file,'r') as file2:
			self.map = json.load(file2)
		# For each state of the map : 
		for state in self.map["transitionActions"]:
			s = str(state["state"])
			t = state["transitions"]
			# -----------------------------------------------------------------------
			self.dict_qvalues[(s,"qvals")] = [self.init_qvalue]*8
			self.dict_qvalues[(s,"visits")] = 0
			# -----------------------------------------------------------------------
			# - initialise the "probabilties of actions" dict
			self.dict_actions_prob["values"].append({"state": s, "actions_prob": [init_actions_prob]*8, "filtered_prob": [init_actions_prob]*8})
			# -----------------------------------------------------------------------
			# - initialise the "identity of the selected action" dict
			self.dict_decision["values"].append({"state": s, "history_decisions": [[0]*6,[0]*6,[0]*6,[0]*6,[0]*6,[0]*6,[0]*6,[0]*6]})
			# -----------------------------------------------------------------------
			# - initialise the duration dict
			self.dict_duration["values"].append({"state": s, "duration": 0.0})
		# ---------------------------------------------------------------------------
		# Initialise logs
		# self.directory_flag = False
		# if not os.path.exists("logs"):
		# 	os.mkdir("logs") 
		# os.chdir("logs")
		# if not os.path.exists("DQN"):
		# 	os.mkdir("DQN") 
		# os.chdir("DQN")
		# if self.log == True:
		# 	directory = "exp"+str(self.experiment)+"_alpha"+str(self.alpha)+"_gamma"+str(self.gamma)+"_beta"+str(self.beta)
		# 	if not os.path.exists(directory):
		# 		os.makedirs(directory)
		# 	os.chdir(directory)
		# 	self.directory_flag = True
		# 	# -----------------------------------------------------------------------
		# 	prefixe = "v"+str(VERSION)+"_TBDQN_exp"+str(self.experiment)+"_"
		# 	# -----------------------------------------------------------------------
		# 	self.reward_log = open(prefixe+'reward_log.dat', 'w')
		# 	self.reward_log.write("timecount"+" "+str(action_count)+" "+str(init_reward)+" currentTime-nodeStartTime"+" currentTime"+"\n")
		# 	# -----------------------------------------------------------------------
		# 	self.states_evolution_log = open(prefixe+'statesEvolution_log.dat', 'w')
		# 	self.states_evolution_log.write("timecount"+" "+str(action_count)+" "+current_state+" "+previous_state+ \
		# 		" currentContactState"+" currentViewState"+" "+str(decided_action)+" currentTime-nodeStartTime"+" currentTime"+"\n")
		# 	# -----------------------------------------------------------------------
		# 	#self.qvalues_evolution_log = open(prefixe+'qvaluesEvolution_log.dat', 'w')
		# 	#self.qvalues_evolution_log.write('{\n"logs" :\n['+json.dumps(self.dict_qvalues))
		# 	# -----------------------------------------------------------------------
		# 	self.actions_evolution_log = open(prefixe+'actions_evolution_log.dat', 'w')
		# 	self.actions_evolution_log.write('{\n"logs" :\n['+json.dumps(self.dict_actions_prob))
		# 	# -----------------------------------------------------------------------
		# 	self.monitoring_values_log = open(prefixe+'monitoring_values_log.dat', 'w')
		# 	self.monitoring_values_log.write(str(action_count)+" "+str(init_plan_time)+" "+str(abs(init_delta))+" "+str(init_delta)+" "+str(init_delta)+"\n")
		# # ---------------------------------------------------------------------------
		# os.chdir("../../../")
		# ---------------------------------------------------------------------------


	def __del__(self):
		"""
		Close all log files
		"""
		# ---------------------------------------------------------------------------
		# if self.log == True:
		# 	self.reward_log.close()
		# 	#self.qvalues_evolution_log.close()
		# 	self.actions_evolution_log.close()
		# 	self.states_evolution_log.close()
		# 	self.monitoring_values_log.close()
		# ---------------------------------------------------------------------------

	def _build_model(self):
		"""
		Neural Net for Deep-Q learning Model
		"""
		# ---------------------------------------------------------------------------
		model = Sequential()
		model.add(Dense(76, input_dim = self.state_space, activation = 'relu'))
		model.add(Dense(76, activation = 'relu'))
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
		for a in range(0,8):
			actions[str(a)] = a
			qvals[str(a)] = qvalues[a]
		# ----------------------------------------------------------------------------
		# Soft-max function
		#actions_prob = {"0": 0.125, "1": 0.125, "2": 0.125, "3": 0.125, "4": 0.125, "5": 0.125, "6": 0.125, "7": 0.125}
		actions_prob = softmax_actions_prob(qvals, self.beta)
		new_probs = list()
		for action, prob in actions_prob.items():
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
		return choosen_action, actions_prob
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
		# ----------------------------------------------------------------------------
		# Get a minibatch of random samples from memory replay table
		minibatch = random.sample(self.replay_memory, self.minibatch_size)
		# Get current stats from minibath
		previous_states = np.array([transition[0] for transition in minibatch])
		current_states = np.array([transition[3] for transition in minibatch])
		# Query neural network model for q-values
		qvalues_previous_states = self.model.predict(previous_states)
		qvalues_current_states = self.target_model.predict(current_states)
		# ----------------------------------------------------------------------------
		# Data structure for training
		x = []
		y = []
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
			plop = qvalues_previous_state[action]
			qvalues_previous_state[action] = new_qvalue
			#print(np.where(previous_state == 1), reward, plop, new_qvalue)
			# if reward == 1:
			# 	print(np.where(previous_state == 1))
			# 	print(self.model.predict(np.array(previous_state).reshape(1,self.state_space))[0])
			# 	print(qvalues_previous_state)
			# ------------------------------------------------------------------------
			# Feed the data for training
			x.append(previous_state)
			y.append(qvalues_previous_state)
			# if reward == 1:
			# 	print("Target")
			# 	print(qvalues_previous_state)
		# ----------------------------------------------------------------------------
		#print("State 19")
		input_ps_19 = np.array([0]*self.state_space)
		input_ps_19[19] = 1
		#print(self.model.predict(np.array(input_ps_19).reshape(1,self.state_space))[0])
		#print("State 17")
		input_ps_17 = np.array([0]*self.state_space)
		input_ps_17[17] = 1
		#print(self.model.predict(np.array(input_ps_17).reshape(1,self.state_space))[0])
		self.model.fit(np.array(x), np.array(y), batch_size = self.minibatch_size, verbose = 1)
		#print("State 19")
		#print(self.model.predict(np.array(input_ps_19).reshape(1,self.state_space))[0])
		#print("State 17")
		#print(self.model.predict(np.array(input_ps_17).reshape(1,self.state_space))[0])
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
		print("Action : "+ str(action))
		if reward == 1:
			new_qvalue = reward
			print(str(reward))
		else:
			new_qvalue = reward + self.gamma * max_qvalues_current_state
			print(str(reward)+" + "+str(self.gamma)+" * "+str(max_qvalues_current_state)+" = "+str(new_qvalue))
		# -------------------------c--------------------------------------------------
		targeted_qvalues = self.model.predict(input_ps)
		targeted_qvalues[0][action] = new_qvalue
		print(self.model.predict(np.array(input_ps).reshape(1,self.state_space))[0])
		#print(targeted_qvalues)
		# ----------------------------------------------------------------------------
		self.model.fit(input_ps, targeted_qvalues, verbose = 1)
		print(self.model.predict(np.array(input_ps).reshape(1,self.state_space))[0])
		# ----------------------------------------------------------------------------

	def update_memory(self, input_ps, decided_action, reward_obtained, input_cs):
		# ----------------------------------------------------------------------------
		self.replay_memory.append((input_ps, decided_action, reward_obtained, input_cs))
		# ----------------------------------------------------------------------------

	def meta_learning(self, reward_obtained, cumulated_reward, current_state):
		# ----------------------------------------------------------------------------
		# Save the identity of final state
		if reward_obtained == 1:
			self.final_state = current_state
		# Reset the value of beta and alpha if the final state doesn't give reward anymore
		if current_state == self.final_state and reward_obtained != 1:
			self.beta = 1
			#self.alpha = 0.6
		# ----------------------------------------------------------------------------
		# Progressively increase the beta and decrease the alpha values
		if cumulated_reward > 0:
			if self.beta < self.beta_max:
				self.beta *= self.beta_decay
			if self.alpha > self.alpha_min:
				self.alpha /= self.alpha_decay
		print("Beta = "+str(self.beta))
		print("Alpha = "+str(self.alpha))
		# ----------------------------------------------------------------------------

	def run(self, action_count, cumulated_reward, reward_obtained, previous_state, decided_action, current_state, do_we_plan): 
		"""
		Run the model-free system
		"""
		# ----------------------------------------------------------------------------
		print("------------------------ DQN --------------------------------")
		# ----------------------------------------------------------------------------
		# Update the actioncount and the number of the visits for the previous state
		self.dict_duration["actioncount"] = action_count
		self.dict_actions_prob["actioncount"] = action_count
		self.dict_qvalues[(str(current_state),"actioncount")] = action_count
		self.dict_qvalues[(str(previous_state),"visits")] += 1
		# ----------------------------------------------------------------------------
		# Update the input vectors for the DQN
		self.input_ps = np.array([0]*self.state_space)
		self.input_ps[int(previous_state)] = 1
		self.input_cs = np.array([0]*self.state_space)
		self.input_cs[int(current_state)] = 1
		# ----------------------------------------------------------------------------
		#self.meta_learning(reward_obtained, cumulated_reward, current_state)
		# ----------------------------------------------------------------------------
		if self.not_learn == False:
			self.update_memory(self.input_ps, decided_action, reward_obtained, self.input_cs)
			# Save the transition on memory for replaying
			if len(self.replay_memory) > self.train_start and cumulated_reward > 0:
				self.learn_whith_replay()
			#if cumulated_reward > 0:
			#	self.learn(reward_obtained, decided_action)
		else:
			print("Not learn at this iteration !")
		# ----------------- ----------------------------------------------------------
		self.target_update_counter += 1
		if self.target_update_counter == self.update_target_every:
			print("Update target_model !")
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
		decided_action, actions_prob = self.decide(current_state, qvalues)
		# ----------------------------------------------------------------------------
		# Maj the history of the decisions
		set_history_decision(self.dict_decision, current_state, decided_action, self.window_size)
		prefered_action = [0]*8
		for action in range(0,len(prefered_action)):
			for dictStateValues in self.dict_decision["values"]:
				if dictStateValues["state"] == current_state:
					prefered_action[action] = sum(dictStateValues["history_decisions"][action])
		# ----------------------------------------------------------------------------
		# Prepare data to return 
		plan_time = get_duration(self.dict_duration, current_state)
		selection_prob = get_filtered_prob(self.dict_actions_prob, current_state)
		# ----------------------------------------------------------------------------
		if reward_obtained > 0.0:
			self.not_learn = True
			for a in range(0,8):
				self.dict_qvalues[(current_state,"qvals")] = [0.0]*8
		else:
			self.not_learn = False
		# ---------------------------------------------------------------------------
		# Logs
		# if self.log == True:
		# 	self.reward_log.write("timecount"+" "+str(action_count)+" "+str(reward_obtained)+" currentTime-nodeStartTime"+" currentTime"+"\n")
		# 	self.states_evolution_log.write("timecount"+" "+str(action_count)+" "+current_state+" "+previous_state+ \
		# 		" currentContactState"+" currentViewState"+" "+str(decided_action)+" currentTime-nodeStartTime"+" currentTime"+"\n")
		# 	#self.qvalues_evolution_log.write(",\n"+json.dumps(self.dict_qvalues))
		# 	self.actions_evolution_log.write(",\n"+json.dumps(self.dict_actions_prob))
		# 	self.monitoring_values_log.write(str(action_count)+" "+str(decided_action)+" "+str(plan_time)+" "+str(selection_prob)+" "+str(prefered_action)+"\n")
		# ---------------------------------------------------------------------------
		# Finish the logging at the end of the simulation (duration or max reward)
		if (action_count == self.duration) or (cumulated_reward == self.max_reward):
			#if self.log == True:
			#	self.qvalues_evolution_log.write('],\n"name" : "Qvalues"\n}')
			#	self.actions_evolution_log.write('],\n"name" : "Actions"\n}')
			# -----------------------------------------------------------------------
			# Build the summary file 
			if self.summary == True:
				if self.directory_flag == True:
					os.chdir("../")
				# -------------------------------------------------------------------
				prefixe = 'v%d_TBDQN_'%(VERSION)
				self.summary_log = open(prefixe+'summary_log.dat', 'a')
				self.summary_log.write(str(self.alpha)+" "+str(self.gamma)+" "+str(self.beta)+" "+str(cumulated_reward)+"\n")
		# ---------------------------------------------------------------------------
		print("Qvalues : ")
		for action in range(0,8):
			print(self.dict_qvalues[str(current_state),"qvals"][int(action)])
		# ----------------------------------------------------------------------------
		return decided_action
		# ----------------------------------------------------------------------------









