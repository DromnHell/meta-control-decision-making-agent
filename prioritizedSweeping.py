#!/usr/bin/env python3
#encoding: utf-8

'''
With this script, the simulated agent use a Prioritized Sweeping algorithm (model-basedbehavior) 
to learn to solve the task.. It was developped by Jeanne Barthelemy during its intership.

With this script, the simulated agent use a Prioritized Sweeping algorithm (model-based
reinforcement learning) to learn to solve the task.

Warning : Currently not connected to the program
'''

__author__ = "Jeanne Barthelemy & Rémi Dromnelle"
__version__ = "1.0"
__maintainer__ = "Rémi Dromnelle"
__email__ = "remi.dromnelle@gmail.com"
__status__ = "Development"

from utility import *

VERSION = 1


class ModelBased:
	"""
	This class implements a model-based learning algorithm (Prioritized Sweeping).
    """

	def __init__(self, experiment, env_file, initial_variables, actions_space, boundaries_exp, parameters, log):
		"""
		Iinitialize values and models
		"""
		# -----------------------------------------------------------------------------
		# initialize all the variables which will be used
		self.print = log["print"]
		self.experiment = experiment
		self.max_reward = boundaries_exp["max_reward"]
		self.duration = boundaries_exp["duration"]
		self.window_size = boundaries_exp["window_size"]
		self.epsilon = boundaries_exp["epsilon"]
		self.alpha = parameters["alpha"]
		self.gamma = parameters["gamma"]
		self.beta = parameters["beta"]
		#self.log = False #log["log"]
		self.logNumber = log["logNumber"]
		self.summary = log["summary"]
		action_count = initial_variables["action_count"]
		decided_action = initial_variables["decided_action"]
		previous_state = initial_variables["previous_state"]
		current_state = initial_variables["current_state"]
		self.init_qvalue = initial_variables["qvalue"]
		init_reward = initial_variables["reward"]
		init_delta = initial_variables["delta"]
		init_plan_time = initial_variables["plan_time"]
		self.actions_space = actions_space
		init_actions_prob = initial_variables["actions_prob"]
		self.not_learn = False
		# Time constant for filtering
		#self.period_PT_MB = 0.6
		#self.period_delta_MB = 0.6
		# ----------------------------------------------------------------------------
		# // List and dicts for store data //
		# Create the list of states
		self.list_states = list()
		# Create the transitions dict according to the type of log used by Erwan
		
		self.dict_transitions = dict()
		self.dict_transitions["actioncount"] = action_count
		self.dict_transitions["transitionActions"] = list()
		

		# Create the rewards dict according to the type of log used by Erwan
		self.dict_rewards = dict()
		self.dict_rewards["actioncount"] = action_count
		self.dict_rewards["transitionActions"] = list()
		# Create the qvalues dict according to the type of log used by Erwan
		self.dict_qvalues = dict()
		#self.dict_qvalues["actioncount"] = action_count
		#self.dict_qvalues["values"] = list()
		#self.dict_qvalues["deltaQ"] = init_delta
		#tab_act = list()
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
		# Create a dict that contains the time of planification for each states
		self.dict_duration = dict()
		self.dict_duration["actioncount"] = action_count
		self.dict_duration["values"] = list()
		
		# -----------------------------------------------------------------------------
		# // Replay //
		#Create a buffer and for the replays
		self.bufferRPE=np.empty((4,0), str) #current_state action RPE abs(RPE) 
		#Initialisation of variables for the replays
		self.replaythreshold=log["thresholdMB"]
		self.replaywindow=log["PS_windows"]
		self.replay_budget=log["PS_budget"]
		self.replayMB=log["replayMB"]



		# -----------------------------------------------------------------------------
		# Load the transition model which will be used as environment
		with open(env_file,'r') as file2:
			self.env = json.load(file2)
		# For each state of the map : 
		for state in self.env["transitionActions"]:
			s = str(state["state"])
			t = state["transitions"]
			q = list()
			# -------------------------------------------------------------------------
			# - identify the different possible transitions using the previous state and the decided action
			'''
			for transition in t:
				q.append(transition["action"])
				# if s == previous_state and transition["action"] == decided_action:
				# 	for i in range(transition["prob"]):
				# 		tab_act.append(str(transition["state"]))
			# -------------------------------------------------------------------------
			# - initialize the qvalues dict 
			
			dictStateValues = dict()
			dictStateValues["state"] = s
			dictStateValues["values"] = list()
			dictStateValues["deltaQ"] = init_delta
			dictStateValues["visits"] = 0
			for i in list(set(q)):
				dictActionValue = dict()
				dictActionValue["action"] = i
				dictActionValue["value"] = self.init_qvalue
				dictStateValues["values"].append(dictActionValue)
			'''

			self.dict_qvalues[(s,"qvals")]=[self.init_qvalue]*8
			self.dict_qvalues[(s,"visits")]=0
			# -----------------------------------------------------------------------
			# - initialize the probabilties of actions
			self.dict_actions_prob["values"].append({"state": s, "actions_prob": [init_actions_prob]*8, "filtered_prob": [init_actions_prob]*8})
			# -------------------------------------------------------------------------
			# - initialize the delta prob dict
			self.dict_delta_prob["values"].append({"state": s, "delta_prob": init_delta})
			# -------------------------------------------------------------------------
			# - initialize the duration dict
			self.dict_duration["values"].append({"state": s, "duration": init_delta})
		# -----------------------------------------------------------------------------
		# initialize logs
		self.directory_flag = False
		if self.logNumber in [1]:
			try:
				os.stat("logs"+str(self.logNumber))
			except:
				os.mkdir("logs"+str(self.logNumber)) 
			os.chdir("logs"+str(self.logNumber))
			try:
				os.stat("MB")
			except:
				os.mkdir("MB") 
			os.chdir("MB")
			directory = "exp"+str(self.experiment)+"_gamma"+str(self.gamma)+"_beta"+str(self.beta)
			if not os.path.exists(directory):
				os.makedirs(directory)
			os.chdir(directory) 
			self.directory_flag = True
			# -------------------------------------------------------------------------
			prefixe = "v"+str(VERSION)+"_TBMB_exp"+str(self.experiment)+"_"
			# -------------------------------------------------------------------------
			'''
			self.reward_log = open(prefixe+'reward_log.dat', 'w')
			self.reward_log.write("timecount"+" "+str(action_count)+" "+str(init_reward)+" "+"currentTime-nodeStartTime"+" "+"currentTime"+"\n")
			# -------------------------------------------------------------------------
			self.states_evolution_log = open(prefixe+'statesEvolution_log.dat', 'w')
			self.states_evolution_log.write("timecount"+" "+str(action_count)+" "+current_state+" "+previous_state+" "+"currentContactState"+ \
				" "+"currentViewState"+" "+str(decided_action)+"currentTime-nodeStartTime"+" "+"currentTime"+"\n")
			# -------------------------------------------------------------------------
			self.qvalues_evolution_log = open(prefixe+'qvaluesEvolution_log.dat', 'w')
			self.qvalues_evolution_log.write('{\n"logs" :\n['+json.dumps(self.dict_qvalues))
			# -----------------------------------------------------------------------
			self.actions_evolution_log = open(prefixe+'actions_evolution_log.dat', 'w')
			self.actions_evolution_log.write('{\n"logs" :\n['+json.dumps(self.dict_actions_prob))
			# -------------------------------------------------------------------------
			self.monitoring_values_log = open(prefixe+'monitoring_values_log.dat', 'w')
			self.monitoring_values_log.write(str(action_count)+" "+str(init_plan_time)+" "+str(init_delta)+" "+str(init_delta)+" "+str(init_delta)+"\n")
			'''
			# -------------------------------------------------------------------------
			self.replay_log = open(prefixe+'replay_log.dat', 'w')
			# -----------------------------------------------------------------------------
			os.chdir("../../../")
			# -----------------------------------------------------------------------------
		
		if self.logNumber in [5,6]:
			try:
				os.stat("logs"+str(self.logNumber))
			except:
				os.mkdir("logs"+str(self.logNumber)) 
			os.chdir("logs"+str(self.logNumber))
			self.replay_log = open(str(self.experiment)+"MB"+'replay_log.dat', 'w')
			os.chdir("../")

	def __del__(self):
		"""
		Close all log files
		"""
		# -----------------------------------------------------------------------------
		if self.logNumber in [1,5,6]:
			'''
			self.reward_log.close()
			self.qvalues_evolution_log.close()
			self.actions_evolution_log.close()
			self.states_evolution_log.close()
			self.monitoring_values_log.close()
			'''
			self.replay_log.close()
		# -----------------------------------------------------------------------------


	def _decide(self, previous_state, current_state):
		"""
		Choose the next action using soft-max policy
		"""
		# ----------------------------------------------------------------------------
		qvalues = self.dict_qvalues[(str(current_state),"qvals")]
		#act_tmp = list((qvalues.keys()))
		actions = dict()
		qvals = dict()
		# ----------------------------------------------------------------------------
		for a in range(0,8):
			actions[str(a)] = a
			qvals[str(a)] = qvalues[a] #repassage en mode dico pour compatibilité avec les fonctions de Rémi
		# ----------------------------------------------------------------------------
		# Soft-max function
		actions_prob = softmax_actions_prob(qvals, self.beta)
		new_probs = [0]*len(actions)
		for action, prob in actions_prob.items():
			new_probs[int(action)]=prob
		set_actions_prob(self.dict_actions_prob, current_state, new_probs)
		# -------------------------------------------------------------------------
		# For each action, sum the probabilitie of selection with a low pass filter
		if self.dict_qvalues[str(current_state),"visits"] == 1:
			filtered_actions_prob = new_probs
		else:
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
		"""
		# loop to find the good state
		for state in self.dict_qvalues["values"]:
			if state["state"] == this_state:
				# ---------------------------------------------------------------------
				# loop througth the qvalues
		"""
		for i in range (0,8):
			action = i 
			previous_qvalue = self.dict_qvalues[(str(this_state),"qvals")][action] 
			flag = False
			accu = 0.0
			# loop to recover reward
			for transition in self.dict_rewards["transitionActions"]:
				if str(transition["state"]) == this_state and transition["action"] == action:
					reward = transition["reward"]
					break
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
							accu += (prob * (reward + self.gamma * vValue))
					break
			# ----------------------------------------------------------------
			if flag == True:
				self.dict_qvalues[(str(this_state),"qvals")][int(action)] = accu
			#else:
			#	qvalue["value"] = self.init_qvalue
			# ----------------------------------------------------------------
			new_deltaQ = abs(self.dict_qvalues[(str(this_state),"qvals")][int(action)] - previous_qvalue)
			"""
			if get_visit(self.dict_qvalues, state["state"]) == 1:
				filtered_sum_deltaQ = new_deltaQ
				global_filtered_sum_deltaQ = new_deltaQ
			else:
				old_deltaQ = get_deltaQ(self.dict_qvalues, state["state"])
				old_global_deltaQ = self.dict_qvalues["deltaQ"]
				filtered_sum_deltaQ = low_pass_filter(self.alpha, old_deltaQ, new_deltaQ)
				global_filtered_sum_deltaQ = low_pass_filter(self.alpha, old_global_deltaQ, new_deltaQ)
			set_deltaQ(self.dict_qvalues, state["state"], filtered_sum_deltaQ)
			self.dict_qvalues["deltaQ"] = global_filtered_sum_deltaQ
			"""
			# ----------------------------------------------------------------
			sum_deltaQ += new_deltaQ
			# ----------------------------------------------------------------
		# ----------------------------------------------------------------------------
		return sum_deltaQ
		# ----------------------------------------------------------------------------


	def plan(self):
		"""
		Update qvalues using value iteration algorithm (VI)
		"""
		# ----------------------------------------------------------------------------
		List_variation=[]
		cycle = 0
		while True:
			# ------------------------------------------------------------------------
			#print("Cycle of VI : "+str(cycle))
			convergence_indicator = 0.0
			# ------------------------------------------------------------------------
			for state in self.list_states:
				# Use low pass filter on the sum of deltaQ
				new_sum_deltaQ = self.update_qvalues(state)
				# --------------------------------------------------------------------
				convergence_indicator += new_sum_deltaQ
			# ------------------------------------------------------------------------
			#print("Convergence indicator : "+str(convergence_indicator))
			cycle += 1
			# ------------------------------------------------------------------------
			# Stop VI when convergence
			List_variation.append(convergence_indicator)
			if convergence_indicator < self.epsilon:

				break
			


		return cycle,List_variation,len(self.list_states)

		# ----------------------------------------------------------------------------
	def planPrioSweep(self,this_state,action):


		
		# Fill the buffer with current state/action
		'''
		for state in self.dict_qvalues["values"]:
			if state["state"] == this_state:
				# ---------------------------------------------------------------------
				# loop througth the qvalues
				for qvalue in state["values"]:
					if qvalue["action"] == action:
		'''
		DTime1=datetime.datetime.now()

		# Fill the buffer with a first experience (previsou_state-->this state)
		Time1=0
		Time2=0
		Time3=0
		previous_qvalue = self.dict_qvalues[(str(this_state),"qvals")][int(action)]
		flag = False
		accu = 0.0

		
		# loop to recover reward
		for transition in self.dict_rewards["transitionActions"]:
			if str(transition["state"]) == this_state and transition["action"] == action:
				reward = transition["reward"]
				break
		# ----------------------------------------------------------------
		# loop througth the transitions
		for state in self.dict_transitions["transitionActions"]:
			if str(state["state"]) == this_state:
				for transition in state["transitions"]:
					link = transition["action"]
					prob = transition["prob"]
					linked_state = str(transition["state"])
					#vValue = get_vval(self.dict_qvalues, linked_state)
					vValue = max(self.dict_qvalues[(str(linked_state),"qvals")])
					if link == action:
						flag = True 
						accu += (prob * (reward + self.gamma * vValue))
		if flag==False:
			print("flag" + str(flag))
			sys.exit("Error message")
		RPE=accu-previous_qvalue
		# we add the (current_state,RPE(action)) to bufferRPE only if current_state/action is not already in the buffer
		index,check=check_buffer(this_state,action,self.bufferRPE)
		if check==False:
			if (abs(RPE) > self.replaythreshold):
				self.bufferRPE=np.append(self.bufferRPE, np.array([[this_state],[action],[RPE],[abs(RPE)]]), axis=1)
		else: # potentially updtate buffer if current state is already in bufferRPE
			if (abs(RPE) > float(self.bufferRPE[3,index])):
				self.bufferRPE[:,index]=[[this_state],[action],[RPE],[abs(RPE)]]

		Time1 += (datetime.datetime.now() - DTime1).total_seconds()

		#Replay Buffer
		nbReplayCycle =0
		NbExpReplay=0
		sum_deltaQ=0
		variation_global=list()
		
		while True :
			nbReplayCycle+=1
			#Replay one buffer
			it=0
			sum_deltaQ=0
			while True :
			#Sort Buffer to Replay
				DTime2=datetime.datetime.now()

				if self.bufferRPE.size!=0:
					end=np.size(self.bufferRPE,1)
					self.bufferRPE=self.bufferRPE[:,max(0,(end-self.replaywindow)):end]
					index=np.argsort(self.bufferRPE[3,:])
					self.bufferRPE=self.bufferRPE[:,index[::-1]]
					this_state=self.bufferRPE[0,0]
					action=int(self.bufferRPE[1,0])   #array is uniform...
				else:
					break
				#VI
				"""
				new_plan_time=(datetime.datetime.now()-old_time).total_seconds()
				for state in self.dict_qvalues["values"]:
					if state["state"] == this_state:
						# ---------------------------------------------------------------------
						# loop througth the qvalues
						for qvalue in state["values"]:
							if qvalue["action"] == action:
				"""
				previous_qvalue = self.dict_qvalues[(str(this_state),"qvals")][int(action)]
				flag = False
				accu = 0.0
				# loop to recover reward
				for transition in self.dict_rewards["transitionActions"]:
					if str(transition["state"]) == this_state and transition["action"] == action:
						reward = transition["reward"]
						break
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
								flag = True #Normally always true
								accu += (prob * (reward + self.gamma * vValue))
				self.dict_qvalues[(str(this_state),"qvals")][int(action)] = accu 
				if flag==False:
					print("flag" + str(flag))
					sys.exit("Error message")
				# ----------------------------------------------------------------
				new_deltaQ = abs(self.dict_qvalues[(str(this_state),"qvals")][int(action)] - previous_qvalue)
				"""
				if get_visit(self.dict_qvalues, state["state"]) == 1:
					filtered_sum_deltaQ = new_deltaQ
					global_filtered_sum_deltaQ = new_deltaQ
				
				else:
					old_deltaQ = get_deltaQ(self.dict_qvalues, state["state"])
					old_global_deltaQ = self.dict_qvalues["deltaQ"]
					filtered_sum_deltaQ = low_pass_filter(self.alpha, old_deltaQ, new_deltaQ)
					global_filtered_sum_deltaQ = low_pass_filter(self.alpha, old_global_deltaQ, new_deltaQ)
				set_deltaQ(self.dict_qvalues, state["state"], filtered_sum_deltaQ)
				self.dict_qvalues["deltaQ"] = global_filtered_sum_deltaQ
				# ----------------------------------------------------------------
				"""
				sum_deltaQ += new_deltaQ				
				# ----------------------------------------------------------------
				self.bufferRPE=np.delete(self.bufferRPE,0,axis=1)

				Time2 += (datetime.datetime.now() - DTime2).total_seconds()

				# we search for predecessors of current state
				DTime3=datetime.datetime.now()
				RPE=new_deltaQ
				prede=list()
				for dicStateTransitions in self.dict_transitions["transitionActions"]:
					previous_state = dicStateTransitions["state"]
					for dictActionStateProb in dicStateTransitions["transitions"]:
						if dictActionStateProb["state"] == this_state:
							prede.append([previous_state,dictActionStateProb["action"],dictActionStateProb["prob"]])
				if abs(RPE)>self.replaythreshold: #To improve !
					index,check=check_buffer(prede[0][0],prede[0][1],self.bufferRPE)
					while (len(prede)!=0):
						if check==False:
							self.bufferRPE=np.append(self.bufferRPE, np.array([[prede[0][0]],[prede[0][1]],[RPE*prede[0][2]],[abs(RPE)*prede[0][2]]]), axis=1)
						else:
							if (abs(RPE)*prede[0][2]>float(self.bufferRPE[3,index])):
								self.bufferRPE[:,index]=[[prede[0][0]],[prede[0][1]],[RPE*prede[0][2]],[abs(RPE)*prede[0][2]]]
						prede.pop(0)

				Time3 += (datetime.datetime.now() - DTime3).total_seconds()

				it += 1
				if it>=self.replaywindow or self.bufferRPE.size==0:
					break
			variation_global.append([sum_deltaQ])
			NbExpReplay+=it
			if (self.replay_budget>0 and nbReplayCycle>=self.replay_budget) or self.bufferRPE.size==0 or sum_deltaQ<=self.replaythreshold:
				break
		return nbReplayCycle,variation_global,NbExpReplay,Time1,Time2,Time3

	def update_reward(self, previous_state, action, current_state, reward_obtained):
		"""
		Update the the model of reward
		"""
		# ----------------------------------------------------------------------------
		expected_reward = reward_obtained
		prob = get_transition_prob(self.dict_transitions, previous_state, action, current_state)
		# ----------------------------------------------------------------------------
		for state in self.dict_goals["values"]:
			# ------------------------------------------------------------------------
			# Change potentially the reward of the rewarded state
			if state["state"] == current_state:
				state["reward"] = reward_obtained
			# ------------------------------------------------------------------------
			for link in state["links"]:
				# If the link toward the rewarded state is known so the expected reward is the reward of the rewarded state
				if link[0] == action and link[1] == previous_state:
					expected_reward = state["reward"]
					prob = get_transition_prob(self.dict_transitions, previous_state, action, state["state"])
					break
		# ----------------------------------------------------------------------------
		relative_reward = prob  * expected_reward
		set_reward(self.dict_rewards, previous_state, action, relative_reward)
		# ----------------------------------------------------------------------------


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
		return delta_prob
		# ----------------------------------------------------------------------------


	def _learn(self, previous_state, action, current_state, reward_obtained):
		"""
		Update the contents of the rewards and the transitions model
		"""
		# ----------------------------------------------------------------------------
		# Compute the delta_qval to send at the MC (criterion for the trade-off)
		current_qvalue = self.dict_qvalues[str(current_state),"qvals"][int(action)]
		previous_qvalue = self.dict_qvalues[str(current_state),"qvals"][int(action)]
		# ----------------------------------------------------------------------------
		# // Update the transition model //
		new_delta_prob = self.update_prob(previous_state, action, current_state)
		# ------------------------------------------------------------------------
		# Use low pass filter on delta prob
		old_delta_prob = get_delta_prob(self.dict_delta_prob, previous_state)
		delta_prob = low_pass_filter(self.alpha, old_delta_prob, new_delta_prob)
		set_delta_prob(self.dict_delta_prob, previous_state, delta_prob)
		# ----------------------------------------------------------------------------
		# // Update the reward model //
		self.update_reward(previous_state, action, current_state, reward_obtained)
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
		# (not needful for the model of qvalues because it has already is final size)
		if previous_state not in self.list_states:
			self.list_states.append(previous_state)
			initialize_rewards(self.dict_rewards, previous_state, self.actions_space)
			initialize_transition(self.dict_transitions, previous_state, action, current_state, self.window_size)
		# ----------------------------------------------------------------------------
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
			add_transition(self.dict_transitions, previous_state, action, current_state, self.window_size)
		# If it exist, update the window of transitions
		else:
			set_transitions_window(self.dict_transitions, previous_state, action, current_state, self.window_size)
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
		# ----------------------------------------------------------------------------


	def run(self, action_count, cumulated_reward, reward_obtained, previous_state, decided_action, current_state, do_we_plan): 
		"""
		Run the model-based system
		"""
		# ----------------------------------------------------------------------------
		if self.print == True :
			print("------------------------ MB --------------------------------")
		# ----------------------------------------------------------------------------
		# Update the actioncount and the number of the visits for the previous state FAUX, DEJA FAIT DANS UPDATEDATASTRUCTURE
		#self.dict_duration["actioncount"] = action_count
		#self.dict_delta_prob["actioncount"] = action_count
		#self.dict_qvalues["actioncount"] = action_count
		#set_visit(self.dict_qvalues, previous_state) FAUX, DEJA FAIT DANS UPDATEDATASTRUCTURE
		# ----------------------------------------------------------------------------
		# Update the data structure of the models (states, rewards, transitions, qvalues)
		self.update_data_structure(action_count, previous_state, decided_action, current_state, reward_obtained)
		# ----------------------------------------------------------------------------
		if self.not_learn == False:
		# Update the transition model and the reward model according to the learning.
			self._learn(previous_state, decided_action, current_state, reward_obtained)
		# ----------------------------------------------------------------------------
		# If the expert was choosen to plan, update all the qvalues using planification
		if do_we_plan or reward_obtained != 0:
			# ------------------------------------------------------------------------
			
			# ------------------------------------------------------------------------
			# Run the planification
			old_time = datetime.datetime.now()
			if self.replayMB == True :
				cycle,List_variation,NbExpReplay,Time1,Time2,Time3=self.planPrioSweep(previous_state,int(decided_action))
			else:
				cycle,List_variation,NbrState=self.plan()


			# ------------------------------------------------------------------------
			# Sum the duration of planification with a low pass filter
			current_time = datetime.datetime.now()
			new_plan_time = (current_time - old_time).total_seconds()
			old_plan_time = get_duration(self.dict_duration, current_state)
			filtered_time = low_pass_filter(self.alpha, old_plan_time, new_plan_time)
			set_duration(self.dict_duration, current_state, filtered_time)
			
			if self.logNumber in [1,5,6] :
				if self.replayMB == False :
					self.replay_log.write(" Action_count "+ str(action_count) + " NbExpReplay "+ str(cycle*8*NbrState) +" nbReplayCycle "+str(cycle) +" Time_Nonfiltered "+str(new_plan_time) +
						" Time_filtered "+str(filtered_time) + " state " + str(current_state) + " ListVariation " + str(List_variation)+ "\n")
				else:
					self.replay_log.write(" Action_count "+ str(action_count) + " NbExpReplay "+ str(NbExpReplay) +" nbReplayCycle "+str(cycle) +" Time_Nonfiltered "+str(new_plan_time) + " Time_filtered "+str(filtered_time) +
						 " state " + str(current_state) + " partTime1 " +str(Time1/new_plan_time)+ " partTime2 " + str(Time2/new_plan_time) + " partTime3 " + str(Time3/new_plan_time) + " ListVariation " + str(List_variation) +"\n")

		# ----------------------------------------------------------------------------
		# Choose the next action to do from the current state using soft-max policy.
		decided_action, actions_prob = self._decide(previous_state, current_state)
		# ----------------------------------------------------------------------------
		# Prepare data to return
		plan_time = get_duration(self.dict_duration, current_state)
		#delta_prob = get_delta_prob(self.dict_delta_prob, previous_state)
		#global_deltaQ = self.dict_qvalues["deltaQ"]
		#local_deltaQ = get_deltaQ(self.dict_qvalues, previous_state)
		selection_prob = get_filtered_prob(self.dict_actions_prob, previous_state)
		# ----------------------------------------------------------------------------
		if reward_obtained > 0.0:
			self.not_learn = True
		else:
			self.not_learn = False
		# ----------------------------------------------------------------------------
		# Logs
		#if self.logNumber == True:
		"""
			self.reward_log.write("timecount"+" "+str(action_count)+" "+str(reward_obtained)+" currentTime-nodeStartTime"+" currentTime"+"\n")
			self.states_evolution_log.write("timecount"+" "+str(action_count)+" "+current_state+" "+previous_state+ \
				" currentContactState"+" currentViewState"+" "+str(decided_action)+" currentTime-nodeStartTime"+" currentTime"+"\n")
			self.qvalues_evolution_log.write(",\n"+json.dumps(self.dict_qvalues))
			self.actions_evolution_log.write(",\n"+json.dumps(self.dict_actions_prob))
			self.monitoring_values_log.write(str(action_count)+" "+str(decided_action)+" "+str(plan_time)+" "+str(delta_prob)+" "+str(global_deltaQ)+" "+str(local_deltaQ)+" "+str(selection_prob)+"\n")
		# ----------------------------------------------------------------------------
		# Finish the logging at the end of the simulation (duration or max reward)
		if (action_count == self.duration) or (cumulated_reward == self.max_reward):
			if self.log == True:
				self.qvalues_evolution_log.write('],\n"name" : "Qvalues"\n}')
				self.actions_evolution_log.write('],\n"name" : "Actions"\n}')
		
			# ------------------------------------------------------------------------
			# Build the summary file 
			if self.summary == True:
				if self.directory_flag == True:
					os.chdir("../")
				# --------------------------------------------------------------------
				prefixe = 'v%d_TBMB_'%(VERSION)
				self.summary_log = open(prefixe+'summary_log.dat', 'a')
				self.summary_log.write(str(self.gamma)+" "+str(self.beta)+" "+str(cumulated_reward)+"\n")
		"""

		# ----------------------------------------------------------------------------
		return decided_action, plan_time, 0, 0, 0, selection_prob
		# ----------------------------------------------------------------------------



