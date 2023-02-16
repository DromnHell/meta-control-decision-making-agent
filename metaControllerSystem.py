#!/usr/bin/env python3
#encoding: utf-8

'''
This script is part of the program to simulate a navigation experiment where a robot
has to discover the rewarded state of the arena in which it evolves using a meta-control
decision algorithm : "Dromnelle, R., Renaudo, E., Chetouani, M., Maragos, P., Chatila, R.,
Girard, B., & Khamassi, M. (2022). Reducing Computational Cost During Robot Navigation and 
Human–Robot Interaction with a Human-Inspired Reinforcement Learning Architecture. 
International Journal of Social Robotics, 1-27."

With this script, the simulated agent can do meta-control to coordinate differents 
behavioral strategies.
'''

__author__ = "Rémi Dromnelle"
__version__ = "2.0"
__maintainer__ = "Rémi Dromnelle"
__email__ = "remi.dromnelle@gmail.com"
__status__ = "Production"

from utility import *

class MetaController:
	"""
	This class implements a meta-controller tha allows the agent to choose which expert 
	will be allowed to schedule at the next iteration.
    """

	def __init__(self, experiment, env_file, initial_variables, action_space, boundaries_exp, parameters_MC, experts, criterion, coeff_kappa, log):
		"""
		Iinitialize values and models
		"""
		# ---------------------------------------------------------------------------
		self.experiment = experiment
		current_state = initial_variables["current_state"]
		action_count = initial_variables["action_count"]
		initial_reward = initial_variables["reward"]
		initial_duration = initial_variables["plan_time"]
		self.action_space = action_space
		self.beta_MC = parameters_MC["beta"]
		self.alpha = parameters_MC["alpha"]
		self.criterion = criterion
		self.coeff_kappa = coeff_kappa
		self.log = log["log"]
		self.epsilon = boundaries_exp["epsilon"]
		# Create a dict that contains the time of planification for each states
		self.dict_duration = dict()
		self.dict_duration["actioncount"] = action_count
		self.dict_duration["values"] = list()
		# initialize the duration dict
		with open(env_file,'r') as file2:
			self.env = json.load(file2)
		for state in self.env["transitionActions"]:
			s = str(state["state"]).replace(" ", "")
			self.dict_duration["values"].append({"state": s, "duration": 0.0})
		# ---------------------------------------------------------------------------
		# initialize logs
		self.directory_flag = False
		try:
			os.stat("logs")
		except:
			os.mkdir("logs") 
		os.chdir("logs")
		if self.log == True:
			self.MC_log = open(f"exp{self.experiment}_{experts[0]}vs{experts[1]}_{self.criterion}_coeff{self.coeff_kappa}_log.dat", "w")
			self.MC_log.write(f"{action_count} {current_state} {initial_reward} {initial_duration} {experts[0]} 0.5 0.0 0.0 0.0\n")
		# ---------------------------------------------------------------------------
		os.chdir("..")
		# ---------------------------------------------------------------------------


	def argmax_function(self, experts_id, qvalues):
		"""
		"""
		# ---------------------------------------------------------------------------
		final_actions_prob = softmax_actions_prob(qvalues, self.beta_MC)
		who_plan = dict()
		if qvalues[experts_id[0]] == qvalues[experts_id[1]]:
			winner = random.choice(experts_id)
			for expert in experts_id:
				if expert == winner:
					who_plan[expert] = True
				else:
					who_plan[expert] = False
		else:
			keymax = max(qvalues, key = lambda x: qvalues[x])
			for expert in experts_id:
				if keymax == expert:
					who_plan[expert] = True
				else:
					who_plan[expert] = False
		# ---------------------------------------------------------------------------
		return final_actions_prob, who_plan
		# ---------------------------------------------------------------------------


	def compute_entropy(self, experts_id, selection_prob):
		"""
		Compute the normalized shanon entropy
		"""
		# ---------------------------------------------------------------------------
		entropy_probs = list()
		for it, probs in enumerate(selection_prob):
			norm_prob = [prob / sum(probs) for prob in probs]
			entropy = shanon_entropy(norm_prob)
			entropy_probs.append(entropy)
			#print(f"Entropy {experts_id[it]} : {entropy}")
		# ---------------------------------------------------------------------------
		max_entropy = shanon_entropy([1/(self.action_space)]*self.action_space)
		mean_entropy = sum(entropy_probs) / 2
		# ---------------------------------------------------------------------------
		self.norm_entropy = dict()
		for it, prob in enumerate(entropy_probs):
			norm_entropy = prob / max_entropy
			self.norm_entropy[experts_id[it]] = norm_entropy
			#print(f"Norm entropy {experts_id[it]} : {norm_entropy}")
		print(f"Norm entropies : {self.norm_entropy}")
		# ---------------------------------------------------------------------------
		return entropy_probs, mean_entropy
		# ---------------------------------------------------------------------------


	def entropy(self, experts_id, selection_prob):
		"""
		Determine which expert will plan raccording to the value of the entropies of 
		the probabilities ditribution of the actions.
		"""
		# ---------------------------------------------------------------------------
		# Compute the quality of learning (entropy)
		mean_entropy, entropy_probs = self.compute_entropy(experts_id, selection_prob)
		# ---------------------------------------------------------------------------
		qvalues = dict()
		for key, value in self.norm_entropy.items():
			qvalues[key] = - value
		# ---------------------------------------------------------------------------
		# Soft-max function
		#final_actions_prob = softmax_actions_prob(qvalues, self.beta_MC)
		#expert, final_decision = softmax_decision(final_actions_prob, decisions)
		# ---------------------------------------------------------------------------
		# Arg-max function
		final_actions_prob, who_plan = self.argmax_function(experts_id, qvalues)
		# ---------------------------------------------------------------------------
		return final_actions_prob, who_plan
		# ---------------------------------------------------------------------------


	def entropy_and_cost(self, experts_id, duration, selection_prob):
		"""
		Determine which expert will plan according to a trade-off betwen the cost 
		(time of planning) and the quality of learning (entropies of the probabilities 
		ditribution of the actions).
		"""
		# ---------------------------------------------------------------------------
		# Compute the quality of learning (entropy)
		entropy_probs, mean_entropy = self.compute_entropy(experts_id, selection_prob)
		# ---------------------------------------------------------------------------
		# Compute the cost (duration)
		max_duration = max(duration)
		if max_duration == 0.0:
			max_duration = 0.000000000001
		norm_duration = dict()
		for it, d in enumerate(duration):
			norm_duration[experts_id[it]] = d / max_duration
		print(f"Norm durations : {norm_duration}")
		# ---------------------------------------------------------------------------
		# Compute the trade-off
		entropy = mean_entropy
		coeff = math.exp(-entropy * self.coeff_kappa)
		# If there is a MF expert, its entropy is choosen instead of the mean entropy
		for it, expert in enumerate(experts_id):
			if expert == "MF":
				entropy = entropy_probs[it]
				coeff = math.exp(-entropy * self.coeff_kappa)
				break
		print(f"Coeff : exp(-{entropy} * {self.coeff_kappa}) = {coeff}")
		# ---------------------------------------------------------------------------
		qvalues = dict()
		for key, value in self.norm_entropy.items():
			qval = - (value + coeff * norm_duration[key])
			qvalues[key] = qval
			#print(f"Qval {key} : - ({value} + {coeff} * {norm_duration[key]}) = {qval}")
		# ---------------------------------------------------------------------------
		# Soft-max function
		#final_actions_prob = softmax_actions_prob(qvalues, self.beta_MC)
		#expert, final_decision = softmax_decision(final_actions_prob, decisions)
		# ---------------------------------------------------------------------------
		# Arg-max function
		final_actions_prob, who_plan = self.argmax_function(experts_id, qvalues)
		# ---------------------------------------------------------------------------
		return final_actions_prob, who_plan
		# ---------------------------------------------------------------------------


	def no_coordination(self, experts_id, selection_prob):
		"""
		The agent used only one expert, so there is no coordination and only one expert can plan.
		But we need this function to recorde the entropie values anyway.
		"""
		# ---------------------------------------------------------------------------
		entropy_probs = list()
		for it, probs in enumerate(selection_prob):
			if experts_id[it] != None:
				norm_prob = [prob / sum(probs) for prob in probs]
				entropy_probs.append(shanon_entropy(norm_prob))
			else:
				entropy_probs.append(None)
		# ---------------------------------------------------------------------------
		max_entropy = shanon_entropy([1/(self.action_space)]*self.action_space)
		# ---------------------------------------------------------------------------
		norm_entropy = list()
		for prob in entropy_probs:
			if prob != None:
				norm_entropy.append(prob / max_entropy)
			else:
				norm_entropy.append(None)
		# ---------------------------------------------------------------------------
		if experts_id[0] != None:
			who_plan = {experts_id[0]: True, experts_id[1]: None}
			final_actions_prob = {experts_id[0]: 1, experts_id[1]: None}
			self.norm_entropy = {experts_id[0]: norm_entropy[0], experts_id[1]: None}
		else:
			who_plan = {experts_id[0]: None, experts_id[1]: True}
			final_actions_prob = {experts_id[0]: None, experts_id[1]: 1}
			self.norm_entropy = {experts_id[0]: None, experts_id[1]: norm_entropy[1]}
		# ---------------------------------------------------------------------------
		return final_actions_prob, who_plan
		# ---------------------------------------------------------------------------


	def random(self, experts_id):
		"""
		Determine which expert will plan randomly.
		"""
		# ---------------------------------------------------------------------------
		#print(f"Decisions : {decisions}")
		randval = np.random.rand()
		if randval <= 0.500000000:
			who_plan = {experts_id[0]: True, experts_id[1]: False}
			self.norm_entropy = {experts_id[0]: 0.0, experts_id[1]: 0.0}
		elif randval > 0.50000000:
			who_plan = {experts_id[0]: False, experts_id[1]: True}
			self.norm_entropy = {experts_id[0]: 0.0, experts_id[1]: 0.0}
		# ---------------------------------------------------------------------------
		final_actions_prob = {experts_id[0]: 0.5, experts_id[1]: 0.5}
		# ---------------------------------------------------------------------------
		return final_actions_prob, who_plan
		# ---------------------------------------------------------------------------


	def decide(self, experts_id, plan_time, selection_prob):
		"""
		Determine which expert will plan based on the action selection probabilities
		and the coordination criterion
		"""
		# ---------------------------------------------------------------------------
		if self.criterion == "random": 
			final_actions_prob, who_plan = self.random(experts_id)
		elif self.criterion == "no_coordination":
			final_actions_prob, who_plan = self.no_coordination(experts_id, selection_prob) 
		elif self.criterion == "entropy_and_cost":
			final_actions_prob, who_plan = self.entropy_and_cost(experts_id, plan_time, selection_prob)
		elif self.criterion == "entropy":
			final_actions_prob, who_plan = self.entropy(experts_id, selection_prob)
		else:
			sys.exit("This criterion is unknown. Retry with a good one.")
		# ---------------------------------------------------------------------------
		return final_actions_prob, who_plan
		# ---------------------------------------------------------------------------


	def run(self, action_count, reward_obtained, current_state, experts_id, plan_time, selection_prob):
		"""
		Run the meta-controller
		"""
		# ---------------------------------------------------------------------------
		old_time = datetime.datetime.now()
		# ---------------------------------------------------------------------------
		print("------------------------ MC --------------------------------") 
		print(f"Plan time : {experts_id[0]} -> {plan_time[0]}, {experts_id[1]} -> {plan_time[1]}")
		print(f"Probabilities of actions : {experts_id[0]} -> {selection_prob[0]}, {experts_id[1]} -> {selection_prob[1]}")
		#print(f"Repartition of the prefered actions : {prefered_action}")
		# ---------------------------------------------------------------------------
		# Decide betwen the two experts accoring to the choosen criterion
		final_actions_prob, who_plan = self.decide(experts_id, plan_time, selection_prob)
		# ---------------------------------------------------------------------------
		# Sum the duration of planification with a low pass filter
		current_time = datetime.datetime.now()
		new_plan_time = (current_time - old_time).total_seconds()
		old_plan_time = get_duration(self.dict_duration, current_state)
		filtered_time = low_pass_filter(self.alpha, old_plan_time, new_plan_time)
		set_duration(self.dict_duration, current_state, filtered_time)
		# ---------------------------------------------------------------------------
		# Register the winner
		for key, value in who_plan.items():
			if value == True:
				winner = key
		# ---------------------------------------------------------------------------
		# If logs are recorded 
		if self.log == True:
			# -----------------------------------------------------------------------
			# Count cumulated time
			time = 0.0
			for key, value in who_plan.items():
				if value == True:
					time += plan_time[experts_id.index(key)]
			# -----------------------------------------------------------------------
			# Write log
			for key, value in who_plan.items():
				if value == True:
					self.MC_log.write(f"{action_count} {current_state} {reward_obtained} {time} {key} {final_actions_prob[key]} {self.norm_entropy[experts_id[0]]} {self.norm_entropy[experts_id[1]]} {filtered_time}\n")
		# ---------------------------------------------------------------------------
		return winner, who_plan
		# ---------------------------------------------------------------------------

