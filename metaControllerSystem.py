#!/usr/bin/env python3
#encoding: utf-8

'''
This script is part of the program to simulate a navigation experiment where a robot
has to discover the rewarded state of the arena in which it evolves using a meta-control
decision algorithm : "Dromnelle, R., Renaudo, E., Chetouani, M., Maragos, P., Chatila, R.,
Girard, B., & Khamassi, M. (2022). Reducing Computational Cost During Robot Navigation and 
Human–Robot Interaction with a Human-Inspired Reinforcement Learning Architecture. 
International Journal of Social Robotics, 1-27."

With this script, the simulated agent can do meta-control and coordinate different 
behavioral strategies.
'''

__author__ = "Rémi Dromnelle"
__version__ = "1.0"
__maintainer__ = "Rémi Dromnelle"
__email__ = "remi.dromnelle@gmail.com"
__status__ = "Production"

from utility import *

VERSION = 1


class MetaController:
	"""
	This class implements a meta-controller that allows the agent to choose between
	several possible decisions according to some crieria
    """

	def __init__(self, experiment, map_file, initial_variables, boundaries_exp, beta_MC, criterion, coeff_kappa, log):
		"""
		Iinitialise values and models
		"""
		# ---------------------------------------------------------------------------
		self.experiment = experiment
		current_state = initial_variables["current_state"]
		action_count = initial_variables["action_count"]
		initial_reward = initial_variables["reward"]
		initial_duration = initial_variables["plan_time"]
		self.beta_MC = beta_MC
		self.criterion = criterion
		self.coeff_kappa = coeff_kappa
		self.log = log["log"]
		self.epsilon = boundaries_exp["epsilon"]
		# Create a dict that contains the time of planification for each states
		self.dict_duration = dict()
		self.dict_duration["actioncount"] = action_count
		self.dict_duration["values"] = list()
		# initialise the duration dict
		with open(map_file,'r') as file2:
			self.map = json.load(file2)
		for state in self.map["transitionActions"]:
			s = str(state["state"]).replace(" ", "")
			self.dict_duration["values"].append({"state": s, "duration": 0.0})
		# ---------------------------------------------------------------------------
		# Initialise logs
		self.directory_flag = False
		try:
			os.stat("logs")
		except:
			os.mkdir("logs") 
		os.chdir("logs")
		# try:
		# 	os.stat("MC")
		# except:
		# 	os.mkdir("MC") 
		# os.chdir("MC")c
		if self.log == True:
			self.MC_log = open(str(self.criterion)+"_coeff"+str(self.coeff_kappa)+"_exp"+str(self.experiment)+"_log.dat", "w")
			self.MC_log.write(str(action_count)+" "+str(current_state)+" "+str(initial_reward)+" "+str(initial_duration)+" MB 0.5 0.0 0.0 0.0\n")
		# ---------------------------------------------------------------------------
		os.chdir("..")
		# ---------------------------------------------------------------------------


	def entropy_and_time(self, duration, selection_prob):
		"""
		Choose the best action according to a trade-off betwen the time of planning 
		and the quality of learning (entropy of the ditribution probability of the actions). These parameters
		are normalized.
		"""
		# ---------------------------------------------------------------------------
		norm_probs_MF = [prob / sum(selection_prob["MF"]) for prob in selection_prob["MF"]]
		norm_probs_MB = [prob / sum(selection_prob["MB"]) for prob in selection_prob["MB"]]
		# ---------------------------------------------------------------------------
		entropy_probs_MF = shanon_entropy(norm_probs_MF)
		print(f"Entropy MF : {entropy_probs_MF}")
		entropy_probs_MB = shanon_entropy(norm_probs_MB)
		print(f"Entropy MB : {entropy_probs_MB}")
		# ---------------------------------------------------------------------------
		max_entropy = shanon_entropy([1/(len(norm_probs_MF))]*len(norm_probs_MF))
		highest_entropy = max(entropy_probs_MF,entropy_probs_MB)
		mean_entropy = (entropy_probs_MF + entropy_probs_MB) / 2
		# ---------------------------------------------------------------------------
		norm_entropy_MF = entropy_probs_MF / max_entropy
		norm_entropy_MB = entropy_probs_MB / max_entropy
		self.norm_entropy = {"MF": norm_entropy_MF, "MB": norm_entropy_MB}
		print(f"Norm entropy MF : {norm_entropy_MF}")
		print(f"Norm entropy MB : {norm_entropy_MB}")
		# ---------------------------------------------------------------------------
		max_duration = max(duration["MF"],duration["MB"])
		if max_duration == 0.0:
			max_duration = 0.000000000001
		norm_duration_MF = (duration["MF"]) / max_duration
		norm_duration_MB = (duration["MB"]) / max_duration
		# ---------------------------------------------------------------------------
		coeff  = math.exp(-entropy_probs_MF * self.coeff_kappa)
		print(f"Coeff : exp(-{entropy_probs_MF} * {self.coeff_kappa}) = {coeff}")
		# ---------------------------------------------------------------------------
		qval_MF = - (norm_entropy_MF + coeff * norm_duration_MF)
		print(f"Qval MF : - ({norm_entropy_MF} + {coeff} * {norm_duration_MF}) = {qval_MF}")
		qval_MB = - (norm_entropy_MB + coeff * norm_duration_MB)
		print(f"Qval MB : - ({norm_entropy_MB} + {coeff} * {norm_duration_MB}) = {qval_MB}")
		qvalues = {"MF": qval_MF, "MB": qval_MB}
		# ---------------------------------------------------------------------------
		# Soft-max function
		#final_actions_prob = softmax_actions_prob(qvalues, self.beta_MC)
		#expert, final_decision = softmax_decision(final_actions_prob, decisions)
		# ---------------------------------------------------------------------------
		# Arg-max function
		final_actions_prob = softmax_actions_prob(qvalues, self.beta_MC)
		if qval_MF > qval_MB:
			expert = "MF"
			who_plan = {"MF": True, "MB": False, "DQN": False}
		elif qval_MF < qval_MB:
			expert = "MB"
			who_plan = {"MF": False, "MB": True, "DQN": False}
		elif qval_MB == qval_MF:
			randval = np.random.rand()
			if randval >= 0.500000000:
				expert = "MF"
				who_plan = {"MF": True, "MB": False, "DQN": False}
			elif randval < 0.50000000:
				expert = "MB"
				who_plan = {"MF": False, "MB": True, "DQN": False}
		# ---------------------------------------------------------------------------
		return final_actions_prob, who_plan
		# ---------------------------------------------------------------------------


	def entropy_only(self, selection_prob):
		"""
		Choose the best action according to the value of the entropies of the ditribution 
		probability of the actions. These parametersare normalized.
		"""
		# ---------------------------------------------------------------------------
		norm_probs_MF = [prob / sum(selection_prob["MF"]) for prob in selection_prob["MF"]]
		norm_probs_MB = [prob / sum(selection_prob["MB"]) for prob in selection_prob["MB"]]
		# ---------------------------------------------------------------------------
		entropy_probs_MF = shanon_entropy(norm_probs_MF)
		print(f"Entropy MF : {entropy_probs_MF}")
		entropy_probs_MB = shanon_entropy(norm_probs_MB)
		print(f"Entropy MB : {entropy_probs_MB}")
		# ---------------------------------------------------------------------------
		max_entropy = shanon_entropy([1/(len(norm_probs_MF))]*len(norm_probs_MF))
		highest_entropy = max(entropy_probs_MF,entropy_probs_MB)
		mean_entropy = (entropy_probs_MF + entropy_probs_MB) / 2
		# ---------------------------------------------------------------------------
		norm_entropy_MF = entropy_probs_MF / max_entropy
		norm_entropy_MB = entropy_probs_MB / max_entropy
		self.norm_entropy = {"MF": norm_entropy_MF, "MB": norm_entropy_MB}
		print(f"Norm entropy MF : {norm_entropy_MF}")
		print(f"Norm entropy MB : {norm_entropy_MB}")
		# ---------------------------------------------------------------------------
		qval_MF = - (norm_entropy_MF)
		print(f"Qval MF : - {norm_entropy_MF}) = {qval_MF}")
		qval_MB = - (norm_entropy_MB)
		print(f"Qval MB : - {norm_entropy_MB}) = {qval_MB}")
		qvalues = {"MF": qval_MF, "MB": qval_MB}
		# ---------------------------------------------------------------------------
		# Soft-max function
		#final_actions_prob = softmax_actions_prob(qvalues, self.beta_MC)
		#expert, final_decision = softmax_decision(final_actions_prob, decisions)
		# ---------------------------------------------------------------------------
		# Arg-max function
		final_actions_prob = softmax_actions_prob(qvalues, self.beta_MC)
		if qval_MF >= qval_MB:
			expert = "MF"
			who_plan = {"MF": True, "MB": False, "DQN": False}
		else:
			expert = "MB"
			who_plan = {"MF": False, "MB": True, "DQN": False}
		# ---------------------------------------------------------------------------
		return final_actions_prob, who_plan
		# ---------------------------------------------------------------------------


	def only_one(self, selection_prob, expert):
		"""
		Choose the action between those proposed randomly
		"""
		# ---------------------------------------------------------------------------
		norm_probs_MF = [prob / sum(selection_prob["MF"]) for prob in selection_prob["MF"]]
		norm_probs_MB = [prob / sum(selection_prob["MB"]) for prob in selection_prob["MB"]]
		norm_probs_DQN = [prob / sum(selection_prob["DQN"]) for prob in selection_prob["DQN"]]
		entropy_probs_MF = shanon_entropy(norm_probs_MF)
		entropy_probs_MB = shanon_entropy(norm_probs_MB)
		entropy_probs_DQN = shanon_entropy(norm_probs_DQN)
		max_entropy = shanon_entropy([1/(len(norm_probs_MF))]*len(norm_probs_MF))
		norm_entropy_MF = entropy_probs_MF / max_entropy
		norm_entropy_MB = entropy_probs_MB / max_entropy
		norm_entropy_DQN = entropy_probs_DQN / max_entropy
		# ---------------------------------------------------------------------------
		if expert == "MF": 
			who_plan = {"MF": True, "MB": False, "DQN": False}
			final_actions_prob = {"MF": 1, "MB": 0, "DQN": 0}
			self.norm_entropy = {"MF": norm_entropy_MF, "MB": 0.0, "DQN": 0.0}
		elif expert == "MB":
			who_plan = {"MF": False, "MB": True, "DQN": False}
			final_actions_prob = {"MF": 0, "MB": 1, "DQN": 0}
			self.norm_entropy = {"MF": 0.0, "MB": norm_entropy_MB, "DQN": 0.0}
		elif expert == "DQN":
			who_plan = {"MF": False, "MB": False, "DQN": True}
			final_actions_prob = {"MF": 0, "MB": 0, "DQN": 1}
			self.norm_entropy = {"MF": 0.0, "MB": 0.0, "DQN": norm_entropy_DQN}
		# ---------------------------------------------------------------------------
		return final_actions_prob, who_plan
		# ---------------------------------------------------------------------------


	def random(self):
		"""
		Choose the action between those proposed randomly
		"""
		# ---------------------------------------------------------------------------
		#print(f"Decisions : {decisions}")
		randval = np.random.rand()
		if randval <= 0.500000000:
			who_plan = {"MF": True, "MB": False, "DQN": False}
			self.norm_entropy = {"MF": 0.0, "MB": 0.0, "DQN": 0.0}
		elif randval > 0.50000000:
			who_plan = {"MF": False, "MB": True, "DQN": 0.0}
			self.norm_entropy = {"MF": 0.0, "MB": 0.0, "DQN": 0.0}
		# ---------------------------------------------------------------------------
		final_actions_prob = {"MF": 0.5, "MB": 0.5, "DQN": 0.0}
		# ---------------------------------------------------------------------------
		return final_actions_prob, who_plan
		# ---------------------------------------------------------------------------


	def decide(self, plan_time, selection_prob):
		"""
		Choose the action between those proposed according to the choosen criteria
		"""
		# ---------------------------------------------------------------------------
		if self.criterion == "random": 
			final_actions_prob, who_plan = self.random()
		elif self.criterion == "MF_only":
			final_actions_prob, who_plan = self.only_one(selection_prob, "MF") 
		elif self.criterion == "MB_only":
			final_actions_prob, who_plan = self.only_one(selection_prob, "MB")
		elif self.criterion == "DQN_only":
			final_actions_prob, who_plan = self.only_one(selection_prob, "DQN")
		elif self.criterion == "Entropy_and_time":
			final_actions_prob, who_plan = self.entropy_and_time(plan_time, selection_prob)
		elif self.criterion == "Entropy_only":
			final_actions_prob, who_plan = self.entropy_only(selection_prob)
		else:
			sys.exit("This criterion is unknown. Retry with a good one.")
		# ---------------------------------------------------------------------------
		return final_actions_prob, who_plan
		# ---------------------------------------------------------------------------


	def run(self, action_count, reward_obtained, current_state, plan_time, selection_prob):
		"""
		Run the metacontroller
		"""
		# ---------------------------------------------------------------------------
		old_time = datetime.datetime.now()
		# ---------------------------------------------------------------------------
		print("------------------------ MC --------------------------------") 
		print(f"Plan time : {plan_time}")
		print(f"Probability of actions : {selection_prob}")
		#print(f"Repartition of the prefered actions : {prefered_action}")
		# ---------------------------------------------------------------------------
		# Decide betwen the two experts accoring to the choosen criterion
		final_actions_prob, who_plan = self.decide(plan_time, selection_prob)
		# ---------------------------------------------------------------------------
		# Sum the duration of planification with a low pass filter
		current_time = datetime.datetime.now()
		new_plan_time = (current_time - old_time).total_seconds()
		old_plan_time = get_duration(self.dict_duration, current_state)
		filtered_time = low_pass_filter(0.6, old_plan_time, new_plan_time)
		set_duration(self.dict_duration, current_state, filtered_time)
		# ---------------------------------------------------------------------------
		# Register the winner
		if who_plan["MF"] == True:
			winner = "MF"
		elif who_plan["MB"] == True:
			winner = "MB"
		elif who_plan["DQN"] == True:
			winner = "DQN"
		# ---------------------------------------------------------------------------
		# Logs
		if self.log == True:
			# -----------------------------------------------------------------------
			time = 0.0
			if who_plan["MF"] == True:
				time += plan_time["MF"]
			if who_plan["MB"] == True:
				time += plan_time["MB"]
			if who_plan["DQN"] == True:
				time += plan_time["DQN"]
			# -----------------------------------------------------------------------
			if who_plan["MF"] == True:
				self.MC_log.write(f"{action_count} {current_state} {reward_obtained} {time} MF {final_actions_prob['MF']} {self.norm_entropy['MF']} {self.norm_entropy['MB']} {filtered_time}\n")
			elif who_plan["MB"] == True:
				self.MC_log.write(f"{action_count} {current_state} {reward_obtained} {time} MB {final_actions_prob['MB']} {self.norm_entropy['MF']} {self.norm_entropy['MB']} {filtered_time}\n")
			elif who_plan["DQN"] == True:
				self.MC_log.write(f"{action_count} {current_state} {reward_obtained} {time} DQN {final_actions_prob['DQN']} {self.norm_entropy['DQN']} {self.norm_entropy['DQN']} {filtered_time}\n")
		# ---------------------------------------------------------------------------
		return winner, who_plan
		# ---------------------------------------------------------------------------

