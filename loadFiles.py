#!/usr/bin/env python3
#encoding: utf-8

'''
This script is part of the program to simulate a navigation experiment where a robot
has to discover the rewarded state of the arena in which it evolves using a meta-control
decision algorithm : "Dromnelle, R., Renaudo, E., Chetouani, M., Maragos, P., Chatila, R.,
Girard, B., & Khamassi, M. (2022). Reducing Computational Cost During Robot Navigation and 
Human–Robot Interaction with a Human-Inspired Reinforcement Learning Architecture. 
International Journal of Social Robotics, 1-27."

This script allows to load some files required to run the simulation.
'''

__author__ = "Rémi Dromnelle"
__version__ = "2.0"
__maintainer__ = "Rémi Dromnelle"
__email__ = "remi.dromnelle@gmail.com"
__status__ = "Production"

from utility import *

def load_key_states(key_states_file):
	"""
	Load the key states.
	"""
	# -------------------------------------------------------------------------------
	# Load the list of initial states and the rewarded state
	init = list()
	goal = 0
	with open(key_states_file,'r') as file1:
		for line in file1:
			if line.split(" ")[0] == "init":
				init = init + list(map(int, line.split(" ")[1:]))
				init_str = [str(item) for item in init]
			elif line.split(" ")[0] == "goal":
				goal = str(line.split(" ")[1])
				win_reward = int(line.split(" ")[2])
			elif line.split(" ")[0] == "new_goal":
				it_goal = int(line.split(" ")[1])
				new_goal = str(line.split(" ")[2])
				win_reward = int(line.split(" ")[3])
			elif line.split(" ")[0] == "del_trans":
				it_trans = int(line.split(" ")[1])
				path1 = str(line.split(" ")[2])
				path2 = str(line.split(" ")[3])
	# -------------------------------------------------------------------------------
	key_states = {"init_states": init_str, "goal": goal, "reward": win_reward}
	switch_goal = {"iteration": it_goal, "new_goal": new_goal, "reward": win_reward}
	del_trans = {"iteration": it_trans, "path1": path1, "path2": path2}
	# -------------------------------------------------------------------------------
	return key_states, switch_goal, del_trans
	# -------------------------------------------------------------------------------

def load_spaces(spaces_file):
	"""
	Load the state and the action spaces.
	"""
	# -------------------------------------------------------------------------------
	error = False
	# Load the list of initial states and the rewarded state
	with open(spaces_file,'r') as file1:
		for line in file1:
			if line.split(" ")[0] == "state":
				try:
					states_space = int(line.split(" ")[1])
				except:
					print(f"Error : the state space has to be defined by an integer.\n")
					error = True
			elif line.split(" ")[0] == "action":
				try:
					actions_space = int(line.split(" ")[1])
				except:
					print(f"Error : the action space has to be defined by an integer.\n")
					error = True
	# -------------------------------------------------------------------------------
	if error == True:
		quit()
	# -------------------------------------------------------------------------------
	return {"states": states_space, "actions": actions_space}
	# -------------------------------------------------------------------------------

def load_parameters(parameters_file, expert_1, expert_2):
	"""
	Load the parameters.
	"""
	# -------------------------------------------------------------------------------
	MF_expert, MB_expert, DQN_expert, error = (False, False, False, False)
	# Load the parameters and check their number and their type
	with open(parameters_file,'r') as file1:
		for line in file1:
			# -----------------------------------------------------------------------
			if line.split(" ")[0] == "MF":
				MF_expert = True
				if len(line.split(" ")) != 4:
					print(f"Error : at least one parameter are missing for the MF expert.\n")
					error = True
				else:
					MF_param = True
					try:
						alpha_MF = float(line.split(" ")[1])
						gamma_MF = float(line.split(" ")[2])
						beta_MF = float(line.split(" ")[3])
					except:
						print(f"Error : the parameters must be float.\n")
						error = True
			# -----------------------------------------------------------------------
			elif line.split(" ")[0] == "MB":
				MB_expert = True
				if len(line.split(" ")) != 4:
					print(f"Error : at least one parameter are missing for the MB expert.\n")
					error = True
				else:
					MB_param = True
					try:
						alpha_MB = float(line.split(" ")[1])
						gamma_MB = float(line.split(" ")[2])
						beta_MB = float(line.split(" ")[3])
					except:
						print(f"Error : the parameters must be float.\n")
						error = True
			# -----------------------------------------------------------------------
			elif line.split(" ")[0] == "DQN":
				DQN_expert = True
				if len(line.split(" ")) != 4:
					print(f"Error : at least one parameter are missing for the DQN expert.\n")
					error = True
				else:
					DQN_param = True
					try:
						alpha_DQN = float(line.split(" ")[1])
						gamma_DQN = float(line.split(" ")[2])
						beta_DQN = float(line.split(" ")[3])
					except:
						print(f"Error : the parameters must be float.\n")
						error = True	
			# -----------------------------------------------------------------------
			elif line.split(" ")[0] == "MC":
				if len(line.split(" ")) != 3:
					print(f"Error : at least one parameter are missing for the MC expert.\n")
					error = True
				else:
					MC_param = True
					try:
						alpha_MC = float(line.split(" ")[1])
						beta_MC = float(line.split(" ")[2])
					except:
						print(f"Error : the parameters must be float.\n")
						error = True
	# -------------------------------------------------------------------------------
	# Check if the parameters of the chosen experts and le MC exist
	if (expert_1 == "MF" or expert_2 == "MF") and MF_expert == False:
		print(f"Error : parameters of the MF expert are missing.\n")
		error = True
	if (expert_1 == "MB" or expert_2 == "MB") and MB_expert == False:
		print(f"Error : parameters of the MB expert are missing.\n")
		error = True
	if (expert_1 == "DQN" or expert_2 == "DQN") and DQN_expert == False:
		print(f"Error : parameters of the DQN expert are missing.\n")
		error = True
	try:
		MC_param
	except:
		print(f"Error : parameters of the MC are missing.\n")
		error = True
	# -------------------------------------------------------------------------------
	# For undefined parameters, put None
	try:
		MF_param
	except:
		alpha_MF, gamma_MF, beta_MF = (None, None, None)
	try:
		MB_param
	except:
		alpha_MB, gamma_MB, beta_MB = (None, None, None)
	try:
		DQN_param
	except:
		alpha_DQN, gamma_DQN, beta_DQN = (None, None, None)
	# -------------------------------------------------------------------------------
	if error == True:
		quit()
	# -------------------------------------------------------------------------------
	parameters_MF = {"alpha": alpha_MF, "gamma": gamma_MF, "beta": beta_MF}
	parameters_MB = {"alpha": alpha_MB, "gamma": gamma_MB, "beta": beta_MB}
	parameters_DQN = {"alpha": alpha_DQN, "gamma": gamma_DQN, "beta": beta_DQN}
	parameters_MC = {"alpha": alpha_MC, "beta": beta_MC}
	# -------------------------------------------------------------------------------
	#quit()
	# -------------------------------------------------------------------------------
	return parameters_MF, parameters_MB, parameters_DQN, parameters_MC
	# -------------------------------------------------------------------------------



