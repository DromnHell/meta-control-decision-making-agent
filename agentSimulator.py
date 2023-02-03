#!/usr/bin/env python3
#encoding: utf-8

'''
This script is part of the program to simulate a navigation experiment where a robot
has to discover the rewarded state of the arena in which it evolves using a meta-control
decision algorithm : "Dromnelle, R., Renaudo, E., Chetouani, M., Maragos, P., Chatila, R.,
Girard, B., & Khamassi, M. (2022). Reducing Computational Cost During Robot Navigation and 
Human–Robot Interaction with a Human-Inspired Reinforcement Learning Architecture. 
International Journal of Social Robotics, 1-27."

This script is the core of the program.
'''

__author__ = "Rémi Dromnelle"
__version__ = "1.0"
__maintainer__ = "Rémi Dromnelle"
__email__ = "remi.dromnelle@gmail.com"
__status__ = "Production"

# -----------------------------------------------------------------------------------
# IMPORT
# -----------------------------------------------------------------------------------
from optparse import OptionParser
import re
import sys
from metaControllerSystem import *
from modelFreeSystem import *
from modelBasedSystem import *
from DQNSystem import *
from manageEnvironment import *
# -----------------------------------------------------------------------------------

def run_simulation(map_file, model_free_agent, model_based_agent, meta_controller, spaces, boundaries_exp, changes_exp, initial_variables):
	"""
	Run the simulation
	"""
	# -------------------------------------------------------------------------------
	# Initialise the environment of the agent
	key_states, switch_goal, add_wall = initialize_environment(key_states_file)
	final_state = {"state": key_states["goal"], "reward": key_states["reward"]}
	init_states = key_states["init_states"]
	# -------------------------------------------------------------------------------
	# Initialize parameters and variables for the loop of simulation
	action_count = initial_variables["action_count"] + 1
	final_decision = initial_variables["decided_action"]
	previous_state = initial_variables["previous_state"]
	current_state = initial_variables["current_state"]
	reward_obtained = initial_variables["reward"]
	duration = boundaries_exp["duration"]
	max_reward = boundaries_exp["max_reward"]
	who_plan = {current_state: {"MF": True, "MB": True, "DQN": True}}
	# -------------------------------------------------------------------------------
	cumulated_reward = 0
	path1 = 0
	path2 = 0
	# -------------------------------------------------------------------------------
	# Run the simulation until the boudaries experiment are not reached
	while (action_count <= duration) and (cumulated_reward <= max_reward):
		print("\n")
		print("------------------------------------------------------------")
		print(f"It {action_count}")
		print(f"Previous state : {previous_state}")
		print(f"Action done : {final_decision}")
		print(f"Current state : {current_state}")
		print(f"Reward obtained : {reward_obtained}")
		# ---------------------------------------------------------------------------
		# Update potentially the environment
		if changes_exp["new_goal"] == True and action_count == switch_goal["iteration"]:
			final_state = {"state": switch_goal["new_goal"], "reward": switch_goal["reward"]}
			map_file = map_file+"_afterSwitch"
			print(f"The rewarded state has changed ! Now the state {final_state['state']} gives the reward.")
		if changes_exp["add_wall"] == True and action_count == add_wall["iteration"]:
			if path1 >= path2:
				map_file = f"{map_file}_wall{add_wall['path1']}"
				print(f"A wall has been added between the states {add_wall['path1']}")
			else:
				map_file = f"{map_file}_wall{add_wall['path2']}"
				print(f"A wall has been added between the states {add_wall['path2']}")
		# ---------------------------------------------------------------------------
		# Get the probabilities of selection of the two expert for the current state accoring to the q-values
		selection_prob_MF = model_free_agent.get_actions_prob(current_state)
		selection_prob_MB = model_based_agent.get_actions_prob(current_state)
		#selection_prob_DQN = DQN_agent.get_actions_prob(current_state)
		selection_prob = {"MF": selection_prob_MF, "MB": selection_prob_MB, "DQN": [1/spaces["actions"]]*spaces["actions"]}
		# Get the the of planification of the two expert for the current state according to the previous one
		plan_time_MF = model_free_agent.get_plan_time(current_state)
		plan_time_MB = model_based_agent.get_plan_time(current_state)
		#plan_time_DQN = DQN_agent.get_plan_time(current_state)
		plan_time = {"MF": plan_time_MF, "MB": plan_time_MB, "DQN": 0.0000000}
		# Choose which expert to inhibit with the MC using a criterion of coordination
		first_visit = False
		if current_state not in who_plan.keys():
			first_visit = True
		winner, who_plan[current_state] = meta_controller.run(action_count, reward_obtained, current_state, plan_time, selection_prob)
		if first_visit == True:
			who_plan[current_state] = {"MF": True, "MB": True, "DQN": True}
		# ---------------------------------------------------------------------------
		# Get the decision, the deltaQ and the planning time of each expert
		decision_MF = model_free_agent.run(action_count, cumulated_reward, reward_obtained, previous_state, final_decision, current_state, who_plan[current_state]["MF"])
		decision_MB = model_based_agent.run(action_count, cumulated_reward, reward_obtained, previous_state, final_decision, current_state, who_plan[current_state]["MB"])
		#decision_DQN = DQN_agent.run(action_count, cumulated_reward, reward_obtained, previous_state, final_decision, current_state, who_plan[current_state]["DQN"])
		decision_DQN = 0
		decisions = {"MF": decision_MF, "MB": decision_MB, "DQN": decision_DQN}
		print("-------------------------------------------------------------")
		# ---------------------------------------------------------------------------
		if winner == "MF":
			print("Winner expert : Hab")
			final_decision = decisions["MF"]
		elif winner == "MB":
			print("Winner expert : GD")		
			final_decision = decisions["MB"]
		elif winner == "DQN":
			print("Winner expert : DQN")		
			final_decision = decisions["DQN"]
		print(f"Final action = {final_decision}")
		# ---------------------------------------------------------------------------
		# The previous state is now the old current state
		previous_state = current_state 
		# ---------------------------------------------------------------------------
		# Count the number of passages in each coridor
		if previous_state == "7":
			path2 += 1
		elif previous_state == "19":
			path1 += 1
		# ---------------------------------------------------------------------------
		# Simulate the effect of the robot's final decision on the environement and find the new current state
		reward_obtained, current_state = update_robot_position(init_states, final_state, map_file, previous_state, final_decision)
		# ---------------------------------------------------------------------------
		# Reset the reward obtained if the map changes
		if action_count == switch_goal["iteration"] or action_count == add_wall["iteration"]:
			reward_obtained = 0
		# ---------------------------------------------------------------------------
		# Update cumulated reward and counter of actions
		cumulated_reward += reward_obtained
		if reward_obtained == 1:
			print("WIN !")
			print(f"Cumulated reward = {cumulated_reward-1} + 1 = {cumulated_reward}")
		else:
			print(f"Cumulated reward = {cumulated_reward}")
		action_count += 1
		# ---------------------------------------------------------------------------
		# If no reward is obtained during the 200 first iteration, reset the run
		#if action_count == 200 and cumulated_reward == 0:
		#	action_count = 0
		#	cumulated_reward = 0
		#	previous_state = "0"
		#	final_decision = 0
		#	current_state = "0"
		#	who_plan = {current_state: {"MF": True, "MB": True, "DQN": True}}
		#	meta_controller_system = MetaController(experiment, map_file, initial_variables, boundaries_exp, beta_MC, criterion, coeff_kappa, log)
			#DQN_agent = DQN(experiment, map_file, initial_variables, action_space, state_space, boundaries_exp, parameters_DQN, log)
		# ---------------------------------------------------------------------------


def parse_parameters(parameters_file):
	"""
	Parse the file that contains the parameters
	"""
	# -------------------------------------------------------------------------------
	with open(parameters_file,'r') as file1:
		for line in file1:
			if line.split(" ")[0] == "MF":
				alpha_MF = float(line.split(" ")[1])
				gamma_MF = float(line.split(" ")[2])
				beta_MF = int(line.split(" ")[3])
			elif line.split(" ")[0] == "MB":
				gamma_MB = float(line.split(" ")[1])
				beta_MB = int(line.split(" ")[2])
			elif line.split(" ")[0] == "MC":
				beta_MC = int(line.split(" ")[1])
	# -------------------------------------------------------------------------------
	parameters_MF = {"alpha": alpha_MF, "gamma": gamma_MF, "beta": beta_MF}
	parameters_MB = {"alpha": alpha_MF, "gamma": gamma_MB, "beta": beta_MB}
	parameters_DQN = {"alpha": alpha_MF, "gamma": gamma_MB, "beta": beta_MB}
	# -------------------------------------------------------------------------------
	return parameters_MF, parameters_MB, parameters_DQN, beta_MC
	# -------------------------------------------------------------------------------


def manage_arguments():
	"""
	Manage the arguments of the script
	"""
	# -------------------------------------------------------------------------------
	usage = "usage: main.py [options] [the id of the experiment] [the file that contains the map of the environment, in the form of a transition model] [the file that contains the key states] [the file that contains the states and the actions spaces] [the file that contains the parameters of each expert]"
	parser = OptionParser(usage)
	parser.add_option("-c", "--criterion", action = "store", type = "string", dest = "criterion", help = "This option is the criterion used for the trade-off betwen the two experts", default = "random")
	parser.add_option("-k", "--coeff_kappa", action = "store", type = "float", dest = "coeff_kappa", help = "This option is the coefficient use by the kappa parameter to weight the time", default = 1.0)
	parser.add_option("-r", "--max_reward", action = "store", type = "int", dest = "max_reward", help = "This option is the maximum cumulated reward that the agent will reach before to stop.", default = 10000)
	parser.add_option("-d", "--duration", action = "store", type = "int", dest = "duration", help = "This option is the maximum duration during which the agent will work.", default = 100000)
	parser.add_option("-w", "--window_size", action = "store", type = "int", dest = "window_size", help = "This option is the size of the window of transitions memorized by the agent.", default = 10)
	parser.add_option("-n", "--new_goal", action = "store_true", dest = "new_goal", help = "This option says if the goal will change during the experiment", default = False)
	parser.add_option("-a", "--add_wall", action = "store_true", dest = "add_wall", help = "This option says if a wall is added during the experiment", default = False)
	parser.add_option("-l", "--log", action = "store_true", dest = "log", help =  "This option permit to log the data.", default = False)
	parser.add_option("-s", "--summary", action = "store_true", dest = "summary", help = "This option permit to make a summary of the data in one file to the grid search.", default = False)
	# -------------------------------------------------------------------------------
	(options, args) = parser.parse_args()
	# -------------------------------------------------------------------------------
	if len(args) != 5:
		parser.error("wrong number of arguments")
	else:
		experiment = sys.argv[1]
		map_file = sys.argv[2]
		key_states_file = sys.argv[3]
		spaces_file = sys.argv[4]
		parameters_file = sys.argv[5]
	# -------------------------------------------------------------------------------
	return(experiment, map_file, key_states_file, spaces_file, parameters_file, options)
	# -------------------------------------------------------------------------------


if __name__ == "__main__":                          
	# -------------------------------------------------------------------------------
	# Manage the arguments et parse the parameters
	experiment, map_file, key_states_file, spaces_file, parameters_file, options = manage_arguments()
	parameters_MF, parameters_MB, parameters_DQN, beta_MC = parse_parameters(parameters_file)
	# -------------------------------------------------------------------------------
	# Initialize and gather the variables and the constants
	spaces = load_spaces(spaces_file)
	boundaries_exp = {"max_reward": options.max_reward, "duration": options.duration, "window_size": options.window_size, "epsilon": 0.01}
	changes_exp = {"new_goal": options.new_goal, "add_wall": options.add_wall}
	log = {"log": options.log, "summary": options.summary}
	criterion = options.criterion
	coeff_kappa = options.coeff_kappa
	initial_variables = {"action_count": 0, "decided_action": 0, "actions_prob": 1/spaces["actions"], "previous_state": "0", "current_state": "0", \
	"qvalue": 1, "delta": 0.0, "plan_time": 0.0, "reward": 0}
	# -------------------------------------------------------------------------------
	# Create instances for the 3 systems used by the virtual agent
	meta_controller = MetaController(experiment, map_file, initial_variables, boundaries_exp, beta_MC, criterion, coeff_kappa, log)
	model_free_agent = ModelFree(experiment, map_file, initial_variables, spaces["actions"], boundaries_exp, parameters_MF, log)
	model_based_agent = ModelBased(experiment, map_file, initial_variables, spaces["actions"], boundaries_exp, parameters_MB, log)
	#DQN_agent = DQN(experiment, map_file, initial_variables, action_space, state_space, boundaries_exp, parameters_DQN, log)
	# -------------------------------------------------------------------------------
	# Run the simulation
	run_simulation(map_file, model_free_agent, model_based_agent, meta_controller, spaces, boundaries_exp, changes_exp, initial_variables)
	# -------------------------------------------------------------------------------
	



