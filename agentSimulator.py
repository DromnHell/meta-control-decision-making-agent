#!/usr/bin/env python3
#encoding: utf-8

'''
This script is part of the program to simulate a navigation experiment where a robot
has to discover the rewarded state of the arena in which it evolves using a meta-control
decision algorithm : "Dromnelle, R., Renaudo, E., Chetouani, M., Maragos, P., Chatila, R.,
Girard, B., & Khamassi, M. (2022). Reducing Computational Cost During Robot Navigation and 
Human–Robot Interaction with a Human-Inspired Reinforcement Learning Architecture. 
International Journal of Social Robotics, 1-27."

This script is the core of the program. It initializes the agent and run the simulation.
'''

__author__ = "Rémi Dromnelle"
__version__ = "2.0"
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
from modelFreeRL import *
from modelBasedRL import *
from DQN import *
from loadFiles import *
# -----------------------------------------------------------------------------------

KNOWN_EXPERTS = ["MF", "MB", "DQN", "None"]
KNOWN_CRITERIA = ["random", "entropy", "entropy_and_cost"]

# -----------------------------------------------------------------------------------

def update_robot_position(init_states, final_state, map_file, start, action):
	"""
	Simulate the effect of the robot's decision on the environement, that is to say,
	the identify of new state reaches after do the action in the previous state and 
	the reward obtains in this new state.
	"""
	# -------------------------------------------------------------------------------
	with open(map_file,'r') as file1:
		# ---------------------------------------------------------------------------
		arena = json.load(file1)
		# If the previous state is the rewarded state, the current state
		# is randomly choose in the list of initial states.
		if start == final_state["state"]:
			arrival = np.random.choice(init_states)
		# ---------------------------------------------------------------------------
		# Else, the current state is choose according to the map of the
		# environement
		else :
			tab_act = list()
			for state in arena["transitionActions"]:
				if str(state["state"]) == start:
					for transition in state["transitions"]:
						if transition["action"] == action:
							l = [str(transition["state"])]
							l = l*transition["prob"]
							tab_act.extend(l)
			arrival = np.random.choice(tab_act)
		# ---------------------------------------------------------------------------
		# If the arrival state is the rewarded state, reward = 1
		if arrival == final_state["state"]:
			reward = final_state["reward"]
		else:
			reward = 0
		# ---------------------------------------------------------------------------
	return reward, arrival
	# -------------------------------------------------------------------------------


def run_simulation(map_file, meta_controller, experts_to_run, spaces, boundaries_exp, changes_exp, initial_variables):
	"""
	Run the simulation.
	"""
	# -------------------------------------------------------------------------------
	# Initialise the key states of the agent's
	key_states, switch_goal, add_wall = load_key_states(key_states_file)
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
	# -------------------------------------------------------------------------------
	# Get the ID of the experts and initialize who_plan dictionnary 
	experts_id = list()
	who_plan = {current_state : {}}
	for expert in experts_to_run:
		if expert != None:
			experts_id.append(expert.ID)
			who_plan[current_state][expert.ID] = True
		else:
			experts_id.append(None)
			who_plan[current_state][None] = None
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
		# Get the probabilities of selection of the experts for the current state accoring to the q-values
		selection_prob = list()
		for expert in experts_to_run:
			if expert != None:
				selection_prob.append(expert.get_actions_prob(current_state))
			else:
				selection_prob.append(None)
		# Get the the of planification of the experts for the current state according to the previous one
		plan_time = list()
		for expert in experts_to_run:
			if expert != None:
				plan_time.append(expert.get_plan_time(current_state))
			else:
				plan_time.append(None)
		# ---------------------------------------------------------------------------
		# Choose which expert to inhibit with the MC using a criterion of coordination
		first_visit = False
		if current_state not in who_plan.keys():
			first_visit = True
		winner, who_plan[current_state] = meta_controller.run(action_count, reward_obtained, current_state, experts_id, plan_time, selection_prob)
		if first_visit == True:
			who_plan[current_state] = {experts_id[0]: True, experts_id[1]: True}
		# ---------------------------------------------------------------------------
		# Get the decision, the deltaQ and the planning time of each expert
		decisions = list()
		for expert in experts_to_run:
			if expert != None:
				decisions.append(expert.run(action_count, cumulated_reward, reward_obtained, previous_state, final_decision, current_state, who_plan[current_state][expert.ID]))
			else:
				decisions.append(None)
		print("-------------------------------------------------------------")
		# ---------------------------------------------------------------------------
		for it, expert in enumerate(experts_id):
			if expert == winner:
				print(f"Winner expert : {expert}")
				final_decision = decisions[it]
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
		#	who_plan[current_state] = {experts_id[0]: True, experts_id[1]: True}
		#	meta_controller_system = MetaController(experiment, map_file, initial_variables, boundaries_exp, parameters_MC, criterion, coeff_kappa, log)
		# ---------------------------------------------------------------------------
		

def manage_arguments():
	"""
	Manage the arguments of the script.
	"""
	# -------------------------------------------------------------------------------
	usage = "usage: agentSimulator.py [options] [the id of the experiment] [the id of the first expert] [the id of the second expert] [the file that contains the map of the environment, in the form of a transition model] [the file that contains the key states] [the file that contains the states and the actions spaces] [the file that contains the parameters of each expert]"
	parser = OptionParser(usage)
	# -------------------------------------------------------------------------------
	# OPTIONS
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
	# ARGUMENTS
	if len(args) != 7:
		parser.error("Wrong number of arguments")
	else:
		experiment = sys.argv[1]
		expert_1 = sys.argv[2]
		expert_2 = sys.argv[3]
		map_file = sys.argv[4]
		key_states_file = sys.argv[5]
		spaces_file = sys.argv[6]
		parameters_file = sys.argv[7]
	# -------------------------------------------------------------------------------
	# ERRORS
	error = False
	# Check if expert_1 exists. If not, quit the program.
	try:
		KNOWN_EXPERTS.index(expert_1)
	except ValueError:
		print(f"Error : '{expert_1}' is not a known expert. The known experts are : {KNOWN_EXPERTS}")
		error = True
	# Check if expert_2 exists. If not, quit the program.
	try:
		KNOWN_EXPERTS.index(expert_2)
	except ValueError:
		print(f"Error : '{expert_2}' is not a known expert. The known experts are : {KNOWN_EXPERTS}")
		error = True
	# Check if options.criterion exists. If not, quit the program.
	try:
		KNOWN_CRITERIA.index(options.criterion)
	except ValueError:
		print(f"Error : '{options.criterion}' is not a known criterion. The known criteria are : {KNOWN_CRITERIA}")
		error = True
	# Check if the two experts are the same. If true, quit the program.
	if expert_1 == expert_2:
		print(f"Error : the two experts used by the agent must be different.")
		error = True
	# Check if at least one expert is defined. If not, quit the program.
	if expert_1 == expert_2 == "None":
		print(f"Error : the agent need at least one expert to run.")
		error = True
	# -------------------------------------------------------------------------------
	# WARNING
	# If only one expert is used, notify that the  critertion of coordination will be not used
	elif expert_1 == "None" or expert_2 == "None":
		print(f"Warning : with only one expert, the criterion of coordination will not be used, because there will be no expert to coordinate.")
		options.criterion = "no_coordination"
	# -------------------------------------------------------------------------------
	print("\n")
	# -------------------------------------------------------------------------------
	if error == True:
		quit()
	else:
		return(experiment, expert_1, expert_2, map_file, key_states_file, spaces_file, parameters_file, options)
	# -------------------------------------------------------------------------------


if __name__ == "__main__":                          
	# -------------------------------------------------------------------------------
	# Manage the arguments et parse the parameters
	experiment, expert_1, expert_2, map_file, key_states_file, spaces_file, parameters_file, options = manage_arguments()
	parameters_MF, parameters_MB, parameters_DQN, parameters_MC = load_parameters(parameters_file, expert_1, expert_2)
	# -------------------------------------------------------------------------------
	# Initialize and regroup the variables and the constants
	experts = (expert_1, expert_2)
	spaces = load_spaces(spaces_file)
	boundaries_exp = {"max_reward": options.max_reward, "duration": options.duration, "window_size": options.window_size, "epsilon": 0.01}
	changes_exp = {"new_goal": options.new_goal, "add_wall": options.add_wall}
	log = {"log": options.log, "summary": options.summary}
	criterion = options.criterion
	coeff_kappa = options.coeff_kappa
	initial_variables = {"action_count": 0, "decided_action": 0, "actions_prob": 1/spaces["actions"], "previous_state": "0", "current_state": "0", \
	"qvalue": 1, "delta": 0.0, "plan_time": 0.0, "reward": 0}
	# -------------------------------------------------------------------------------
	# Create instances for the systems used by the virtual agent
	meta_controller = MetaController(experiment, map_file, initial_variables, spaces["actions"], boundaries_exp, parameters_MC, experts, criterion, coeff_kappa, log)
	experts_to_run = list()
	for expert in experts:
		if expert == "MF":
			new_expert = ModelFree(expert, experiment, map_file, initial_variables, spaces["actions"], boundaries_exp, parameters_MF, log)
			experts_to_run.append(new_expert)
		elif expert == "MB":
			new_expert = ModelBased(expert, experiment, map_file, initial_variables, spaces["actions"], boundaries_exp, parameters_MB, log)
			experts_to_run.append(new_expert)
		elif expert == "DQN":
			new_expert = DQN(expert, experiment, map_file, initial_variables, spaces, boundaries_exp, parameters_DQN, log)
			experts_to_run.append(new_expert)
		elif expert == "None":
			new_expert = None
			experts_to_run.append(new_expert)
	# -------------------------------------------------------------------------------
	# Run the simulation
	run_simulation(map_file, meta_controller, experts_to_run, spaces, boundaries_exp, changes_exp, initial_variables)
	# -------------------------------------------------------------------------------
	



