#!/usr/bin/env python3
#encoding: utf-8

'''
This script is part of the program to simulate a navigation experiment where a robot
has to discover the rewarded state of the arena in which it evolves using a meta-control
decision algorithm : "Dromnelle, R., Renaudo, E., Chetouani, M., Maragos, P., Chatila, R.,
Girard, B., & Khamassi, M. (2022). Reducing Computational Cost During Robot Navigation and 
Human–Robot Interaction with a Human-Inspired Reinforcement Learning Architecture. 
International Journal of Social Robotics, 1-27."

This script allows to set up the environment and to simulate the actions of the agent on it.
'''

__author__ = "Rémi Dromnelle"
__version__ = "1.0"
__maintainer__ = "Rémi Dromnelle"
__email__ = "remi.dromnelle@gmail.com"
__status__ = "Production"

from utility import *


def initialize_environment(key_states_file):
	"""
	Initialize the environment 
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
				switch_goal = int(line.split(" ")[1])
				new_goal = str(line.split(" ")[2])
				win_reward = int(line.split(" ")[3])
			elif line.split(" ")[0] == "new_wall":
				add_wall = int(line.split(" ")[1])
				path1 = str(line.split(" ")[2])
				path2 = str(line.split(" ")[3])
	# -------------------------------------------------------------------------------
	key_states = {"init_states": init_str, "goal": goal, "reward": win_reward}
	after_switch_goal = {"it_switch": switch_goal, "new_goal": new_goal, "reward": win_reward}
	after_add_wall = {"it_add": add_wall, "path1": path1, "path2": path2}
	# -------------------------------------------------------------------------------
	return key_states, after_switch_goal, after_add_wall
	# -------------------------------------------------------------------------------


def load_spaces(spaces_file):
	"""
	Initialize the state and the action spaces
	"""
	# -------------------------------------------------------------------------------
	# Load the list of initial states and the rewarded state
	init = list()
	goal = 0
	with open(spaces_file,'r') as file1:
		for line in file1:
			if line.split(" ")[0] == "state":
				state_space = int(line.split(" ")[1])
			elif line.split(" ")[0] == "action":
				action_space = int(line.split(" ")[1])
	# -------------------------------------------------------------------------------
	return state_space, action_space
	# -------------------------------------------------------------------------------


def update_robot_position(init_states, final_state, map_file, start, action):
	"""
	Simulate the effect of the robot's decision on the environement, that is to say,
	the identify of new state reaches after do the action in the previous state and 
	the reward obtains in this new state
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




