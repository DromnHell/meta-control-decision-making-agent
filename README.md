# Meta-control decision-making experiment

## Overview 

This program has been developed during my PhD: ["Dromnelle, R. (2021). Architecture cognitive générique pour la coordination de stratégies d'apprentissage en robotique (Doctoral dissertation, Sorbonne université)."](https://www.theses.fr/2021SORUS039) It allows an agent to solve learning problems in abstract virtual environments containing reward.

The main objective of my thesis was to create a meta-control algorithm that allows an agent to coordinate several online behavior strategies based on reinforcement learning. This program allowed me to quickly evaluate several coordination criteria in simulation before performing experiments in the real environment with a real robot. 

## Script to simulate an agent evolving / acting in an environment containing a reward source

* The *loadFiles.py* script allows to load some files required to run the simulation.
* The *agentSimulator.py* script is the core of the program.

  It takes as input 7 mandatory ordered arguments :
  1.  the id of the experiment that we are going to launch. It must be an integer.
  2.  the id of the first expert, among this list : MF, MB, DQN.
  3.  the id of the second expert among this list : MF, MB, DQN.
  4.  the file that contains the representation of the environment,
  5.  the file that describes the key states of the environment,
  7.  the file that describes the state space and the action space,
  7.  the file that contains the parameters of the agent.
  
  In addition, it can also take 9 optional arguments :
  * (-c) the coordination criterion the agent will use, among this list : random (default value), entropy, entropy_and_cost.
  * (-k) the value of the kappa coefficient,
  * (-r) the amount of reward expected before the simulation stops,
  * (-d) the maximum duration of the simulation,
  * (-w) the window size of the filtering,
  * (-g) the indication of an upcoming goal change in the environment. Can't be set with (-t) option,
  * (-t) the indication of an upcoming transitions change in the environment. Can't be set with (-g) option,
  * (-l) the record of the data,
  * (-s) the record of a summarized version of the data.
  
  Exceptions :
  * ```Error : 'expert' is not a known expert. The known experts are : ['MF', 'MB', 'DQN', 'None']``` if one of the experts (arguments 2 and 3) are unknown.
  * ```Error : 'criterion' is not a known criterion. The known criteria are : ["random", "entropy", "entropy_and_cost"]``` if the criterion (option -c) is unknown.
  * ```Error : the two experts used by the agent must be different.``` if the two experts are the same.
  * ```Error : the agent need at least one expert to run.``` if the two experts (arguments 2 and 3) are both 'None'.

  Warning :
  * ```Warning : with only one expert, the criterion of coordination will not be used, because there will be no experts to coordinate.```

* The *modelFreeRL.py* script allows the agent to use a Q-learning algorithm (model-free reinforcement learning) 
to learn to solve the task.
* The *modelBasedRL.py* script allows the agent to use a Value-Iteration algorithm (model-based reinforcement learning) to learn to solve the task,
* The *DQN.py* script allows the agent to use a Deep Q-Network algorithm (deep reinforcement learning) 
to learn to solve the task. This DQN agent uses prioritized experience replay and a second network to compute the targeted qvalues.
* The *prioritizedSweeping.py* script allows the agent to use a Prioritized Sweeping algorithm (model-based
reinforcement learning) to learn to solve the task (NOT CONNECTED TO THE PROGRAM AT THE MOMENT). 
* The *metaControllerSystem.py* script allows the agent to coordinate the different behavioral strategies implemented in the previous scripts.
* The *utility.py* script contains some functions used by the other scripts.

 ## Files that describe the navigation environment used during my PhD
 

* The *realisticNavWorld.json* file is a transition model generated by a Turtlebot having explored a navigation arena for 13 hours. This transition model contains the consequences of each action of the agent in the environment, in the form of a set of discrete probabilities. We can see it as an "image" of the real environment, where only the information of interest is kept, that is to say in this case only what happens when the agent moves in such direction.
* The *realisticNavWorld_newGoal.json* file is identical to the realisticNavWorld.json file, with a change in the location of the reward, from the state 18 to the state 34. When the option (-g) is set, the associated transition model file has to be named with the original name, following with '_newGoal.json'. 
* The *realisticNavWorld_noTrans20-21.json* file is identical to the realisticNavWorld.json file, with the deletion of the transition between the states 20 and 21. When the option (-t) is set, the associated transition model file has to be named with the original name, following with '_noTransX-Y.json', where X and Y are the states concerned by the transiton deletion.
* The *realisticNavWorld_noTrans6-7.json* file is identical to the realisticNavWorld.json file, with the deletion of the transition between the states 6 and 7.
* The *keyStates.txt* file contains the id of the rewarded states and the initial states.
* The *spaces.txt* file contains the actions space and the states space.
* The *parameters.txt* file contains the parameters of the agent.
  * For the MF expert, the parameters are the learning rate (alpha), the discount factor (gamma) and the exploration rate (beta),
  * For the MB expert, the parameters are the same as the MF expert,
  * For the DQN expert, the parameters are initial alpha, alpha min, alpha decay, gamma, initial epsilon, epsilon min and epsilon decay,
  * For the MC system, the parameters are alpha and beta.

#### (A) The map autonomously build by the robot (the *realisticNavWorld.json* file is an abstraction of this map). (B) Photo of the arena and the robot.

![arena](https://raw.githubusercontent.com/DromnHell/meta-control-decision-making-agent/main/map_actions_photo.png)
 
## Dependencies

* python 3.10
* tensorflow 2.9
* numpy

## Examples of commands to run the program

```
python agentSimulator.py 0 MF MB realisticNavWorld.json keyStates.txt spaces.txt parameters.txt -d 1600 -c entropy_and_cost
```

This code will run the experiment 0, where an agent using an MB and MF expert will perform a navigation task for a duration of 1600 iterations, and using the "Entropy and Cost" coordination criterion.

```
python agentSimulator.py 1 DQN None realisticNavWorld.json keyStates.txt spaces.txt parameters.txt -r 100 -l -c entropy
```
This code will run the experiment 1, where an agent using only one expert (a DQN) will perform a navigation task until 100 units of reward are cumulated. Because only one expert is used, the "Entropy" coordination criterion will not be used (see "Warning" above). The data will be reccorded.

## Example of an output file when the log option is set


```
...
1598 18 1 9.936632209345335e-07 MF 0.9233713747401607 1.0 1.0 5.608592928084312e-05
1599 0 0 9.462273008389549e-07 MF 0.9904485966915921 0.6847743322036773 0.6494277836137697 6.362911275139294e-05
1600 0 0 9.784909203355819e-07 MF 0.9968824302693863 0.674786980876488 0.658064217038684 6.505164510055717e-05
```
These data correspond to : 

* the current iteration,
* the current state where the agent is,
* the potentially reward obtained by the agent,
* the time of planning at this iteration,
* the ID of the winner expert who controlled the robot at this iteration,
* the probability of selection of the action that has been performed by the agent,
* the entropy of the distribution of action selection probabilities of the winner expert,
* the same metric for the looser expert,
* the cumulated filtered planning time of the agent.

