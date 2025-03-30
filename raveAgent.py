# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from MCTS import MCTS
from baselineTeam import ReflexCaptureAgent
from math import sqrt, log
import numpy as np

from baselineTeam import OffensiveReflexAgent
from baselineTeam import DefensiveReflexAgent
from heuristicTeam import HeuristicAgent
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
							 first = 'TreeSearch', second = 'TreeSearch'):
	"""
	This function should return a list of two agents that will form the
	team, initialized using firstIndex and secondIndex as their agent
	index numbers.  isRed is True if the red team is being created, and
	will be False if the blue team is being created.

	As a potentially helpful development aid, this function can take
	additional string-valued keyword arguments ("first" and "second" are
	such arguments in the case of this function), which will come from
	the --redOpts and --blueOpts command-line arguments to capture.py.
	For the nightly contest, however, your team will be created without
	any extra arguments, so you should make sure that the default
	behavior is what you want for the nightly contest.
	"""

	# The following line is an example only; feel free to change it.
	return [eval(first)(firstIndex), eval(second)(secondIndex)]

import random, math
from captureAgents import CaptureAgent
from game import Directions

##########################################################
# Node class with full RAVE (AMAF) statistics
##########################################################

class Node():
    def __init__(self, parent, action_taken, agent_id, gameState):
        self.parent = parent               # Parent node (None for root)
        self.parent_action = action_taken  # The action taken to reach this node from the parent
        self.agent_id = agent_id           # Which agent’s turn is represented at this node
        self.state = gameState             # The game state at this node
        self.children = []                 # Child nodes (expanded moves)
        self.visits = 0                    # Standard MCTS visit count
        self.tot_s = 0                     # Standard total reward

        # For tree policy: available moves that haven’t been expanded yet
        self.untried_moves = gameState.getLegalActions(agent_id)[:]

        # RAVE (AMAF) statistics: for each move that could be taken from this state,
        # we maintain an (amaf count, amaf total reward) pair.
        self.amaf_counts = {}
        self.amaf_total = {}
        for a in gameState.getLegalActions(agent_id):
            self.amaf_counts[a] = 0
            self.amaf_total[a] = 0

    def expand(self):
        """Choose one untried move at random, remove it from untried_moves,
           create a new child node, and return it."""
        a = random.choice(self.untried_moves)
        self.untried_moves.remove(a)
        next_state = self.state.generateSuccessor(self.agent_id, a)
        child = Node(self, a, (self.agent_id + 1) % 4, next_state)
        self.children.append(child)
        return child

    def bestChild(self, exploration_constant=2, rave_constant=300):
        """Select the child with the highest blended value.
           The blended value uses a dynamic weighting factor (beta) that favors
           the AMAF (RAVE) estimate when the visit count is low."""
        best_value = -float('inf')
        best_child = None

        for child in self.children:
            # Standard Q-value (if no visits, treat as 0)
            q = child.tot_s / child.visits if child.visits > 0 else 0

            # AMAF (RAVE) estimate for the move that led to this child:
            amaf_count = self.amaf_counts.get(child.parent_action, 0)
            amaf_value = (self.amaf_total.get(child.parent_action, 0) / amaf_count) if amaf_count > 0 else 0

            # Dynamic weighting parameter beta:
            # When visits are low, beta is high so that the AMAF estimate dominates;
            # as visits increase, beta decays.
            beta = math.sqrt(rave_constant / (3 * child.visits + rave_constant)) if child.visits > 0 else 1

            combined_value = (1 - beta) * q + beta * amaf_value

            # UCT exploration term; note that if a child has not been visited, we give it a high bonus.
            uct_value = exploration_constant * math.sqrt(math.log(self.visits + 1) / (child.visits + 1))
            value = combined_value + uct_value

            if value > best_value:
                best_value = value
                best_child = child

        return best_child

##########################################################
# TreeSearch Agent using Full RAVE MCTS
##########################################################

class TreeSearch(CaptureAgent):
    def __init__(self, agent_id, rollout_depth=25, rollout_i=100, rave_constant=100, custom_heuristic = False):
        super().__init__(agent_id)
        self.rollout_depth = rollout_depth  # Maximum depth for the rollout
        self.rollout_i = rollout_i          # Number of MCTS simulations per move
        self.rave_constant = rave_constant  # Constant used in dynamic beta for RAVE
        self.custom_heuristic = custom_heuristic
        
    def simulation(self, root):
        """
        Run one simulation (playout) from the root.
        Returns:
          - visited: the list of nodes (the tree path) that were traversed
          - actions: the list of actions taken during the simulation (both in tree and rollout)
          - score: the evaluation (reward) from the simulation.
        """
        visited = []
        node = root

        # Tree Policy: traverse until a node with untried moves is found or state is terminal.
        while True:
            visited.append(node)
            if node.state.isOver():
                break
            if node.untried_moves:
                # Expand one move and add the new node to the tree
                node = node.expand()
                visited.append(node)
                break
            else:
                # Select the best child according to the UCT+RAVE blended value.
                node = node.bestChild()
        # Record actions from the tree part of the simulation.
        # (Here we simply record the actions stored in the nodes on the path, except the root.)
        tree_actions = [n.parent_action for n in visited if n.parent is not None]

        # Default Policy (Rollout): simulate until terminal state or a maximum depth is reached.
        simulation_state = node.state
        current_agent = node.agent_id
        rollout_actions = []
        depth = 0
        
        if self.custom_heuristic:
            agents = [HeuristicAgent(i) for i in range(4)]
            list(map(lambda agent: agent.registerInitialState(node.state), agents))
        while (not simulation_state.isOver()) and (depth < self.rollout_depth):
            legal_moves = simulation_state.getLegalActions(current_agent)
            if not legal_moves:
                break
            if self.custom_heuristic:
                curr_agent = agents[current_agent]
                a = curr_agent.chooseAction(simulation_state)
            else:
                a = random.choice(legal_moves)
            rollout_actions.append(a)
            simulation_state = simulation_state.generateSuccessor(current_agent, a)
            current_agent = (current_agent + 1) % 4
            depth += 1

        # Combine the actions from the tree path and the rollout.
        actions = tree_actions + rollout_actions

        # Evaluate the final simulation state.
        foodList = self.getFood(simulation_state).asList()
        food_d=0
        if foodList:
            myPos = simulation_state.getAgentState(self.index).getPosition()
            food_d = min([self.getMazeDistance(myPos, food) for food in foodList])
        else:
            food_d = 0

        if self.custom_heuristic:
            score = self.calculate_score_extended(simulation_state, food_d)
        else:
            score = self.calculate_score(simulation_state, food_d)
        return visited, actions, score

    def backup(self, visited, actions, score):
        """
        For each node in the visited tree path, update:
         - Standard statistics (visits and total reward)
         - AMAF (RAVE) statistics for all moves played later in the simulation
           (if those moves are legal in that node).
        """
        for i, node in enumerate(visited):
            node.visits += 1
            node.tot_s += score
            # For each action that occurs after this node in the simulation,
            # update the AMAF stats if the action is legal in the node.
            for a in actions[i:]:
                if a in node.amaf_counts:
                    node.amaf_counts[a] += 1
                    node.amaf_total[a] += score

    def chooseAction(self, gameState):
        """
        Build the search tree via multiple simulations and return the best action.
        """
        root = Node(None, None, self.index, gameState)
        for _ in range(self.rollout_i):
            visited, actions, score = self.simulation(root)
            self.backup(visited, actions, score)

        # At the root, choose the child with the highest average value (set exploration term to 0).
        best_child = root.bestChild(exploration_constant=0, rave_constant=self.rave_constant)
        if best_child is None:
            # Fall back to a random legal action if something goes wrong.
            return random.choice(gameState.getLegalActions(self.index))
        return best_child.parent_action

    def calculate_score(self, state, food_d):
        """
        Compute a score for a state based on the game score,
        food distance, opponents’ carried food, and our team’s progress.
        """
        base_score = super().getScore(state) * 100
        opponents_food_penalty = sum(state.getAgentState(i).numCarrying for i in self.getOpponents(state))
        team_food_reward = sum(state.getAgentState(i).numCarrying + state.getAgentState(i).numReturned for i in self.getTeam(state))
        final_score = base_score - food_d - opponents_food_penalty + team_food_reward
        return final_score

    def calculate_score_extended(self, node, food_d):
        # state = node.state
        state = node
        features = util.Counter()
		# score of the game
        features["score"] = super().getScore(state)

		# amount of food left
        foodList = self.getFood(state).asList()    
        features['foodLeft'] = -len(foodList)#self.getScore(successor)

		# dist to nearest food
        features['distanceToFood'] = food_d
        for i in self.getTeam(state):
			# position heuristics
            myState = state.getAgentState(i)
            myPos = myState.getPosition()
	
			# Computes whether we're on defense (1) or offense (0)
            features['onDefense'] = 1
            if myState.isPacman: features['onDefense'] = 0

            # Computes distance to invaders we can see
            enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
            invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
            features['numInvaders'] = len(invaders)
            if len(invaders) > 0:
                dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
                if features['invaderDistance']:
                    features['invaderDistance'] = np.mean(min(dists), features['invaderDistance'])

            # action = node.parent_a
            # if action == Directions.STOP: features['stop'] = 1
            # rev = Directions.REVERSE[state.getAgentState(i).configuration.direction]
            # if action == rev: features['reverse'] = 1

            # Compute distance to defenders
            if myState.isPacman:
                defenders = [a for a in enemies if (not a.isPacman) and a.getPosition() != None]
                if len(defenders) > 0:
                    dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
                    features['ghostDistance'] = min(dists)

            weights = {'successorScore': 1000, 'foodLeft': 100, 'distanceToFood': -1,
            'numInvaders': -100, 'onDefense': 1,
            'invaderDistance': -10, 'ghostDistance': 1,'stop': -100, 'reverse': -2}

            return features*weights