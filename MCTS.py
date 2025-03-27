import copy
import random
import math
import time 
from baselineTeam import ReflexCaptureAgent
from baselineTeam import OffensiveReflexAgent
from baselineTeam import DefensiveReflexAgent
from heuristicTeam import HeuristicAgent


def evaluateGameState(gameState, agents, blueTeamIndices):
    """
    Evaluate the game state as a heuristic game score.
    
    For each agent, we compute V(s) as the maximum Q-value over legal actions.
    We then average these values for each team:
      - Blue team: agents with indices in blueTeamIndices.
      - Red team: agents whose indices are not in blueTeamIndices.
      
    The game score is defined as:
          gameScore = (average value for red team) - (average value for blue team)
    
    A positive score indicates an advantage for red, while a negative score indicates
    an advantage for blue.
    
    Parameters:
      gameState: The current game state.
      agents: A list of Reflex agents.
      blueTeamIndices: A list of indices corresponding to agents on the blue team.
    
    Returns:
      A numeric game score.
    """
    # Compute the best Q-value (V(s)) for each agent by evaluating all legal actions.
    agentValues = {}
    for agent in agents:
        legalActions = gameState.getLegalActions(agent.index)
        if not legalActions:
            # If there are no legal actions, assign a default value (e.g., 0)
            agentValues[agent.index] = 0
        else:
            qValues = [agent.evaluate(gameState, action) for action in legalActions]
            agentValues[agent.index] = max(qValues)
    
    # Separate agents into blue and red teams based on provided indices.
    blueValues = [agentValues[i] for i in agentValues if i in blueTeamIndices]
    redValues = [agentValues[i] for i in agentValues if i not in blueTeamIndices]
    
    # Compute the average value for each team.
    redAvg = sum(redValues) / len(redValues) if redValues else 0
    blueAvg = sum(blueValues) / len(blueValues) if blueValues else 0
    
    # Return the game score: positive if red is favored, negative if blue is.
    return redAvg - blueAvg

def binary_score_with_food_stats(game_state):
    """
    Computes a binary score for the game based on the final score and
    prints additional details regarding the food returned by each team.

    Returns:
        1  if the final score is positive (Red wins),
       -1  if the final score is negative (Blue wins),
        0  if the final score is zero (tie).
    """
    final_score = game_state.data.score

    # Compute food returned counts for each team.
    red_food = 0
    blue_food = 0
    num_agents = game_state.getNumAgents()
    for i in range(num_agents):
        agent_state = game_state.data.agentStates[i]
        # Assume game_state.getRedTeamIndices() returns a list of indices for Red agents.
        if i in game_state.getRedTeamIndices():
            red_food += agent_state.numReturned
        else:
            blue_food += agent_state.numReturned

    # Print additional statistics.
    print("Red team's food returned:", red_food)
    print("Blue team's food returned:", blue_food)
    print("Final game score:", final_score)

    # Determine and print the result.
    if final_score > 0:
        print("Result: Red wins")
        return final_score
    elif final_score < 0:
        print("Result: Blue wins")
        return  final_score
    else:
        print("Result: Tie")
        return final_score

class MCTSNode:
    def __init__(self, agent_id, state, parent=None, action=None):
        """
        Initialize the MCTS Node.
        
        Args:
            agent_id (int): The index of the agent (e.g. 0 for red, 1 for blue).
            state: The current game state.
            parent (MCTSNode, optional): The parent node. Defaults to None.
            action (str, optional): The action taken from the parent to reach this state. Defaults to None.
        """
        self.agent_id = agent_id
        self.state = state
        self.parent = parent
        self.action = action
        # Get available moves from the state for the current agent.
        self.untried_moves = state.getLegalActions(agent_id)[:]  # copy the list
        self.children = {}  # dict mapping action string -> child MCTSNode
        self.visits = 0
        self.score = 0.0

    def state_heuristic(game_state):
        pass

    def is_blue(self):
        return self.agent_id in self.state.blueTeam

    def is_leaf(self):
        """Return True if the node has no children."""
        return len(self.children) == 0

    def is_terminal(self):
        """Return True if the game is over or no further moves are available."""
        return self.state.gameOver or len(self.untried_moves) == 0

    def uct_value(self, exploration=1.41):
        """
        Compute the UCT value for this node.
        (Assumes that self.parent exists and self.visits > 0.)
        """
        return (self.score / self.visits) + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)

    def select(self):
        """
        Recursively select a node for expansion.
        If the current node has untried moves or is terminal, return it.
        Otherwise, select the child with the best UCT value.
        """
        if len(self.untried_moves) > 0 or self.state.isOver():
            return self
        else:
            # All moves have been tried: select a child based on UCT.
            best_child = max(self.children.values(), key=lambda child: child.uct_value())
            return best_child.select()

    def expand(self):
        """
        Expand the node by taking one of its untried moves.
        This involves:
          - Popping an action from the untried moves.
          - Creating a deep copy of the current state.
          - Acting the move on the state to get a new state.
          - Creating a new child node with the new state.
        Returns:
            The new child node.
        """
        if len(self.untried_moves) == 0:
            return None
        
        # Select an untried move
        move = self.untried_moves.pop()
        new_state = copy.deepcopy(self.state)
        new_state = new_state.generateSuccessor(self.agent_id, move)
        
        # Assuming a two-agent game; switch agent for the next turn.
        next_agent = (self.agent_id +1) % 4
        
        child_node = MCTSNode(next_agent, new_state, parent=self, action=move)
        self.children[move] = child_node
        return child_node

    def rollout(self, rollout_depth,rollout_method,game_score, softmax_rollout, softmax_temp, debug=False):
        """
        Simulate a random playout (rollout) from the current state until game over.
        For each turn, the agent randomly chooses one legal move.
        Returns:
            The game score from the rollout (typically from the perspective of red).
        """
        simulation_state = copy.deepcopy(self.state)
        curr_agent_id = self.agent_id
        move_count= 0

        # if rollout_method == "reflex":
        if rollout_method == "reflex" or rollout_method == "random":
            agents_defense= [DefensiveReflexAgent(i) for i in range(2)]
            agents_offense= [OffensiveReflexAgent(i) for i in range(2)]
            agents= agents_defense + agents_offense
            list(map(lambda agent: agent.registerInitialState(simulation_state), agents))
        elif rollout_method == "custom_heuristic":
            agents= [HeuristicAgent(i) for i in range(4)]
            list(map(lambda agent: agent.registerInitialState(simulation_state), agents))

        # Rollout until the game is over.
        actions_taken = []
        while not simulation_state.isOver() and move_count <= rollout_depth:
            move_count += 1
            legal_actions = simulation_state.getLegalActions(curr_agent_id)
            if not legal_actions:
                break  # no legal moves available

            curr_agent= agents[curr_agent_id] # NOT used if random rollout
            if rollout_method == "random":
                action = random.choice(legal_actions)
                simulation_state= simulation_state.generateSuccessor(curr_agent_id, action)
            elif rollout_method == "reflex":
                action= curr_agent.chooseAction(simulation_state)
                simulation_state = curr_agent.getSuccessor(simulation_state, action)
            elif rollout_method == "custom_heuristic":
                if softmax_rollout:
                    action = curr_agent.chooseActionSoftmax(simulation_state, softmax_temp)
                else:
                    action= curr_agent.chooseAction(simulation_state)
                simulation_state = curr_agent.getSuccessor(simulation_state, action)
            if curr_agent_id == ((self.agent_id-1)%4):
                actions_taken.append(action)
            curr_agent_id = (curr_agent_id + 1) % 4
        
        print(actions_taken)
        score= None
        
        if game_score == "reflex_heuristic":
            score= evaluateGameState(simulation_state, agents, simulation_state.blueTeam)
        elif game_score == "custom_heuristic":
            # score = HeuristicAgent.evaluateState(simulation_state) 
            score = HeuristicAgent.evaluateState(root.agent_id, simulation_state) 
        else:
            score= simulation_state.data.score

        if debug:
            print("Rollout ended: ",score,"in",move_count,"steps", "game over:",simulation_state.isOver())

        return score

    def backpropagate(self, rollout_score):
        """
        Backpropagate the rollout score up the tree.
        At each node, increment the visit count and add the (possibly flipped) score.
        Blue nodes (agent_id == 1) flip the score.
        
        Args:
            rollout_score (float): The score obtained from the rollout.
        """
        node = self
        score = rollout_score
        while node is not None:
            # If the node's agent is blue (minimizer), flip the score.
            if not node.is_blue():
                score = -score
            node.visits += 1
            node.score += score
            node = node.parent

class MCTS:
    def __init__(self, agent_id, game_state, iterations=10, rollout_method= "random", state_heuristic= "default", rollout_depth= 1000, softmax_rollout= False, softmax_temp = 1, debug=False):
        """
        Initialize MCTS with the starting agent and game state.
        
        Args:
            agent_id (int): The starting agent's index.
            game_state: The current game state.
            iterations (int, optional): The number of MCTS iterations. Defaults to 1000.
        """
        self.rollout_method= rollout_method
        self.root = MCTSNode(agent_id, game_state)
        self.iterations = iterations
        self.state_heuristic= state_heuristic
        self.rollout_method= rollout_method
        self.rollout_depth= rollout_depth
        self.softmax_rollout= softmax_rollout
        self.softmax_temp= softmax_temp
        self.debug = debug

    def run(self):
        """
        Run the MCTS algorithm for a given number of iterations.
        For each iteration, do the following:
          1. Select a node.
          2. Expand the node (if possible).
          3. Perform a rollout from the new node.
          4. Backpropagate the rollout score.
        
        Finally, select the root child with the highest visit count and return its associated action.
        
        Returns:
            The best action determined by MCTS.
        """
        for i in range(self.iterations):
            # 1. Selection: traverse down the tree until a node with untried moves or terminal state is reached.
            node = self.root.select()
            
            # 2. Expansion: if the node is not terminal and has untried moves, expand it.
            # TODO is this correct
            if not node.state.isOver() and len(node.untried_moves) > 0:
                node = node.expand()
            
            #print(f"Rollout {i} starting")
            start_time = time.time()  # record the start time

            print(f"rollout {i} for {node.parent.agent_id} performing {node.action}")
            rollout_score = node.rollout(self.rollout_depth, self.rollout_method, self.state_heuristic, self.softmax_rollout, self.softmax_temp, root = self.root)  # simulate a random playout

            end_time = time.time()  # record the end time
            elapsed_time = end_time - start_time  # calculate elapsed time in seconds

            if self.debug:
                print(f"Rollout {i} ended with score {rollout_score} in {elapsed_time:.2f} seconds")
                        
            # 4. Backpropagation: propagate the score back up the tree.
            node.backpropagate(rollout_score)


        
        # After iterations, select the child of the root with the most visits.
        best_move, best_child = max(self.root.children.items(), key=lambda item: item[1].visits)
        return best_move