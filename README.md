# MCTS Workflow
## Notes

### Get available actions for agent 
- action string

gameState.getLegalActions(agent index) -> list of strings


### Act the action
self.state = gameState.generateSuccessor( agentIndex, action)


### Get the position of the agent
myState = gameState.getAgentState(agentIndex)
myPos = myState.getPosition()

### Game score

state.data.score

self.gameOver

Blue wants to minimize, red wants to maximize

## MCTS Node

- game state
- agentId
- untried moves: list[str]
- visits, accumulated score
- childs: dict[str]

### Node Selection

- if node has untried moves stay on it (expand on those)
- if all tried select one child based on **UCT**

### Expansion
- create a new child node
- select an untried move, pop the move
- act the move and get the new state
    - first deep copy the original state and act on it

#### Node initialization
gets -> agent id, game state
init
    -> get available moves from state -> set untried moves

### Rollout(method)

deep copy state

untill game over

    for each agent

        randomly act on state
return game score
    
### Backpropagate
-> gets the rollout score
-> starts from the just expanded state
update the accumulated results for nodes 
while updating a nodes result incrememnt visit count

While node is not root
if team is blue on current state
- flip score  

Update score

pass it to the parent


## MCTS Algorithm

Initialize root mcts node

for mcts iteration
1. Select Node
2. Expand Node
3. Rolout on node
4. Backpropagate

Among the roots childs select the one with the most visit count 

return the action for that child node



## Data Structures

MCTS(agent id, game_state)
-> root: MCTS Node
-> mcts iteration

mcts() -> return action:
    for mcts iteration
        node = root.select()
        node= node.expand()
        score= node.rollout()
        node.backpropagate(score)

    Among the roots childs select the one with the most visit count 

    return the action for that child node



MCTS_Node(agent id, game_state)
    init
    -> get available moves from state -> set untried moves


    is_leaf()
        has no child

    is_terminal()
        has no untried moves






