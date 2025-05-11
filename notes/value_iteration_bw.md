## Implementation Details
Please help me implement the `get_next_state_bw` function to solve the problem step by step. This function is similar to the forward version `get_next_state_fw`, but we start from the destination/goal node of the route graph, and work backward to the preceding nodes in the route graph, following the descent (or cruise) profile.

## Algorithm Draft

> Because we are going backward, there could be confusion about which nodes are sources, which nodes are targets. We define a source node the node where we fly from, and a target node is the node we fly to (same direction as the directed route graph). 

> We start from the targets and work our way back to the sources.

1. If the phases at the target node is `CRUISE`, the phase at the source node is also `CRUISE`. The altitude of the source node is the same as of the target node. The ETA at the source node will be calculated later. 

2. Follow the same logic as `get_next_state_fw`, only this time the `performance_table` is given as descent performance.

3. The ETA at the sources will be given (in fact, they are heuristically estimated) in order to pick the correct wind condition.

### Edge case
It is possible that the top of descent may be between the source and target node. In this case, the top of descent is the first row in `descent_performance`. Thus the distance to be covered by TAS from haversine distance (adjusted by the wind) will be the distance until the top of descent (we are talking in a backward sense, going from the target to the source), plus the remaining distance traveled at the cruising TAS, which is given by the first row of the descent performance table. This allows us to compute the ETA at the target (the altitude is of course the cruise altitude).

### Requirements
- Because the inputs are torch.tensor, implement a torch vectorized algorithm of everything.
- Everything should be done with vectorization to maximize performance.

