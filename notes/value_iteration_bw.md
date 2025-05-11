## Implementation Details
Please help me implement the `get_next_state_bw` function to solve the problem step by step. This function is similar to the forward version `get_next_state_fw`, but we start from the destination/goal node of the route graph, and work backward to the preceding nodes in the route graph, following the descent (or cruise) profile.

## Algorithm Draft

0. If the phase at 


## Prompt

0. If the phase is `cruise` (or 1), then the altitude at targets is the same as the sources.
1. Compute the haversine distance between the source nodes and the target nodes, and the track between the source nodes and target nodes.
2. Given the time of arrival at source nodes `eta_src`, interpolate from the `climb_performance` table (using the second time column) to obtain the altitude at source nodes `alt_src`.
3. Using the `wind_model`, find the along-track wind at the source nodes (at the time `eta_src` and the `alt_src` as well). Positive means tail-wind, negative means head-wind.
4. From the time values given in `climb_performance`, compute the vector of distance covered by the wind. This value will be subtracted from the wind_free_covered_distance in climb_performance, so we can use them to find out the remaining distance to be covered by TAS (true air speed). This is another column in the `climb_performance` table.
5. Linearly interpolate this column with the haversine distance found from step 1, read out the altitude and the time reaching the target.
6. If from step 0, aircraft is crusing, we estimate the ETA at targets by dividing the adjusted distance (for wind) by the cruising TAS (last line of the performance table).

### Edge case
It is possible that the top of climb may be between the source and target node. In this case, the top of climb is the last row in `climb_performance`. Thus the distance to be covered by TAS from haversine distance (adjusted by the wind) will be the distance until the top of climb, plus the remaining distance. This distance will be traveled by the cruising TAS, which is given by the last row of the climb performance table. This allows us to compute the ETA at the target (the altitude is of course the cruise altitude).

### Requirements
- Because the inputs are torch.tensor, implement a torch vectorized algorithm of everything.
- Everything should be done with vectorization to maximize performance.

