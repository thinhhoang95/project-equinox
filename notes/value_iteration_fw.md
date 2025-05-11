# Cost, Wind and Arrival Time (ETA)
- We represent a DAG route graph by its **cost matrix** `C` of size $|E| \times |E|$. It has infinite cost between pairs of nodes that do not have an edge (or link), and finite cost when they have an edge.
- In order to compute the cost matrix `C`, we need to know the **passing time** at each node in the graph. We call this vector of size $|N|$ as `ETA`. The reason is that we need to know the *altitude* and the *valid time* so we can pull the right weather forecast to get wind components, which in turn affect how fast one can fly on a given link.
- But to compute the `ETA`, we need to know the wind speed (we know the distance of every given link).

> The dilemma is that we need the `ETA` to find the wind, but we need the wind to know how fast we go, thus finding the `ETA`. Usually, we may iteratively estimate both, but **it might be too slow**.

# State Vector
The state vector at each node is: `[EETA, phase, altitude, logZ]` where:
- $E_{ETA}$ is the expected Estimated Time of Arrival, which is the average time that the node will be passed, given the overall probabilistic routing model:

  $$ E_{ETA}(n) = \sum_{\xi_{s,n,g}} t_n P[\xi_{s,n,g}]$$

- `phase` could be `climb`, `cruise` or `descent`, depending on $E_{ETA}$.

- `logZ` is the negative log partition function, which is defined as:

  $$ logZ = -\log \sum_{\xi_{s \to n}} \exp(-c(\xi_{s \to n};\theta)) $$

Because the graph is DAG, that means we can start from the source node `s`, then we begin propagating the probability mass `logZ` to the successor nodes, as well as the `EETA`. For simplicity, we will propagate the probability mass first, then the `EETA` later, although <font color='red'>It is not certain at this point whether this is really the best idea</font>.

# Algorithm (Forward Pass, before Top of Descent)
1. Start from the source node `s` with probability mass 1 (i.e., `logZ = 0`). The take-off time `CTOT` is given as parameter.
2. Propagating to immediate successors of `s`, which we will call `i`:
    1. Pick the `EETA` of `s`, the latitude and logitude of `i`. To get the altitude, call the function `get_heuristic_altitude` with coordinates of `s`, `i`, the `EETA` of `s`, and the `phase` at `s`. Once we have the triplet, we get the wind component. The link `si` will use this constant wind values.
    2. Knowing the wind values, we populate the cost matrix row corresponding to `s` and use the recursive formula:

        $$ V(i) = -\log \sum_{j \in pred(i)} \exp(-c(ji; \theta) - V(j)) $$

        where $pred(i)$ are predecessors of the node i. In our first pass, we only have one predecessor of `i`, which is `s`.

    3. After getting the probability mass updated for all successor nodes of `s`, we compute the $E_{ETA}$ for these nodes:

        $$ E_{ETA}(i) = \sum_{j \in pred(i)} (t + t_{ji}) \exp(-V(i) - c(ij; \theta) + V(j)) $$

        where $t_{ji}$ is the travel time for the link `ji`.

    4. Infer the `phase` of nodes `i` by comparing it between the climb time.

## Implementation Details
Please help me implement the barebone structure for the `get_next_state` function to solve the problem step by step:
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

