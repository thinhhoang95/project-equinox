## Post-train checkpoint helpers

This page shows small usage snippets for reading `lin_disent` checkpoints and visualizing
edge preferences on a Cartopy map.

### Load common weights + preference matrix

```python
from equinox.posttrain import load_cost_model_parameters

checkpoint_path = "data/cases/LEMD_EGLL/batch_sgd_results/checkpoint_iter_100.pt"

params = load_cost_model_parameters(checkpoint_path)
print(params["common"])  # {"bias": ..., "ac_dist": ..., "time": ...}
print(params["preference_matrix"].shape)  # (num_nodes, num_nodes)
```

### Map preferences to graph edges (from GML)

```python
from equinox.posttrain import load_checkpoint_edge_preferences

checkpoint_path = "data/cases/LEMD_EGLL/batch_sgd_results/checkpoint_iter_100.pt"
gml_path = "data/cases/LEMD_EGLL/graphs/routes.gml"

payload = load_checkpoint_edge_preferences(checkpoint_path, gml_path)
edge_prefs = payload["edge_preferences"]

# Example: read a single edge preference
print(edge_prefs[("MADRID", "PENIL")])
```

### Attach preferences directly to graph edges

```python
import networkx as nx
from equinox.posttrain import (
    attach_preferences_to_graph,
    load_cost_model_parameters,
    load_graph_from_gml,
)

checkpoint_path = "data/cases/LEMD_EGLL/batch_sgd_results/checkpoint_iter_100.pt"
gml_path = "data/cases/LEMD_EGLL/graphs/routes.gml"

params = load_cost_model_parameters(checkpoint_path)
graph, node_to_idx, _ = load_graph_from_gml(gml_path)
graph = attach_preferences_to_graph(
    graph,
    node_to_idx,
    params["preference_matrix"],
    attr_name="preference",
)

print(graph.edges[list(graph.edges)[0]]["preference"])
```

### Plot edge preferences on a Cartopy map

```python
from equinox.posttrain import load_checkpoint_edge_preferences, plot_edge_preferences_cartopy

checkpoint_path = "data/cases/LEMD_EGLL/batch_sgd_results/checkpoint_iter_100.pt"
gml_path = "data/cases/LEMD_EGLL/graphs/routes.gml"

payload = load_checkpoint_edge_preferences(checkpoint_path, gml_path)
plot_edge_preferences_cartopy(
    payload["graph"],
    edge_preferences=payload["edge_preferences"],
    cmap="coolwarm",
    linewidth=1.2,
)
```

### Important: filter out edges not present in CLSR

Some edges in `routes.gml` may never appear in any feasible state-transition closure (`CLSR_*.pkl`).
For those edges, `preference_matrix_p[u, v]` is typically unidentifiable and can look arbitrarily
"strong" when plotted. You can filter to only CLSR-supported edges:

```python
from equinox.posttrain import (
    filter_edge_preferences_to_support,
    load_checkpoint_edge_preferences,
    load_clsr_transition_edge_counts,
)

case_dir = "data/cases/LEMD_EGLL"
checkpoint_path = f"{case_dir}/results_full/final_results.pt"
gml_path = f"{case_dir}/graphs/routes.gml"

payload = load_checkpoint_edge_preferences(checkpoint_path, gml_path)
support = load_clsr_transition_edge_counts(case_dir, gml_path)
edge_prefs = filter_edge_preferences_to_support(payload["edge_preferences"], support)
```

If you use `load_checkpoint_edge_preferences(...)`, CLSR support is automatically attached
to `payload["graph"]` as the edge attribute `clsr_transition_count`, and
`plot_edge_preferences_cartopy(...)` hides edges without CLSR support
(missing or zero `clsr_transition_count`) by default.
