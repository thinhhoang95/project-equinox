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
