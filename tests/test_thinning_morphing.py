import networkx as nx


def test_thin_closures_morphs_adjacent_k_states():
    """
    Adjacent k states with close ETA ranges should be morphed onto the earlier k.
    """
    from equinox.dp.trespass.thinning import thin_closures

    # Dummy graph (thin_closures only uses it for signature).
    G = nx.DiGraph()
    G.add_nodes_from(["0", "1", "2"])

    source_node_idx = 0
    goal_node_idx = 2
    max_rho = 1
    dt = 600
    tolerance_s = 20.0

    closures = [
        (0, 0, 1, 0, 0, 1, 1, 0, 100, 1, 0.0, 600.0),
        (0, 0, 1, 0, 0, 1, 2, 0, 100, 1, 0.0, 610.0),
        (1, 2, 0, 100, 1, 2, 3, 0, 100, 1, 610.0, 1200.0),
        (1, 1, 0, 100, 1, 2, 2, 0, 100, 1, 600.0, 1150.0),
    ]

    thinned = thin_closures(
        source_node_idx=source_node_idx,
        goal_node_idx=goal_node_idx,
        max_rho=max_rho,
        G=G,
        closures=closures,
        wallclock_time_bin_k_tolerance_s=tolerance_s,
        delta_t_seconds_wall_clock=dt,
        include_wait_edges_in_output=False,
    )

    bases = {tuple(t[:10]) for t in thinned}
    expected = {
        (0, 0, 1, 0, 0, 1, 1, 0, 100, 1),
        (1, 1, 0, 100, 1, 2, 3, 0, 100, 1),
        (1, 1, 0, 100, 1, 2, 2, 0, 100, 1),
    }
    assert expected.issubset(bases)
    assert not any(
        (t[0] == 1 and t[1] == 2) or (t[5] == 1 and t[6] == 2)
        for t in bases
    )
