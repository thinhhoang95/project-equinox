import numpy as np
import networkx as nx

def _latlon_to_vec(lat, lon):
    """Convert lat, lon (in degrees) to a 3D unit vector."""
    φ = np.radians(lat)
    θ = np.radians(lon)
    return np.array([np.cos(φ) * np.cos(θ),
                     np.cos(φ) * np.sin(θ),
                     np.sin(φ)])

def refine_graph(G, tol_plane=1e-6, tol_angle=1e-6):
    """
    Replace any edge u→v in the DiGraph G with u→w and w→v whenever
    there exists a node w (≠u,v) that lies on the same great circle
    between u and v.
    
    This edits G in place and also returns it.
    
    Parameters
    ----------
    G : nx.DiGraph
        Nodes must have 'lat' and 'lon' attributes.
    tol_plane : float
        Tolerance for testing (A×B)⋅C ≈ 0 (coplanarity).
    tol_angle : float
        Tolerance for testing |∠A–C + ∠C–B – ∠A–B| ≈ 0.
    """
    # Precompute 3D unit vectors for each node
    nodes = list(G.nodes())
    vecs = {}
    for n in nodes:
        lat = G.nodes[n]['lat']
        lon = G.nodes[n]['lon']
        vecs[n] = _latlon_to_vec(lat, lon)
    # Build an array of all vectors for fast dot/cross
    all_vecs = np.vstack([vecs[n] for n in nodes])
    
    # Iterate over a static list of edges, since we'll mutate G
    for u, v in list(G.edges()):
        A = vecs[u]
        B = vecs[v]
        # cross‑vector defines the great‑circle plane
        plane_norm = np.cross(A, B)
        norm = np.linalg.norm(plane_norm)
        if norm < tol_plane:
            # A and B are (nearly) antipodal or identical → skip
            continue
        plane_norm /= norm
        
        # Dot all vectors with plane_norm: collinear if ≈0
        dots = all_vecs.dot(plane_norm)
        mask = np.isclose(dots, 0, atol=tol_plane)
        
        # Exclude u and v themselves
        candidates = [nodes[i] for i, m in enumerate(mask) if m and nodes[i] not in (u, v)]
        if not candidates:
            continue
        
        # Precompute the total arc‑angle A–B
        cosAB = np.clip(A.dot(B), -1.0, 1.0)
        total_angle = np.arccos(cosAB)
        
        # Find the candidate w minimizing |angle(A,C)+angle(C,B) – total|
        best_w = None
        best_diff = tol_angle
        for w in candidates:
            C = vecs[w]
            # arc‑angles via dot‑product
            angle_AC = np.arccos(np.clip(A.dot(C), -1.0, 1.0))
            angle_CB = np.arccos(np.clip(C.dot(B), -1.0, 1.0))
            diff = abs((angle_AC + angle_CB) - total_angle)
            if diff < best_diff:
                best_diff = diff
                best_w = w
        
        if best_w is not None:
            # Preserve any existing edge attributes
            edge_attr = G.get_edge_data(u, v, default={})
            G.remove_edge(u, v)
            G.add_edge(u, best_w, **edge_attr)
            G.add_edge(best_w, v, **edge_attr)
    
    return G