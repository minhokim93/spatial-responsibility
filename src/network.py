"""
Spatial responsibility networks:

- build_network: directed graph with nodes weighted by PR and links weighted by SR or OR
- subnetworks / network_metrics: connected-component (subnetwork) breakdown of a network, with per-subnetwork responsibility
- percolate_graph: removes one batch of links from a graph under a given removal strategy (Random / Localized / Targeted)
- percolate: repeatedly applies `percolate_graph` over a fraction of links removed schedule and records subnetwork-level responsibility at each step
- run_percolation_trials: repeats `percolate` over multiple trials/modes for comparison (e.g. Viz for seed-averaged curves).
- run_simulation / run_simulation_multiseed: Subnetwork monitoring on top of `percolate` for the stacked-bar and heatmap figures
"""

import math
import random

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx
import networkx as nx
from utils import SQM_TO_SQFT


# ---------------------------------------------------------------------------
# 1. Network construction
# ---------------------------------------------------------------------------
def build_network_v2(gdf, edge_variable='SR', node_variable='TR', crs1=None, crs2=None, directed=False, weight_by='responsibility'):
    '''
    weight_by='responsibility' (default): Link weight is Area x Risk metric giving product [ft^3/min]
    weight_by='area': Link weight is overlapping raw area [ft^2] unweighted by risk.
    '''

    if crs1:
        gdf = gdf.set_crs(crs1)
    if crs2:
        gdf = gdf.to_crs(crs2)

    # Set centroids (based on building geometry)
    centroids = gdf['geometry'].centroid
    centroids_gdf = gpd.GeoDataFrame(geometry=centroids, crs=gdf.crs)

    # Build responsibility (or area) networks
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    for idx, centroid in centroids_gdf.iterrows():
        G.add_node(idx, pos=(centroid.geometry.x, centroid.geometry.y), TR=gdf.loc[idx, node_variable])

    # Add links to network (avg SR and OR)
    for i, row_i in gdf.iterrows():

        if edge_variable == 'SR':
            # SR (Owner and neighbors) --> Cumulative
            if isinstance(row_i['SR_ID'], (list, tuple, np.ndarray)):
                for n, j in enumerate(row_i['SR_ID']):
                    if i != j: # Avoid self-loops

                        # Area-weighted links
                        if weight_by == 'area':
                            sr_total_ij = row_i['SR_area'][n] * SQM_TO_SQFT
                        
                        # SR links
                        else:
                            # SR_owner gives SR value of owner i's specific parcel share given "n" neighbors
                            # Hence: SR_total_ij includes all SR_ij summed together
                            sr_owner_ij = row_i['SR_owner']
                            sr_total_ij = sr_owner_ij[n]

                        if sr_total_ij is not None:
                            if sr_total_ij == 'nan' or None:
                                sr_total_ij = 0
                            G.add_edge(i, j, SR=sr_total_ij)

        elif edge_variable == 'OR':
            # OR (IN direction --> Find area on owner i's parcel)
            if isinstance(row_i['owed_ID'], (list, tuple, np.ndarray)):
                for n, k in enumerate(row_i['owed_ID']):
                    if i != k: # Avoid self-loops

                        # Area-weighted links
                        or_ij = row_i['owed_overlap_area'][n] * SQM_TO_SQFT if weight_by == 'area' else row_i['OR'][n]

                        # OR_in links
                        if or_ij is None or (isinstance(or_ij, float) and math.isnan(or_ij)):
                            or_ij = 0
                        # Link points to whoever owes whom (i.e., Row i's owed_ID/OR is in the IN direction)
                        # so "i" is the one who owes (i's DS does not cover) and arrow directs to "i" 
                        # Formally: D_k \cap t_i \ D_i (DS of neighbor K overlaps with parcel of owner i but NOT with DS of owner i)
                        G.add_edge(k, i, OR=or_ij)

        elif edge_variable == 'OR_out':
            # OR in OUT direction (i.e., neighbor owes it)
            if isinstance(row_i['owed_ID_out'], (list, tuple, np.ndarray)):
                for n, k in enumerate(row_i['owed_ID_out']):
                    if i != k: # Avoid self-loops

                        # Area-weighted links
                        or_ij = row_i['owed_overlap_area_out'][n] * SQM_TO_SQFT if weight_by == 'area' else row_i['OR_out'][n]

                        # OR_out links
                        if or_ij is None or (isinstance(or_ij, float) and math.isnan(or_ij)):
                            or_ij = 0
                        G.add_edge(i, k, OR_out=or_ij)

    return G




def subnetworks(G_edges, variable='SR', directed=False):
    """
    Rebuilds a graph from G_edges using weighted links
    Returns output which has one entry per connected component (subnetwork)
    """

    results_dict = {'subnetwork_nodes': [], 'subnetwork_risk': [], 'n_links': [], 'n_nodes': [], 'total_risk': [], 'avg_risk': []}

    if directed == False:
        G = nx.Graph()
    elif directed == True:
        G = nx.DiGraph()

    for edge0, edge1, value in G_edges.edges(data=variable):
        if variable == 'SR' and value is not None:
            G.add_edge(edge0, edge1, SR=value)
        if variable == 'OR' and value is not None:
            G.add_edge(edge0, edge1, OR=value)

    if directed == False:
        num_components = nx.number_connected_components(G)
        components = list(nx.connected_components(G))
    elif directed == True:
        components = list(nx.weakly_connected_components(G))
        num_components = len(components)

    for i, comp in enumerate(components, 1):
        edge_subgraph = G.subgraph(comp).edges(data=variable)
        edge_weights = [e[2] for e in edge_subgraph]

        results_dict['subnetwork_nodes'].append(comp)
        results_dict['subnetwork_risk'].append(edge_weights)
        results_dict['n_links'].append(len(edge_weights))
        results_dict['n_nodes'].append(len(G.subgraph(comp)))
        results_dict['total_risk'].append(np.nansum(edge_weights))
        results_dict['avg_risk'].append(np.nansum(edge_weights) / len(edge_weights) if edge_weights else 0)

    return G, results_dict


def network_metrics(percolated_graph, variable, directed=False):
    """
    Prepare subnetworks for percolate function
    """
    _, subnetworks_results = subnetworks(percolated_graph, variable=variable, directed=directed)
    dd = pd.DataFrame(subnetworks_results)

    if percolated_graph.is_directed():
        dd = dd[dd['n_nodes'] != 0]
        num_components = len(dd)
    else:
        num_components = nx.number_connected_components(percolated_graph)

    if len(dd) > 0 and len(dd['subnetwork_nodes']) > 0:
        lcc_only = dd.iloc[dd['n_nodes'].argmax()]
        lcc_size = len(lcc_only['subnetwork_risk'])
        lcc_risk = dd.iloc[np.argsort(list(dd['n_nodes']))[0]]
        lcc_risk_size = len(lcc_risk['subnetwork_risk'])
    else:
        lcc_size = 0
        lcc_risk_size = 0

    return dd, lcc_size, lcc_risk_size, num_components


# ---------------------------------------------------------------------------
# 2. Percolation (network link removal) engine
# ---------------------------------------------------------------------------

def percolate_graph(G, removal_probability=None, variable=None, mode=0, num_sample=1, rng=None, state=None):
    """
    Removes one batch of links from network under a removal strategy:
    - mode=0 Random: Links chosen uniformly at random.
    - mode=1 Localized: Randomly select a node then find adjacent links and remove. If links run out, find another node at random
    - mode=2 Targeted: Links removed in order of highest remaining variable value (i.e., SR or OR)

    NOTE:`rng` seed is used to ensure reproducibility
    
    """
    rng = rng or random
    edges_to_remove = []

    if removal_probability:

        # Random
        if mode == 0:
            for edge in G.edges(data=True):
                if rng.uniform(0, 1) < removal_probability:
                    edges_to_remove.append(edge[:2])

        # Localized: select one node at random and remove its links
        elif mode == 1:
            node_to_remove = rng.choice(list(G.nodes()))
            neighbors = list(G.neighbors(node_to_remove))
            edges_to_remove.extend([(node_to_remove, neighbor) for neighbor in neighbors])

        # Targeted: remove links based on descending responsibility value
        elif mode == 2:
            edges_with_val = [(u, v, data) for u, v, data in G.edges(data=True) if variable in data and not pd.isna(data[variable])]
            edges_sorted = sorted(edges_with_val, key=lambda x: x[2][variable], reverse=True)
            num_edges_to_remove = int(removal_probability * len(edges_sorted))
            edges_to_remove.extend(edges_sorted[:num_edges_to_remove])

    else:
        num_sample = int(num_sample)
        if num_sample == 0:
            return G 

        # Random -- uniformly remove links at random
        if mode == 0:
            edges_to_remove_list = list(G.edges())
            n = min(num_sample, len(edges_to_remove_list))
            if n > 0:
                edges_to_remove.extend(rng.sample(edges_to_remove_list, n))

        # Localized -- grow outward from a randomly-seeded NODE 
        # When the local cluster runs out of adjacent edges, reseed with a random node and continue
        elif mode == 1:
            frontier = state.get('frontier', set()) if state is not None else set()

            candidate_edges = [(u, v) for u, v in G.edges() if u in frontier or v in frontier] if frontier else []

            if not candidate_edges:
                all_nodes = list(G.nodes())
                if all_nodes:
                    seed_node = rng.choice(all_nodes)
                    frontier = {seed_node}
                    candidate_edges = [(u, v) for u, v in G.edges() if u == seed_node or v == seed_node]

            n = min(num_sample, len(candidate_edges))
            chosen = rng.sample(candidate_edges, n) if n > 0 else []
            edges_to_remove.extend(chosen)

            for u, v in chosen:
                frontier.add(u)
                frontier.add(v)

            if state is not None:
                state['frontier'] = frontier

        # Targeted: remove the highest-responsibility links
        elif mode == 2:
            edges_with_val = [(u, v, data) for u, v, data in G.edges(data=True) if variable in data and not pd.isna(data[variable])]
            edges_sorted = sorted(edges_with_val, key=lambda x: x[2][variable], reverse=True)
            edges_to_remove.extend(edges_sorted[:num_sample])

        G.remove_edges_from(edges_to_remove)

    return G


def percolate(G_test=None, variable=None, probability_step=0.1, mode=0, interval=None,
              fraction_threshold=None, rng=None, return_details=False):
    """
    Repeatedly applies `percolate_graph` for up to fraction_threshold or 1.0
    Records each step's subnetwork metrics via `network_metrics`. 
    
    return_details: if True, also returns the full per-step subnetwork DataFrame (with subnetwork_nodes)
    """
    rng = rng or random
    state = {} 

    directed = isinstance(G_test, networkx.classes.digraph.DiGraph)

    sub_size = []
    sub_risk = []
    sub_avg_risk = []
    num_connected_components = []
    sub_dfs = [] 

    percolated_graph = percolate_graph(G_test.copy(), removal_probability=None, variable=variable,
                                        mode=mode, num_sample=0, rng=rng, state=state)

    if fraction_threshold:
        threshold = fraction_threshold * (len(G_test.edges(data=variable)))
    else:
        threshold = len(G_test.edges(data=variable))

    n_total_edges = len(G_test.edges(data=variable))

    i = 0
    if interval:
        for _ in range(int(1 / interval) + 1):
            if i > 0:
                percolated_graph = percolate_graph(percolated_graph, variable=variable, removal_probability=None,
                                                     mode=mode, num_sample=interval * n_total_edges, rng=rng, state=state)

            sub_df, lcc_size, lcc_risk_size, num_components = network_metrics(percolated_graph, variable=variable, directed=directed)

            num_connected_components.append(num_components)
            sub_size.append(sub_df['n_nodes'])
            sub_risk.append(sub_df['total_risk'])
            sub_avg_risk.append(sub_df['avg_risk'])
            if return_details:
                sub_dfs.append(sub_df)

            i += 1
    else:
        while i <= threshold:
            if i > 0:
                percolated_graph = percolate_graph(percolated_graph, variable=variable, removal_probability=None,
                                                     mode=mode, num_sample=1, rng=rng, state=state)

            sub_df, lcc_size, lcc_risk_size, num_components = network_metrics(percolated_graph, variable=variable, directed=directed)

            num_connected_components.append(num_components)
            sub_size.append(sub_df['n_nodes'])
            sub_risk.append(sub_df['total_risk'])
            sub_avg_risk.append(sub_df['avg_risk'])
            if return_details:
                sub_dfs.append(sub_df)

            i += 1

    if return_details:
        return sub_size, sub_risk, sub_avg_risk, num_connected_components, sub_dfs
    return sub_size, sub_risk, sub_avg_risk, num_connected_components

#TODO:Check Betweenness method and network metrics
def run_percolation_trials(G_pre=None, variable=None, num_trials=None, step=None, interval=None,
                            fraction_threshold=None, seed=42):
    """
    Runs `percolate` over `num_trials` trials for each of 4 modes (Random, Localized, "High Risk"/Targeted, Betweenness Centrality), 
    Records per-step subnetwork breakdown for each (trial, mode) used for comparisons across repeated trials
    """

    labels = ['Random', 'Localized', 'High Risk', 'Betweenness Centrality']
    subnetwork_metrics = ['sub_size', 'sub_risk', 'sub_avg_risk', 'ncc']
    subnetwork_dict = {metric: {str(mode): [] for mode in range(len(labels))} for metric in subnetwork_metrics}

    threshold = fraction_threshold
    for trial in range(num_trials):
        trial_rng = random.Random(seed + trial)
        for mode in range(len(labels)):
            sub_size, sub_risk, sub_avg_risk, ncc = percolate(
                G_test=G_pre.copy(), variable=variable, probability_step=step, mode=mode,
                interval=interval, fraction_threshold=threshold, rng=trial_rng
            )
            subnetwork_dict['sub_risk'][str(mode)].append(sub_risk)
            subnetwork_dict['sub_avg_risk'][str(mode)].append(sub_avg_risk)
            subnetwork_dict['ncc'][str(mode)].append(ncc)

    return subnetwork_dict



# Track SR or OR by subnetwork
def assign_lineage(prev_node_map, current_components, next_new_id):
    """
    Matches each current connected component to the subnetwork ID it descended from
    To enable this --> The largest child of a split keeps the parent's ID ("Original"); smaller splits get new IDs ("New").
    """

    component_info = []
    parent_groups = {}

    for comp in current_components:
        rep_node = next(iter(comp))
        parent_id = prev_node_map.get(rep_node, -1)
        parent_groups.setdefault(parent_id, []).append(comp)

    for parent_id, children_components in parent_groups.items():
        children_components.sort(key=len, reverse=True)
        largest_child = children_components[0]
        component_info.append({'nodes': largest_child, 'id': parent_id, 'type': 'Original'})

        for split_off in children_components[1:]:
            component_info.append({'nodes': split_off, 'id': next_new_id, 'type': 'New'})
            next_new_id += 1

    return component_info, next_new_id


def get_component_risk(G, component_nodes, variable):
    """Total (summed) responsibility of the links within one component."""
    subg = G.subgraph(component_nodes)
    edge_weights = nx.get_edge_attributes(subg, variable)
    if edge_weights:
        return np.nansum(list(edge_weights.values()))
    return 0


def run_simulation(G_pre, variable='SR', interval=0.1, seed=42):
    """
    Runs percolation via `percolate` for each removal mode (Random, Localized, Targeted), 
    Record layers subnetwork lineage
    
    Returns {mode_name: {'data': DataFrame, 'lifecycle': dict}}.
    """
    modes = {0: 'Random', 1: 'Localized', 2: 'Targeted'}
    results = {}

    risk_col = f'total_{variable.lower()}'

    for mode_code, mode_name in modes.items():
        mode_rng = random.Random(seed)

        sub_size, sub_risk, sub_avg_risk, ncc, sub_dfs = percolate(
            G_test=G_pre, variable=variable, mode=mode_code, interval=interval,
            fraction_threshold=1.0, rng=mode_rng, return_details=True
        )

        tracking_data = []
        subnetwork_lifecycle = {}
        next_new_id = 1
        prev_node_map = {}

        for step, step_df in enumerate(sub_dfs):
            fraction = min(1.0, step * interval)
            curr_comps = [frozenset(nodes) for nodes in step_df['subnetwork_nodes']]
            risk_by_comp = {frozenset(row['subnetwork_nodes']): row['total_risk'] for _, row in step_df.iterrows()}

            if step == 0:
                comps_sorted = sorted(curr_comps, key=len, reverse=True)
                new_node_map = {}
                for comp in comps_sorted:
                    cid = next_new_id
                    next_new_id += 1
                    subnetwork_lifecycle[cid] = {'birth_step': 0, 'death_step': None, 'type': 'Original'}
                    tracking_data.append({'step': 0, 'fraction': 0.0, 'id': cid, risk_col: risk_by_comp[comp], 'type': 'Original'})
                    for node in comp:
                        new_node_map[node] = cid
                prev_node_map = new_node_map
                continue

            comp_infos, next_new_id = assign_lineage(prev_node_map, curr_comps, next_new_id)
            new_node_map = {}

            for info in comp_infos:
                cid = info['id']
                comp = frozenset(info['nodes'])
                risk = risk_by_comp[comp]

                if cid not in subnetwork_lifecycle:
                    subnetwork_lifecycle[cid] = {'birth_step': step, 'death_step': None, 'type': info['type']}

                tracking_data.append({
                    'step': step,
                    'fraction': fraction,
                    'id': cid,
                    risk_col: risk,
                    'type': info['type']
                })

                for node in comp:
                    new_node_map[node] = cid
            prev_node_map = new_node_map

        last_recorded_step = tracking_data[-1]['step'] if tracking_data else 0
        for cid, lifecycle in subnetwork_lifecycle.items():
            if lifecycle['death_step'] is None:
                lifecycle['death_step'] = last_recorded_step

        results[mode_name] = {
            'data': pd.DataFrame(tracking_data),
            'lifecycle': subnetwork_lifecycle
        }

    return results


def run_simulation_multiseed(G_pre, variable, interval, seeds=range(1, 11)):

    risk_col = f'total_{variable.lower()}'
    rows = []
    for seed in seeds:
        results = run_simulation(G_pre=G_pre, variable=variable, interval=interval, seed=seed)
        for mode_name, mode_result in results.items():
            df = mode_result['data']
            total_by_fraction = df.groupby('fraction')[risk_col].sum().reset_index()
            total_by_fraction['mode'] = mode_name
            total_by_fraction['seed'] = seed
            rows.append(total_by_fraction)
    return pd.concat(rows, ignore_index=True)