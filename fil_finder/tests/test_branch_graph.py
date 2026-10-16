import numpy as np
import astropy.units as u

from ..filament import Filament2D

# Skeleton with a long main path and several side branches. See
# test_Filament2D_multibranch for the pruned version.
PIXELS = (np.array([-1, 0, 2, 1, 4, 2, 3, 1, 4, 5, 1, 1, 1]) + 2,
          np.array([-1, 0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 5, 6]) + 2)


def make_filament(prune=True):
    image = np.zeros((8, 9))
    image[PIXELS] = 2.

    fil = Filament2D(PIXELS)
    if prune:
        fil.skeleton_analysis(image, prune_criteria='length',
                              branch_thresh=1 * u.pix)
    else:
        fil.skeleton_analysis(image, prune_criteria='length',
                              branch_thresh=0 * u.pix)
    return fil, image


def test_edges_store_branch_info():

    for prune in [True, False]:
        fil, image = make_filament(prune=prune)

        assert fil.graph.number_of_edges() > 0

        for a, b, data in fil.graph.edges(data=True):
            assert set(['weight', 'labels', 'length', 'intensity']) <= set(data)
            assert len(data['labels']) >= 1
            assert data['length'] > 0

        # No pruning/loops here, so every branch is on exactly one edge
        edge_len = sum(d['length'] for _, _, d in fil.graph.edges(data=True))
        assert np.isclose(edge_len, fil.branch_properties['length'].sum().value)

        edge_labels = [lab for _, _, d in fil.graph.edges(data=True)
                       for lab in d['labels']]
        assert len(edge_labels) == len(set(edge_labels))


def test_nodes_store_positions():

    for prune in [True, False]:
        fil, image = make_filament(prune=prune)
        pts = np.vstack(fil.branch_pts())

        for node, data in fil.graph.nodes(data=True):
            assert 'pos' in data
            dist = np.sqrt(((pts - np.array(data['pos'])) ** 2).sum(axis=1))
            if isinstance(node, str):
                # Intersection pixels are removed from the branches, so the
                # node sits next to (not on) them.
                assert dist.min() <= 2
            else:
                # End nodes sit on their branch
                assert dist.min() == 0


def test_merge_nodes_combines_branch_info():
    import networkx as nx
    from ..length import merge_nodes

    G = nx.Graph()
    G.add_edge(1, 'A', weight=0.2, labels=[1], length=2., intensity=1.)
    G.add_edge('A', 2, weight=0.3, labels=[5], length=6., intensity=3.)
    G.add_edge('A', 3, weight=0.5, labels=[7], length=1., intensity=1.)
    G.add_edge(3, 'B', weight=0.1, labels=[2], length=4., intensity=2.)
    G.add_edge('B', 4, weight=0.1, labels=[3], length=4., intensity=4.)

    # A has three neighbors: nothing to merge
    merge_nodes('A', G)
    assert 'A' in G

    G = merge_nodes('B', G)
    assert 'B' not in G
    edge = G[3][4]
    assert np.isclose(edge['weight'], 0.2)
    assert edge['length'] == 8.
    assert np.isclose(edge['intensity'], 3.)
    assert sorted(edge['labels']) == [2, 3]
    assert 'displaced' not in edge


def test_merge_nodes_without_branch_info():
    import networkx as nx
    from ..length import merge_nodes

    G = nx.Graph()
    G.add_edge(1, 'A', weight=0.2)
    G.add_edge('A', 2, weight=0.3)

    G = merge_nodes('A', G)
    assert np.isclose(G[1][2]['weight'], 0.5)
    assert 'labels' not in G[1][2]


def test_merge_nodes_keeps_displaced_edge():
    import networkx as nx
    from ..length import merge_nodes

    G = nx.Graph()
    G.add_edge(1, 2, weight=1., labels=[9], length=1., intensity=1.)
    G.add_edge(1, 'A', weight=1., labels=[1], length=1., intensity=1.)
    G.add_edge('A', 2, weight=1., labels=[2], length=1., intensity=1.)

    G = merge_nodes('A', G)
    assert G[1][2]['labels'] == [1, 2]
    assert G[1][2]['displaced']['labels'] == [9]


def test_edge_indices_match_branch_properties():

    for prune in [True, False]:
        fil, image = make_filament(prune=prune)

        assert len(fil.branch_labels) == fil.branch_properties['number']
        assert list(fil.branch_label_map.values()) == \
            list(range(fil.branch_properties['number']))

        for a, b, data in fil.graph.edges(data=True):
            assert len(data['indices']) == len(data['labels'])
            lengths = fil.branch_properties['length'][data['indices']].value
            assert np.isclose(lengths.sum(), data['length'])
            for idx, label in zip(data['indices'], data['labels']):
                assert fil.branch_labels[idx] == label


def test_edge_indices_with_pruned_branches():
    # Pruning removes branches; label - 1 is then no longer the index.
    fil, image = make_filament(prune=False)
    fil_pruned, image = make_filament(prune=True)

    assert fil_pruned.branch_properties['number'] < \
        fil.branch_properties['number']

    for a, b, data in fil_pruned.graph.edges(data=True):
        idx = data['indices'][0]
        pts = fil_pruned.branch_pts()[idx]
        assert len(pts) > 0
        assert np.isclose(fil_pruned.branch_properties['length'][idx].value,
                          data['length'])


def test_edge_orientation_matches_rht():

    for prune in [True, False]:
        fil, image = make_filament(prune=prune)
        fil.rht_branch_analysis(radius=3 * u.pix, min_branch_length=0 * u.pix)

        for a, b, data in fil.graph.edges(data=True):
            assert len(data['indices']) == 1
            idx = data['indices'][0]
            assert np.isclose(data['orientation'],
                              fil.orientation_branches[idx].value,
                              equal_nan=True)
            assert np.isclose(data['curvature'],
                              fil.curvature_branches[idx].value,
                              equal_nan=True)


def test_merged_edge_orientation():
    # Orientation of a merged edge is a length-weighted mean mod pi
    fil, image = make_filament(prune=True)
    fil.rht_branch_analysis(radius=3 * u.pix, min_branch_length=0 * u.pix)

    # Pretend two branches were merged into one edge
    edge = fil.graph.edges[next(iter(fil.graph.edges))]
    edge['indices'] = [0, 0]
    edge['labels'] = [1, 1]
    fil._update_graph_orientation()

    assert np.isclose(edge['orientation'] % np.pi,
                      fil.orientation_branches[0].value % np.pi)


def test_junction_orientations():

    fil, image = make_filament(prune=True)

    angles = fil.junction_orientations('A', radius=3 * u.pix)
    assert set(angles) == set(fil.graph['A'])

    # Arms leave the hub at (4, 4) along the diagonal, along the row, and
    # along the diagonal.
    expected = {1: np.pi / 4, 2: 0., 3: np.pi / 4}
    for node, angle in angles.items():
        assert np.isclose(angle, expected[int(node)])

    # Too small a radius: no pixels to fit
    angles = fil.junction_orientations('A', radius=0.1 * u.pix)
    assert np.isnan(list(angles.values())).all()
