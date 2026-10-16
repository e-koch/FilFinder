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
