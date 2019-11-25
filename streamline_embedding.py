import numpy as np
from dipy.tracking.distances import bundles_distances_mam
from euclidean_embeddings.dissimilarity import compute_dissimilarity
from euclidean_embeddings.distances import parallel_distance_computation
from functools import partial
from dipy.tracking.streamline import length
from dipy.tracking.metrics import frenet_serret


def check_same_nb_points(streamlines):
    """Check that all streamlines have the same number of points.
    """
    assert(len(np.unique([len(s) for s in streamlines])) == 1)


def embed_flattened(streamlines):
    """Basic embedding of streamlines: just flattening all coordinates in
    a vector.
    """
    check_same_nb_points(streamlines)
    X_flattened = np.array([s.flatten() for s in streamlines])
    return X_flattened


def embed_flipped_flattened(streamlines):
    """Flip the order of points of every streamline and create a flattened
    embedding with them.
    """
    X_flipped_flattened = np.array([s[::-1].flatten() for s in streamlines])
    return X_flipped_flattened


def embed_flattened_plus_flipped(streamlines):
    """Embedding of streamlines: flattened plus flipped-flattened.

    Note: the number of rows of the resulting matrix is twice the
    number of streamlines.
    """
    check_same_nb_points(streamlines)
    return np.vstack([embed_flattened(streamlines),
                      embed_flipped_flattened(streamlines)])


def embed_ordered(streamlines):
    """Embedding of streamlines, by sorting all coordinates.
    """
    check_same_nb_points(streamlines)
    X_ordered = np.array([np.sort(s, axis=0).flatten() for s in streamlines])
    return X_ordered


def embed_dissimilarity(streamlines, distance=bundles_distances_mam,
                        k=100, parallel=True):
    if not isinstance(streamlines, np.ndarray):
        streamlines = np.array(streamlines, dtype=np.object)

    if parallel:
        distance = partial(parallel_distance_computation, distance=distance)

    X_dissimilarity = compute_dissimilarity(streamlines,
                                            distance=distance, k=k)
    return X_dissimilarity


def embed_flattened_plus_length(streamlines):
    lengths = length(streamlines)
    X = embed_flattened(streamlines)
    X = np.concatenate([X, lengths[:, None]], axis=1)
    return X


def embed_flattened_plus_flipped_plus_length(streamlines):
    lengths = length(streamlines)
    X = embed_flattened_plus_flipped(streamlines)
    X = np.concatenate([X, np.concatenate([lengths, lengths])[:, None]],
                       axis=1)
    return X


def embed_flattened_plus_flipped_plus_length_plus_curvature(streamlines):
    lengths = length(streamlines)
    # Mean curvature of the streamline (TO BE CHECKED!):
    # curvature = np.vstack([np.linalg.norm(np.gradient(s, axis=0), axis=0) for s in streamlines])
    curvature = np.array([frenet_serret(s)[3].mean() for s in streamlines])[:, None]
    torsion = np.array([frenet_serret(s)[4].mean() for s in streamlines])[:, None]
    X = embed_flattened_plus_flipped(streamlines)
    X = np.concatenate([X, np.concatenate([lengths, lengths])[:, None],
                        np.vstack([curvature, curvature[::-1]]),
                        np.vstack([torsion, torsion[::-1]])], axis=1)
    return X


if __name__ == '__main__':

    from nilab.load_trk import load_streamlines
    from dipy.tracking.streamline import set_number_of_points
    from dipy.tracking.distances import bundles_distances_mdf
    from time import time

    trk_fn = 'sub-100206_var-FNAL_tract.trk'
    nb_points = 32
    nb_prototypes = 100
    streamline_distance = bundles_distances_mdf

    streamlines, header, lengths, idxs = load_streamlines(trk_fn,
                                                          idxs=None,
                                                          apply_affine=True,
                                                          container='list',
                                                          verbose=True)
    print("Resampling to %s points" % nb_points)
    streamlines = set_number_of_points(streamlines, nb_points)
    # streamlines = np.array(set_number_of_points(streamlines, nb_points),
    #                        dtype=np.object)

    print("embed_flattened():")
    t0 = time()
    X_flattened = embed_flattened(streamlines)
    print("%s sec." % (time() - t0))

    print("embed_flattened_plus_flipped():")
    t0 = time()
    X_flattened_plus_flipped = embed_flattened_plus_flipped(streamlines)
    print("%s sec." % (time() - t0))

    print("embed_ordered():")
    t0 = time()
    X_ordered = embed_ordered(streamlines)
    print("%s sec." % (time() - t0))

    print("embed_dissimilarity():")
    t0 = time()
    X_dissimilarity, prototype_idx = embed_dissimilarity(streamlines,
                                                         distance=streamline_distance,
                                                         k=nb_prototypes)
    print("%s sec." % (time() - t0))
