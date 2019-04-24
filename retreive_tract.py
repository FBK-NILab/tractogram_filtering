#!/usr/bin/env python
import sys
import os
import argparse
import ConfigParser
import numpy as np
import nibabel as nib
import glob
from time import time
import pickle

from functools import partial
from dipy.tracking.distances import bundles_distances_mam
from sklearn.neighbors import KDTree
from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)
from dipy.tracking.utils import streamline_near_roi
from dipy.core.geometry import dist_to_corner
import dipy.tracking.utils as ut
from nibabel.affines import apply_affine

try:
    from joblib import Parallel, delayed, cpu_count
    joblib_available = True
except:
    joblib_available = False

def compute_superset(true_bundle, kdt, prototypes, k=1000, distance_func=bundles_distances_mam, eps=10e-8):
    """Compute a superset of the true target tract with k-NN.
    """
    true_bundle = np.array(true_bundle, dtype=np.object)
    dm_true_bundle = distance_func(true_bundle, prototypes)
    D, I = kdt.query(dm_true_bundle, k=k)
    superset_idx = np.unique(I.flat)
    # recompute the gt streamlines idx
    I = kdt.query_radius(dm_true_bundle, eps)
    gt_idx = []
    for arr in I:
        for e in arr:
            gt_idx.append(e)
    gt_idx = list(set(gt_idx))
    print('gt length: %d' % len(gt_idx))
    additional_idx = list(set(superset_idx) - set(gt_idx))

    return D, additional_idx, gt_idx

def compute_kdt_and_dr(tract, num_prototypes=None):
    """Compute the dissimilarity representation of the tract and
    build the kd-tree.
    """
    tract = np.array(tract, dtype=np.object)
    print("Computing dissimilarity matrices...")
    if num_prototypes is None:
        num_prototypes = 40
        print("Using %s prototypes as in Olivetti et al. 2012."
              % num_prototypes)
    else:
        print("Using %s prototypes" % num_prototypes)
    t0 = time()
    distance = partial(parallel_distance_computation,
                            distance=bundles_distances_mam)
    dm_tract, prototype_idx = compute_dissimilarity(tract,
                                                    distance,
                                                    num_prototypes,
                                                    prototype_policy='sff',
                                                    verbose=False)
    print("%s sec." % (time() - t0))
    prototypes = tract[prototype_idx]
    print("Building the KD-tree of tract.")
    kdt = KDTree(dm_tract)
    return kdt, prototypes


def compute_dissimilarity(dataset, distance, k,
                          prototype_policy='sff', verbose=False):
    """Compute the dissimilarity (distance) matrix between a dataset of N
    objects and prototypes, where prototypes are selected among the
    objects with a given policy.

    Parameters
    ----------
    dataset : list or array of objects
           an iterable of objects.
    distance : function
           Distance function between groups of objects or sets of objects.
    k : int
           The number of prototypes/landmarks.
    prototype_policy : string
           The prototype selection policy. The default value is 'sff',
           which is highly scalable.
    verbose : bool
           If true prints some messages. Deafault is True.

    Return
    ------
    dissimilarity_matrix : array (N, k)

    See Also
    --------
    subsampling.furthest_first_traversal,
    subsampling.subset_furthest_first

    Notes
    -----

    """
    if verbose:
        print("Generating %s prototypes with policy %s." % (k, prototype_policy))

    prototype_idx = compute_subset(dataset, distance, k,
                                   landmark_policy=prototype_policy)
    prototypes = [dataset[i] for i in prototype_idx]
    dissimilarity_matrix = distance(dataset, prototypes)
    return dissimilarity_matrix, prototype_idx



def euclidean_distance(A, B):
    """Wrapper of the euclidean distance between two vectors, or array and
    vector, or two arrays.
    """
    return distance_matrix(np.atleast_2d(A), np.atleast_2d(B), p=2)


def parallel_distance_computation(A, B, distance, n_jobs=-1,
                                  granularity=2, verbose=False,
                                  job_size_min=1000):
    """Computes the distance matrix between all objects in A and all
    objects in B in parallel over all cores.

    This function can be partially instantiated with a given distance,
    in order to obtain a the parallel version of a distance function
    with the same signature as the distance function. Example:
    distance_parallel = functools.partial(parallel_distance_computation, distance=distance)
    """
    if (len(A) > job_size_min) and joblib_available and (n_jobs != 1):
        if n_jobs is None or n_jobs == -1:
            n_jobs = cpu_count()

        if verbose:
            print("Parallel computation of the distance matrix: %s cpus." % n_jobs)

        if n_jobs > 1:
            tmp = np.linspace(0, len(A), granularity * n_jobs + 1).astype(np.int)
        else:  # corner case: joblib detected 1 cpu only.
            tmp = (0, len(A))

        chunks = zip(tmp[:-1], tmp[1:])
        dissimilarity_matrix = np.vstack(Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(distance)(A[start:stop], B) for start, stop in chunks))
    else:
        dissimilarity_matrix = distance(A, B)

    if verbose:
        print("Done.")

    return dissimilarity_matrix


def furthest_first_traversal(dataset, k, distance, permutation=True):
    """This is the farthest first traversal (fft) algorithm which selects
    k objects out of an array of objects (dataset). This algorithms is
    known to be a good sub-optimal solution to the k-center problem,
    i.e. the k objects are sequentially selected in order to be far
    away from each other.

    Parameters
    ----------

    dataset : array of objects
        an iterable of objects which supports advanced indexing.
    k : int
        the number of objects to select.
    distance : function
        a distance function between two objects or groups of objects,
        that given two groups as input returns the distance or distance
        matrix.
    permutation : bool
        True if you want to shuffle the objects first. No
        side-effect on the input dataset.

    Return
    ------
    idx : array of int
        an array of k indices of the k selected objects.

    Notes
    -----
    - Hochbaum, Dorit S. and Shmoys, David B., A Best Possible
    Heuristic for the k-Center Problem, Mathematics of Operations
    Research, 1985.
    - http://en.wikipedia.org/wiki/Metric_k-center

    See Also
    --------
    subset_furthest_first

    """
    if permutation:
        idx = np.random.permutation(len(dataset))
        dataset = dataset[idx]
    else:
        idx = np.arange(len(dataset), dtype=np.int)

    T = [0]
    while len(T) < k:
        z = distance(dataset, dataset[T]).min(1).argmax()
        T.append(z)

    return idx[T]


def subset_furthest_first(dataset, k, distance, permutation=True, c=2.0):
    """The subset furthest first (sff) algorithm is a stochastic
    version of the furthest first traversal (fft) algorithm. Sff
    scales well on large set of objects (dataset) because it
    does not depend on len(dataset) but only on k.

    Parameters
    ----------

    dataset : list or array of objects
        an iterable of objects.
    k : int
        the number of objects to select.
    distance : function
        a distance function between groups of objects, that given two
        groups as input returns the distance matrix.
    permutation : bool
        True if you want to shuffle the objects first. No
        side-effect.
    c : float
        Parameter to tune the probability that the random subset of
        objects is sufficiently representive of dataset. Typically
        2.0-3.0.

    Return
    ------
    idx : array of int
        an array of k indices of the k selected objects.

    See Also
    --------
    furthest_first_traversal

    Notes
    -----
    See: E. Olivetti, T.B. Nguyen, E. Garyfallidis, The Approximation
    of the Dissimilarity Projection, Proceedings of the 2012
    International Workshop on Pattern Recognition in NeuroImaging
    (PRNI), pp.85,88, 2-4 July 2012 doi:10.1109/PRNI.2012.13
    """
    size = compute_subsample_size(k, c=c)
    if permutation:
        idx = np.random.permutation(len(dataset))[:size]
    else:
        idx = range(size)

    return idx[furthest_first_traversal(dataset[idx],
                                        k, distance,
                                        permutation=False)]


def compute_subsample_size(n_clusters, c=2.0):
    """Compute a subsample size that takes into account a possible cluster
    structure of the dataset, in n_clusters, based on a solution of
    the coupon collector's problem, i.e. k*log(k).

    Notes
    -----
    See: E. Olivetti, T.B. Nguyen, E. Garyfallidis, The Approximation
    of the Dissimilarity Projection, Proceedings of the 2012
    International Workshop on Pattern Recognition in NeuroImaging
    (PRNI), pp.85,88, 2-4 July 2012 doi:10.1109/PRNI.2012.13

    """
    return int(max(1, np.ceil(c * n_clusters * np.log(n_clusters))))



def compute_subset(dataset, distance, num_landmarks,
                   landmark_policy='sff'):
    """Wrapper code to dispatch the computation of the subset according to
    the required policy.
    """
    if landmark_policy == 'random':
        landmark_idx = np.random.permutation(len(dataset))[:num_landmarks]
    elif landmark_policy in ('fft', 'minmax'):
        landmark_idx = furthest_first_traversal(dataset,
                                                 num_landmarks, distance)
    elif landmark_policy == 'sff':
        landmark_idx = subset_furthest_first(dataset, num_landmarks, distance)
    else:
        if verbose:
            print("Landmark selection policy not supported: %s" % landmark_policy)

        raise Exception

    return landmark_idx


def target_mod(streamlines, target_mask, affine, include=True):
    """Filters streamlines based on whether or not they pass through an ROI.
    Parameters
    ----------
    streamlines : iterable
        A sequence of streamlines. Each streamline should be a (N, 3) array,
        where N is the length of the streamline.
    target_mask : array-like
        A mask used as a target. Non-zero values are considered to be within
        the target region.
    affine : array (4, 4)
        The affine transform from voxel indices to streamline points.
    include : bool, default True
        If True, streamlines passing through `target_mask` are kept. If False,
        the streamlines not passing through `target_mask` are kept.
    Returns
    -------
    streamlines : generator
        A sequence of streamlines that pass through `target_mask`.
    Raises
    ------
    ValueError
        When the points of the streamlines lie outside of the `target_mask`.
    See Also
    --------
    density_map
    """
    target_mask = np.array(target_mask, dtype=bool, copy=True)
    lin_T, offset = _mapping_to_voxel(affine, voxel_size=None)
    #yield
    # End of initialization

    ids = []
    for sl in range(len(streamlines)):
        try:
            ind = _to_voxel_coordinates(streamlines[sl], lin_T, offset)
            i, j, k = ind.T
            state = target_mask[i, j, k]
        except IndexError:
            raise ValueError("streamlines points are outside of target_mask")
        if state.any() == include:
            ids.append(sl)

    return ids



def select_by_rois_mod(streamlines, rois, include, mode=None, affine=None,
                   tol=None):
    """Select streamlines based on logical relations with several regions of
    interest (ROIs). For example, select streamlines that pass near ROI1,
    but only if they do not pass near ROI2.
    Parameters
    ----------
    streamlines : list
        A list of candidate streamlines for selection
    rois : list or ndarray
        A list of 3D arrays, each with shape (x, y, z) corresponding to the
        shape of the brain volume, or a 4D array with shape (n_rois, x, y,
        z). Non-zeros in each volume are considered to be within the region
    include : array or list
        A list or 1D array of boolean values marking inclusion or exclusion
        criteria. If a streamline is near any of the inclusion ROIs, it
        should evaluate to True, unless it is also near any of the exclusion
        ROIs.
    mode : string, optional
        One of {"any", "all", "either_end", "both_end"}, where a streamline is
        associated with an ROI if:
        "any" : any point is within tol from ROI. Default.
        "all" : all points are within tol from ROI.
        "either_end" : either of the end-points is within tol from ROI
        "both_end" : both end points are within tol from ROI.
    affine : ndarray
        Affine transformation from voxels to streamlines. Default: identity.
    tol : float
        Distance (in the units of the streamlines, usually mm). If any
        coordinate in the streamline is within this distance from the center
        of any voxel in the ROI, the filtering criterion is set to True for
        this streamline, otherwise False. Defaults to the distance between
        the center of each voxel and the corner of the voxel.
    Notes
    -----
    The only operation currently possible is "(A or B or ...) and not (X or Y
    or ...)", where A, B are inclusion regions and X, Y are exclusion regions.
    Returns
    -------
    generator
       Generates the streamlines to be included based on these criteria.
    See also
    --------
    :func:`dipy.tracking.utils.near_roi`
    :func:`dipy.tracking.utils.reduce_rois`
    Examples
    --------
    >>> streamlines = [np.array([[0, 0., 0.9],
    ...                          [1.9, 0., 0.]]),
    ...                np.array([[0., 0., 0],
    ...                          [0, 1., 1.],
    ...                          [0, 2., 2.]]),
    ...                np.array([[2, 2, 2],
    ...                          [3, 3, 3]])]
    >>> mask1 = np.zeros((4, 4, 4), dtype=bool)
    >>> mask2 = np.zeros_like(mask1)
    >>> mask1[0, 0, 0] = True
    >>> mask2[1, 0, 0] = True
    >>> selection = select_by_rois(streamlines, [mask1, mask2],
    ...                            [True, True],
    ...                            tol=1)
    >>> list(selection) # The result is a generator
    [array([[ 0. ,  0. ,  0.9],
           [ 1.9,  0. ,  0. ]]), array([[ 0.,  0.,  0.],
           [ 0.,  1.,  1.],
           [ 0.,  2.,  2.]])]
    >>> selection = select_by_rois(streamlines, [mask1, mask2],
    ...                            [True, False],
    ...                            tol=0.87)
    >>> list(selection)
    [array([[ 0.,  0.,  0.],
           [ 0.,  1.,  1.],
           [ 0.,  2.,  2.]])]
    >>> selection = select_by_rois(streamlines, [mask1, mask2],
    ...                            [True, True],
    ...                            mode="both_end",
    ...                            tol=1.0)
    >>> list(selection)
    [array([[ 0. ,  0. ,  0.9],
           [ 1.9,  0. ,  0. ]])]
    >>> mask2[0, 2, 2] = True
    >>> selection = select_by_rois(streamlines, [mask1, mask2],
    ...                            [True, True],
    ...                            mode="both_end",
    ...                            tol=1.0)
    >>> list(selection)
    [array([[ 0. ,  0. ,  0.9],
           [ 1.9,  0. ,  0. ]]), array([[ 0.,  0.,  0.],
           [ 0.,  1.,  1.],
           [ 0.,  2.,  2.]])]
    """
    if affine is None:
        affine = np.eye(4)
    # This calculates the maximal distance to a corner of the voxel:
    dtc = dist_to_corner(affine)
    if tol is None:
        tol = dtc
    elif tol < dtc:
        w_s = "Tolerance input provided would create gaps in your"
        w_s += " inclusion ROI. Setting to: %s" % dist_to_corner
        warn(w_s)
        tol = dtc
    include_roi, exclude_roi = ut.reduce_rois(rois, include)
    include_roi_coords = np.array(np.where(include_roi)).T
    x_include_roi_coords = apply_affine(affine, include_roi_coords)
    exclude_roi_coords = np.array(np.where(exclude_roi)).T
    x_exclude_roi_coords = apply_affine(affine, exclude_roi_coords)

    if mode is None:
        mode = "any"
    ids = []
    for i in range(len(streamlines)):
        include = streamline_near_roi(streamlines[i], x_include_roi_coords, tol=tol,
                                      mode=mode)
        exclude = streamline_near_roi(streamlines[i], x_exclude_roi_coords, tol=tol,
                                      mode=mode)
        if include & ~exclude:
                ids.append(i)

    return ids

