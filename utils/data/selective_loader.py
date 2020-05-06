import sys
import nibabel as nib
import os
import struct
import numpy as np
from time import time
from nibabel.affines import apply_affine
from nibabel.streamlines.trk import get_affine_trackvis_to_rasmm

def load_selected_streamlines(trk_fn, idxs=None, return_scalars=False):

    lazy_trk = nib.streamlines.load(trk_fn, lazy_load=True)
    header = lazy_trk.header
    header_size = header['hdr_size']
    nb_streamlines = header['nb_streamlines']
    n_scalars = header['nb_scalars_per_point']
    n_properties = header['nb_properties_per_streamline']
    if idxs is None:
        idxs = np.arange(nb_streamlines)
        
    ## See: http://www.trackvis.org/docs/?subsect=fileformat
    length_bytes = 4
    point_size = 3 + n_scalars
    point_bytes = 4 * point_size
    properties_bytes = n_properties * 4
    
    lengths = np.empty(nb_streamlines, dtype=np.int)
    if float(sys.version[:3]) > 3.2:
        with open(trk_fn, 'rb') as f:
            f.seek(header_size)
            for idx in range(nb_streamlines):
                l = int.from_bytes(f.read(4), byteorder='little')
                lengths[idx] = l
                jump = point_bytes * l + properties_bytes
                f.seek(jump, 1)
    else:
        with open(trk_fn, 'rb') as f:
            f.seek(header_size)
            for idx in range(nb_streamlines):
                l = np.fromfile(f, np.int32, 1)[0]
                lengths[idx] = l
                jump = point_bytes * l + properties_bytes
                f.seek(jump, 1)

    # position in bytes where to find a given streamline in the TRK file:
    index_bytes = lengths * point_bytes + properties_bytes + length_bytes
    index_bytes = np.concatenate([[length_bytes], index_bytes[:-1]]).cumsum() + header_size

    n_floats = lengths * point_size  # better because it skips properties, if they exist
    streams = np.empty((lengths[idxs].sum(), 3), dtype=np.float32)
    scalars = np.empty((lengths[idxs].sum(), n_scalars), dtype=np.float32) if n_scalars > 0 else None
    j = 0
    with open(trk_fn, 'rb') as f:
        for idx in idxs:
            # move to the position initial position of the coordinates
            # of the streamline:
            f.seek(index_bytes[idx])
            # Parse the floats:
            s = np.fromfile(f, np.float32, n_floats[idx])
            s.resize(lengths[idx], point_size)

            if n_scalars > 0:
                scalars[j:j+lengths[idx], :] = s[:, 3:]
                s = s[:, :3]

            streams[j:j+lengths[idx], :] = s
            j += lengths[idx]

    # apply affine
    aff = get_affine_trackvis_to_rasmm(lazy_trk.header)
    streams = apply_affine(aff, streams)
   
    if return_scalars:
        streams = np.hstack((streams, scalars))   
        
    return streams, lengths[idxs]

def load_selected_streamlines_uniform_size(trk_fn, idxs=None, return_scalars=False):

    lazy_trk = nib.streamlines.load(trk_fn, lazy_load=True)
    header = lazy_trk.header
    header_size = header['hdr_size']
    nb_streamlines = header['nb_streamlines']
    n_scalars = header['nb_scalars_per_point']
    n_properties = header['nb_properties_per_streamline']
    if idxs is None:
        idxs = np.arange(nb_streamlines)

    ## See: http://www.trackvis.org/docs/?subsect=fileformat
    length_bytes = 4
    point_size = 3 + n_scalars
    point_bytes = 4 * point_size
    properties_bytes = n_properties * 4

    with open(trk_fn, 'rb') as f:
        f.seek(header_size)
        l = np.fromfile(f, np.int32, 1)[0]

    lengths = np.array([l] * nb_streamlines).astype(np.int)

     # position in bytes where to find a given streamline in the TRK file:
    index_bytes = lengths * point_bytes + properties_bytes + length_bytes
    index_bytes = np.concatenate([[length_bytes], index_bytes[:-1]]).cumsum() + header_size

    n_floats = lengths * point_size  # better because it skips properties, if they exist
    streams = np.empty((lengths[idxs].sum(), 3), dtype=np.float32)
    scalars = np.empty((lengths[idxs].sum(), n_scalars), dtype=np.float32) if n_scalars > 0 else None
    j = 0
    with open(trk_fn, 'rb') as f:
        for idx in idxs:
            # move to the position initial position of the coordinates
            # of the streamline:
            f.seek(index_bytes[idx])
            # Parse the floats:
            s = np.fromfile(f, np.float32, n_floats[idx])
            s.resize(lengths[idx], point_size)

            if n_scalars > 0:
                scalars[j:j+lengths[idx], :] = s[:, 3:]
                s = s[:, :3]
            
            streams[j:j+lengths[idx], :] = s
            j += lengths[idx]
    # apply affine
    aff = get_affine_trackvis_to_rasmm(lazy_trk.header)
    streams = apply_affine(aff, streams)

    if return_scalars:
        streams = np.hstack((streams, scalars))    
        
    return streams, lengths[idxs]
