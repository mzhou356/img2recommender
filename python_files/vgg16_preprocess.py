"""Utilities for ImageNet data preprocessing & prediction decoding.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras_applications import get_submodules_from_kwargs

def _preprocess_numpy_input(x, data_format, **kwargs):
    """Preprocesses a Numpy array encoding a batch of images
       changing RGB to BGR.

    # Arguments
        x: Input array, 3D or 4D.
        data_format: Data format of the image array.

    # Returns
        Preprocessed Numpy array.
    """
    backend, _, _, _ = get_submodules_from_kwargs(kwargs)
    if not issubclass(x.dtype.type, np.floating):
        x = x.astype(backend.floatx(), copy=False)

    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]

    return x


def _preprocess_symbolic_input(x, data_format, **kwargs):
    """Preprocesses a tensor encoding a batch of images
       changing RGB to BGR.

    # Arguments
        x: Input tensor, 3D or 4D.
        data_format: Data format of the image tensor.
    # Returns
        Preprocessed tensor.
    """

    backend, _, _, _ = get_submodules_from_kwargs(kwargs)

    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        if backend.ndim(x) == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]

    return x


def preprocess_input(x, data_format=None, **kwargs):
    """Preprocesses a tensor or Numpy array encoding a batch of images
       changing RGB to BGR.

    # Arguments
        x: Input Numpy or symbolic tensor, 3D or 4D.
            The preprocessed data is written over the input data
            if the data types are compatible. To avoid this
            behaviour, `numpy.copy(x)` can be used.
        data_format: Data format of the image tensor/array.

    # Returns
        Preprocessed tensor or Numpy array.

    # Raises
        ValueError: In case of unknown `data_format` argument.
    """
    backend, _, _, _ = get_submodules_from_kwargs(kwargs)

    if data_format is None:
        data_format = backend.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))

    if isinstance(x, np.ndarray):
        return _preprocess_numpy_input(x, data_format=data_format, **kwargs)
    else:
        return _preprocess_symbolic_input(x, data_format=data_format, **kwargs)


