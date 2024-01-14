import json
import io

import h5py
import numpy as np
from PIL import Image


def get_h5_struct(path):
    data = []
    group = []
    hf = h5py.File(path, 'r')

    def visit_h5_file(name, obj):
        if isinstance(obj, h5py.Dataset):
            data.append(name)
        elif isinstance(obj, h5py.Group):
            group.append(name)
    
    hf.visititems(visit_h5_file)
    
    struct = {g: [] for g in group}
    for d in data:
        group_name = d.rsplit('/', 1)[0]  # Extract the group name from the dataset path
        if group_name in struct:
            struct[group_name].append(d)

    return hf, struct


def get_raw_from_h5(hf, path):
    '''
    Read raw data from the HDF5 file, given the HDF5 file object and the path.
    '''
    # For npy files, return the ndarray as it is stored. In other cases,
    # all files are stored as binary data, so convert to bytes.
    rtn = np.array(hf[path])
    if not path.endswith('.npy'):
        rtn = rtn.tobytes()
    return rtn


def get_file_from_h5(hf, path):
    '''
    Read file from the HDF5 file with extra data conversion, given the HDF5 file object and the path.
    '''
    raw_data = get_raw_from_h5(hf, path)
    if path.endswith('.npy'):
        # For npy files, return the ndarray as it is stored.
        rtn = raw_data
    elif path.endswith('.jpg') or path.endswith('.jpeg') or path.endswith('.png'):
        # For image files, convert to PIL Image.
        rtn = Image.open(io.BytesIO(raw_data))
    elif path.endswith('.json'):
        # For json files, convert to dict.
        rtn = json.loads(raw_data.decode('utf-8'))
    elif path.endswith('.txt'):
        # For text files, convert to str.
        rtn = raw_data.decode('utf-8')
    else:
        # For all other files, return as bytes.
        rtn = raw_data
    return rtn
