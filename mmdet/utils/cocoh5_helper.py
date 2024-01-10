import json

import numpy as np
import h5py


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


def get_h5_file(hf, path):
    if path.endswith('.jpg'):
        # saved the image as raw binary, read in as raw bytes
        rtn = np.array(hf[path]).tobytes()
    elif path.endswith('.json'):
        # saved as a dataset string, need to convert to json dict
        rtn = json.loads(np.array(hf[path]).tobytes().decode('utf-8'))
    elif path.endswith('.npy'):
        # saved as array, no need to convert
        rtn = np.array(hf[path])
    else:
        raise ValueError('Unknown file type: {}'.format(path))
    return rtn
