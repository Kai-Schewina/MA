from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os


def read_chunk(reader, chunk_size):
    data = {}
    for i in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]
    return data


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_data(reader, discretizer, normalizer, small_part=False, return_names=False):
    n = reader.get_number_of_examples()
    if small_part:
        n = 1000
    ret = read_chunk(reader, n)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    whole_data = (np.array(data, dtype=object), labels)

    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}


def save_results(names, pred, y_true, path):
    create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        f.write("stay,prediction,y_true\n")
        for (name, x, y) in zip(names, pred, y_true):
            f.write("{},{:.6f},{}\n".format(name, x, y))
