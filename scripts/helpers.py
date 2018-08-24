import json
import os

def __json_to_dict(fn):
    return json.loads(''.join(open(fn).readlines()))


def read_all_metadata(path, indices):
    return {idx : __json_to_dict(os.path.join(path, '{}.json'.format(idx))) \
            for idx in indices}


def sort_by_scene_metadata(path, indices, key):
    metadata = read_all_metadata(path, indices)
    return sorted(indices, key=lambda index: metadata[index]['scene'][key])
