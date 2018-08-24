import json
import os

# Constants ---

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)    

# Functions ---
    
def __json_to_dict(fn):
    return json.loads(''.join(open(fn).readlines()))


def read_metadata(path, indices):
    '''
    Reads metadata for the specified indices and returns a dictionary
    mapping the index to the metadata dictionary.
    '''
    return {idx : __json_to_dict(os.path.join(path, '{}.json'.format(idx))) \
            for idx in indices}


def sort_by_scene_metadata(path, indices, key):
    '''
    Returns an array of indices that is sorted by the specified key in the scene metadata.
    '''
    metadata = read_metadata(path, indices)
    return sorted(indices, key=lambda index: metadata[index]['scene'][key])


def metadata_values(path, indices, key):
    '''
    Extracts the metadata values for the specified indices
    '''
    metadata = read_metadata(path, indices)
    return [metadata[index]['scene'][key] for index in indices]
