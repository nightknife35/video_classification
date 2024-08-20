"""
import sys
sys.path.append('/home/nightknife35/projects/note-app/v1-cude-and-pytorch')
"""

import json

def import_settings():

    with open('settings/settings.json', 'r') as file:
        json_content = file.read()

    data = json.loads(json_content)
    return data



"""
    for i in sys.path: # some list of paths
        print(i)
    path1 = os.path.abspath(__file__) # /home/nightknife35/projects/note-app/v1-cude-and-pytorch/settings/import_settings.py
    path2 = os.path.dirname(path1) # /home/nightknife35/projects/note-app/v1-cude-and-pytorch/settings
    path3 = os.path.dirname(path2) # /home/nightknife35/projects/note-app/v1-cude-and-pytorch
    # print(path1, '\n', path2, '\n', path3)
    sys.path.append(path3)
    for i in sys.path :# some list of paths + /home/nightknife35/projects/note-app/v1-cude-and-pytorch
        print(i)

    import sys
    sys.path.append('/home/nightknife35/projects/note-app/v1-cude-and-pytorch')
    
"""