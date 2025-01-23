from dotenv import load_dotenv
from math import ceil
import importlib
import inspect
import os

def get_captioners():
    module = importlib.import_module('.', package='captioners')
    classes = inspect.getmembers(module, inspect.isclass)
    return {k:v for k,v in classes}

def split_dict(dictionary, n):
    items = list(dictionary.items())
    section_size = ceil(len(dictionary) / n)
    
    # Split into n approximately equal sections
    sections = []
    for i in range(0, len(items), section_size):
        section = dict(items[i:i + section_size])
        sections.append(section)
    
    return sections


def merge_dicts(dict1, dict2):
   for key, value in dict2.items():
       if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
           dict1[key].update(value)
       else:
           dict1[key] = value
   return dict1


def load_huggingface_environment(env_path,vars=["HF_TOKEN","HF_HOME"])->dict[str,str]:
    envs={}
    load_dotenv(env_path)
    for env in vars:
        envs[env]=os.getenv(env)
    return envs