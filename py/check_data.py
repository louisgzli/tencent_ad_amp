import os
import json
from gensim.models import Word2Vec, KeyedVectors

if __name__ == "__main__":
    wv_registry = None
    target_list  = ['creative', 'ad', 'product', 'advertiser']
    embed_path = "./embed_artifact"
    with open(os.path.join(embed_path, 'wv_registry.json'), 'rb') as f:
        wv_registry = json.load(f)
    for target in target_list:
        KeyedVectors.load(wv_registry[target])
