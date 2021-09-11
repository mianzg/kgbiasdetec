import os
import numpy as np
import pickle
from pykeen.datasets.base import PathDataSet
from pykeen.triples import TriplesFactory

class Wiki5m(PathDataSet):
    def __init__(self, **kwargs):
        cache_root = "./data/wiki5m"
        self.name = "wiki5m"

        self.training_path = os.path.join(cache_root, "train.tsv")
        self.test_path = os.path.join(cache_root, "test.tsv")
        self.val_path = os.path.join(cache_root, "val.tsv")
        if not os.path.exists(self.training_path):
            tf = TriplesFactory(path=os.path.join(cache_root,"wiki5m_human_triples.tsv"))
            train, val, test = tf.split([0.8, 0.1, 0.1], random_state=0)
            np.savetxt(fname=self.training_path, X=train.triples, fmt = '%s', delimiter='\t')
            np.savetxt(fname=self.test_path, X=test.triples, fmt = '%s', delimiter='\t')
            np.savetxt(fname=self.val_path, X=val.triples, fmt = '%s', delimiter='\t')
        
        super(Wiki5m,self).__init__(training_path = self.training_path, 
                         testing_path = self.test_path,
                         validation_path = self.val_path,
                         **kwargs)

if __name__ == '__main__':
    wiki5m = Wiki5m()
    with open("data/wiki5m/human_ent_rel_sorted_list.pkl", "rb") as f:
        ent, rel = pickle.load(f)
    assert list(wiki5m.training.relation_to_id.keys()) == list(wiki5m.testing.relation_to_id.keys())
    assert list(wiki5m.training.relation_to_id.keys()) == rel 
    assert list(wiki5m.training.entity_to_id.keys()) == ent