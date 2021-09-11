from pykeen.models import ComplEx, TransE, DistMult, RotatE
import pickle
import torch
from Wiki5m import Wiki5m


class WIKI5M_TransE(TransE):
    def __init__(self, dataset, model_path="./data/wiki5m/human_transe.pkl"):
        with open(model_path, "rb") as fin:
            entity_embeddings, relation_embeddings = pickle.load(fin)
            embedding_dim = entity_embeddings.shape[1]
            assert dataset.num_entities == entity_embeddings.shape[0]
            assert dataset.num_relations == relation_embeddings.shape[0]
        super(WIKI5M_TransE, self).__init__(dataset, embedding_dim, random_seed=0)
        self.entity_embeddings.weight.data = torch.from_numpy(entity_embeddings)
        self.relation_embeddings.weight.data = torch.from_numpy(relation_embeddings)

class WIKI5M_ComplEx(ComplEx):
    def __init__(self, dataset, model_path="./data/wiki5m/human_complex.pkl"):
        with open(model_path, "rb") as fin:
            entity_embeddings, relation_embeddings = pickle.load(fin)
            embedding_dim = entity_embeddings.shape[1]//2
            assert dataset.num_entities == entity_embeddings.shape[0]
            assert dataset.num_relations == relation_embeddings.shape[0]
        super(WIKI5M_ComplEx, self).__init__(dataset, embedding_dim, random_seed=0)
        self.entity_embeddings.weight.data = torch.from_numpy(entity_embeddings)
        self.relation_embeddings.weight.data = torch.from_numpy(relation_embeddings)

class WIKI5M_DistMult(DistMult):
    def __init__(self, dataset, model_path="./data/wiki5m/human_distmult.pkl"):
        with open(model_path, "rb") as fin:
            entity_embeddings, relation_embeddings = pickle.load(fin)
            embedding_dim = entity_embeddings.shape[1]
            assert dataset.num_entities == entity_embeddings.shape[0]
            assert dataset.num_relations == relation_embeddings.shape[0]
        super(WIKI5M_DistMult, self).__init__(dataset, embedding_dim, random_seed=0)
        self.entity_embeddings.weight.data = torch.from_numpy(entity_embeddings)
        self.relation_embeddings.weight.data = torch.from_numpy(relation_embeddings)

class WIKI5M_RotatE(RotatE):
    def __init__(self, dataset, model_path="./data/wiki5m/human_rotate.pkl"):
        with open(model_path, "rb") as fin:
            entity_embeddings, relation_embeddings = pickle.load(fin)
            embedding_dim = entity_embeddings.shape[1]//2
            print(dataset.num_entities, entity_embeddings.shape[0])
            assert dataset.num_entities == entity_embeddings.shape[0]
            assert dataset.num_relations == relation_embeddings.shape[0]
        super(WIKI5M_RotatE, self).__init__(dataset, embedding_dim, random_seed=0)
        self.entity_embeddings.weight.data = torch.from_numpy(entity_embeddings)
        self.relation_embeddings.weight.data = torch.from_numpy(relation_embeddings)

if __name__ == '__main__':
    wiki5m = Wiki5m()
    models = {"trans_e": WIKI5M_TransE, "complex": WIKI5M_ComplEx, "distmult":WIKI5M_DistMult, "rotate":WIKI5M_RotatE}

    for k in models.keys():
        save_path = "./trained_models/wiki5m/{}.pkl".format(k)
        model = models[k](wiki5m.training)
        torch.save(model, save_path)
    print("Done")