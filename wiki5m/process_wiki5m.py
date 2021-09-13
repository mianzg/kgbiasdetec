import pickle

def load_embeddings(embedding_path):
    with open(embedding_path, "rb") as fin:
        model = pickle.load(fin)
    entity2id = model.graph.entity2id
    relation2id = model.graph.relation2id
    entity_embeddings = model.solver.entity_embeddings
    relation_embeddings = model.solver.relation_embeddings
    print("Num Entities: ", len(entity2id.keys()))
    print("Num Relations: ", len(relation2id.keys()))
    return (entity2id, relation2id), (entity_embeddings, relation_embeddings)

def get_human_triples(triple_path):
    with open("../data/wiki5m/wikidata5m_all_triplet.txt") as f:
        triples = [t.split() for t in f.readlines()]
        humans = {}
        for t in triples:
            if t[1]=="P31" and t[2]=="Q5":
                humans[t[0]] = True
            else:
                humans[t[0]] = False
        human_triples = [t for t in triples if humans[t[0]]]
        print("Num human triples: ", len(human_triples))
        print("Num total triples: ", len(triples))
    return triples, human_triples

def convert2humanembeddings(embedding_name, entlist, rellist):
    model_path = "./data/wiki5m/{}_wikidata5m.pkl".format(embedding_name)
    (ent2id, rel2id), (entity_embeddings, relation_embeddings) = load_embeddings(model_path)
    human_ent_ids = [ent2id[i] for i in entlist]
    human_rel_ids = [rel2id[i] for i in rellist]
    human_entity_embeddings = entity_embeddings[human_ent_ids]
    human_relation_embeddings = relation_embeddings[human_rel_ids]
    pickle.dump((human_entity_embeddings, human_relation_embeddings), "./data/wiki5m/human_{}.pkl".format(embedding_name))

embeddings = ["transe", "distmult", "complex", "rotate"]
with open("../data/wiki5m/human_ent_rel_sorted_list.pkl", 'rb') as f:
    entlist, rellist = pickle.load(f)
for e in embeddings:
    convert2humanembeddings(e, entlist, rellist)
