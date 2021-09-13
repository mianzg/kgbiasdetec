from datetime import datetime
import os
import pandas as pd

def get_classifier(dataset, target_relation, num_classes, batch_size, embedding_model_path,
                   classifier_type='mlp',
                   **model_kwargs):
    """
    Return a classifier that will classify the tails for the target relations
    Currently only MLP classifier is implemented, but can look into others
    dataset: pykeen.Dataset, knowledge graph dataset e.g fb15k-237
    target_relation: str, 
    num_classes: int, number of classification classes
    embedding_model_path: str, path to trained embedding model using pykeen library
    """
    if classifier_type == 'mlp':
        from classifier import TargetRelationClassifier
        classifier = TargetRelationClassifier(dataset=dataset,
                                              embedding_model_path=embedding_model_path,
                                              target_relation=target_relation,
                                              num_classes=num_classes,
                                              batch_size=batch_size,
                                              **model_kwargs
                                              )
    elif classifier_type == 'rf':
        from classifier import RFRelationClassifier
        classifier = RFRelationClassifier(dataset=dataset,
                                          embedding_model_path=embedding_model_path,
                                          target_relation=target_relation,
                                          num_classes=num_classes,
                                          batch_size=batch_size,
                                          **model_kwargs
                                          )
    return classifier

def suggest_relations(dataset):
    """
    Suggest a list of relations to detect bias based on knowledge graph datasets.

    dataset: pykeen.Dataset, knowledge graph dataset e.g fb15k-237
    """
    if dataset.lower() == "fb15k237":
        target_relation = '/people/person/profession'
        bias_relations = ['/people/person/gender',
                        '/people/person/languages',
                        '/people/person/nationality',
                        '/people/person/profession',
                        '/people/person/places_lived./people/place_lived/location',
                        '/people/person/spouse_s./people/marriage/type_of_union',
                        '/people/person/religion'
                        #/people/person/place_of_birth - top have 14, 13, 9, 4, 4, 3,3..
                        ]
    elif dataset.lower() == "wikidata":
        target_relation = 'P21'
        bias_relations = ['P102', 'P106', 'P169']
    elif dataset.lower() == "wiki5m":
        target_relation = 'P106'
        bias_relations = ['P27', 'P735', 'P19', 'P54', 'P69', 'P641', 'P20', 'P1344', 'P1412', 'P413']
    return target_relation, bias_relations

def save_result(result, dataset, args):
    """
    Save dataset summary, and output from Evaluator

    result: dict, bias evaluation result 
    dataset: pykeen.Dataset, knowledge graph dataset e.g fb15k-237
    args: arguments passed when running main program
    """
    if args.embedding_path:
        embedding = os.path.splitext(os.path.split(args.embedding_path)[-1])[0]
    else:
        embedding = args.embedding
    date = datetime.now().strftime("%Y%m%d%H%M")
    dir = os.path.join("./results/", args.dataset+"_"+embedding+"_"+date)
    if not os.path.exists(dir):
        os.makedirs(dir)
        # Save Dataset Summary
    with open(os.path.join(dir, args.dataset+".txt"), 'w') as f: # save dataset summary
        f.writelines(dataset.summary_str())
        #TODO: save embedding training configuration?
    for k in result.keys():
        measure_dir = os.path.join(dir, k)
        os.mkdir(measure_dir)
        if isinstance(result[k], pd.DataFrame):
            save_path = os.path.join(measure_dir, "{}.csv".format(k))
            print("Save to {}".format(save_path))
            result[k].to_csv(save_path)
        elif isinstance(result[k], dict):
            for rel in result[k].keys():
                df = pd.DataFrame(result[k][rel])
                rel = rel.split('/')[-1] if args.dataset == 'fb15k237' else rel
                save_path = os.path.join(measure_dir, "{}_{}.csv".format(k,rel))
                print("Save to {}".format(save_path))
                df.to_csv(save_path)

def remove_infreq_attributes(attr_counts, key, threshold=10,nan_val=-1):
    if attr_counts[key] <= threshold:
        return nan_val
    return key
