from pykeen.datasets import FB15k237
import pandas as pd
import numpy as np
import os
import torch
from collections import Counter

from utils import get_classifier, suggest_relations, remove_infreq_attributes
from classifier import TargetRelationClassifier
from collections import Counter

from BiasEvaluator import BiasEvaluator

def add_relation_values(dataset, preds_df, bias_relations):
    """
    Given a datafrae with predicitons for the target relation,
    add the true tail values of the entities and the relations
    that are being examined for bias evaluation

    dataset: pykeen.Dataset, knowledge graph dataset e.g fb15k-237
    preds_df: pd.DataFrame, .
    bias_relations: list of str,
    """

    def get_tail(rel, x):
        try:
            return entity_to_tail[rel][x]
        except KeyError:
            return -1

    triplets = dataset.testing.get_triples_for_relations(bias_relations) #len 3357
    triplets = [tr for tr in triplets if dataset.entity_to_id[tr[0]] in preds_df.entity.values] #len 1808
    entity_to_tail = {}
    for rel in bias_relations:
        entity_to_tail[rel] = {}
    for head, rel, tail in triplets:
        head_id = dataset.entity_to_id[head]
        tail_id = dataset.entity_to_id[tail]
        entity_to_tail[rel][head_id] = tail_id
    for rel in bias_relations:
        preds_df[rel] = [get_tail(rel, e_id) for e_id in preds_df.entity.values]
        attr_counts = Counter(preds_df[rel])
        preds_df[rel] = preds_df[rel].apply(lambda x: remove_infreq_attributes(attr_counts, x))
    return preds_df

def predict_relation_tails(dataset, trained_classifier, target_test_triplets):
    """
    predict the tail t for (h,r,t)
    for each head entity h in the dataset
    return a dataframe with the predictions - each row is an entity

    dataset: pykeen.Dataset, knowledge graph dataset e.g fb15k-237
    trained_classifier:
    target_test_triplets:
    relation: str,
    binary: bool, whether classification task is binary, default to False
    """

    preds_df = pd.DataFrame({'entity': target_test_triplets[:, 0],
                   'relation': target_test_triplets[:,1],
                   'true_tail': target_test_triplets[:, 2],
                   })
    target_relation = preds_df.relation.loc[0]
    preds_df['entity'] = preds_df['entity'].apply(lambda head : dataset.entity_to_id[head])
    heads = torch.Tensor(preds_df['entity'].values).long()
    preds_df['pred'] = trained_classifier.predict_tails(heads, target_relation)
    preds_df['pred'] = preds_df['pred'].apply(lambda x: int(x))
    preds_df['true_tail'] = preds_df['true_tail'].apply(lambda tail_entity: dataset.entity_to_id[tail_entity])
    preds_df['true_tail'] = preds_df['true_tail'].apply(lambda tail_entity : trained_classifier.target2label(tail_entity))

    return preds_df

def predict_relation_tails_rf(dataset, trained_classifier, target_test_triplets):
    """
    predict the tail t for (h,r,t)
    for each head entity h in the dataset
    return a dataframe with the predictions - each row is an entity

    dataset: pykeen.Dataset, knowledge graph dataset e.g fb15k-237
    trained_classifier:
    target_test_triplets:
    relation: str,
    """

    preds_df = pd.DataFrame({'entity': target_test_triplets[:, 0],
                   'relation': target_test_triplets[:,1],
                   'true_tail': target_test_triplets[:, 2],
                   })
    preds_df['entity'] = preds_df['entity'].apply(lambda head : dataset.entity_to_id[head])
    heads = torch.Tensor(preds_df['entity'].values).int()
    heads = trained_classifier._model.entity_embeddings(heads).detach().numpy()
    preds_df['pred'] = trained_classifier._classifier.predict(heads)
    preds_df['pred'] = preds_df['pred'].apply(lambda x: int(x))
    preds_df['true_tail'] = preds_df['true_tail'].apply(lambda tail_entity: dataset.entity_to_id[tail_entity])
    preds_df['true_tail'] = preds_df['true_tail'].apply(lambda tail_entity : trained_classifier.target2label(tail_entity))

    return preds_df

def get_preds_df(dataset, classifier_args, model_args, target_relation, bias_relations, preds_df_path=None):
    """
    Get predictions dataframe used in parity distance calculation

    dataset: pykeen.Dataset, knowledge graph dataset e.g fb15k-237
    classifier_args: dict, parameters passed to train classifier
    model_args: dict,
    target_relation: str,
    bias_relations: list of str,
    preds_df_path: str, path to predictions dataframe, default to None
    """
    if preds_df_path is not None and os.path.exists(preds_df_path):
        preds_df = pd.read_csv(preds_df_path)
        del preds_df['Unnamed: 0']
        print(f"Load MLP classifier from : {preds_df_path}")
        return preds_df
    else:
        clsf_type = "mlp"
        if "type" in classifier_args: clsf_type = classifier_args["type"]
        classifier = get_classifier(dataset=dataset,
                                    target_relation=target_relation,
                                    num_classes=classifier_args["num_classes"],
                                    batch_size=classifier_args["batch_size"],
                                    embedding_model_path=model_args['embedding_model_path'],
                                    classifier_type = clsf_type,
                                    )
        target_test_triplets = dataset.testing.get_triples_for_relations([target_relation])
        if clsf_type == "mlp":
            classifier.train(classifier_args['epochs'])
            preds_df = predict_relation_tails(dataset, classifier, target_test_triplets)
        elif clsf_type == "rf":
            classifier.train()
            preds_df = predict_relation_tails_rf(dataset, classifier, target_test_triplets)
    preds_df = add_relation_values(dataset, preds_df, bias_relations)
    if preds_df_path:
        preds_df.to_csv(preds_df_path)
    print(f"Load classifier from : {preds_df_path}")
    return preds_df

# TODO: Pass less default param, maybe add kwargs
def eval_bias(evaluator,
              classifier_args,
              model_args,
              bias_relations=None,
              bias_measures=None,
              preds_df_path=None,
              ):
    """
    evaluator: instance of Evaluator(see BiasEvaluator.py),
    classifier_args: dict,
    model_args: dict,
    bias_relations: list of str,
    bias_measures: list of instances of Measurement,
    preds_df_path: str, path to predictions
    """
    # Train tail-prediction classifier and get predictions
    target_relation = evaluator.target_relation
    dataset = evaluator.dataset
    require_preds_df = False
    for m in bias_measures:
        if m.require_preds_df:
            require_preds_df = True
            break
    if require_preds_df:
        preds_df = get_preds_df(dataset=dataset,
                            classifier_args=classifier_args,
                            model_args=model_args,
                            target_relation=target_relation,
                            bias_relations=bias_relations,
                            preds_df_path=preds_df_path,
                            )
        print("Got predictions dataframe")
        evaluator.set_predictions_df(preds_df)
    eval_bias = evaluator.evaluate_bias(bias_relations, bias_measures)
    return eval_bias

def predict_from_embedding_model():

    triples = rf.train_triples_factory.triples
    possible_professions = ['/m/02hrh1q', '/m/0dxtg', '/m/01d_h8', '/m/02jknp', '/m/03gjzk', 'OTHER'] #set(rf.train_triples_factory.triples[:, 2])
    #labels = ['/m/02hrh1q', '/m/0dxtg', '/m/01d_h8', '/m/02jknp', '/m/03gjzk', 'OTHER']
    predictions, true_tails = [], []
    for (h,r,t) in triples:
        tails = rf._model.predict_tails(h,r)
        prds = rf._model.predict_tails(h, target_relation)
        max_score = np.max(prds.score)
        pred = prds.tail_label[max_score == prds.score].values[0]
        i = 0
        while pred not in possible_professions or i < 3:
            i += 1
            new_max_score = np.max(prds.score[prds.score < max_score])
            pred = prds.tail_label[new_max_score == prds.score].values[0]
            max_score = new_max_score
        predictions.append(pred)
        true_tails.append(t)

    new_tails = []
    for tl in true_tails:
        if tl in possible_professions: new_tails.append(tl)
        else: new_tails.append('OTHER')
    return predictions, new_tails

def save_preds_df(file_name, model_name, idx=0, plot=False):
    dataset = FB15k237()
    dataset_name = 'fb15k237'

    target_relation, bias_relations = suggest_relations(dataset_name)
    num_classes = 11
    import  time
    s = time.time()
    rf = RFRelationClassifier(
        dataset=dataset,
        embedding_model_path=file_name,
        target_relation=target_relation,
        num_classes=num_classes,
        batch_size=256,
        class_weight='balanced',
        max_depth=4,
        random_state=111,
    )

    rf.train()

    e = time.time()
    print("rf took", e-s, "seconds tu run on fb15k237")
    target_test_triplets = dataset.testing.get_triples_for_relations([target_relation])
    preds_df = predict_relation_tails_rf(dataset, rf, target_test_triplets)
    preds_df = add_relation_values(dataset, preds_df, bias_relations)

    random_preds = [np.random.randint(num_classes) for __ in preds_df.pred]
    print("classification accuracy for random labels", accuracy_score(preds_df.true_tail, random_preds))
    print("balanced classification accuracy for random labels",
          balanced_accuracy_score(preds_df.true_tail, random_preds))

    print("classification accuracy for rf model", accuracy_score(preds_df.true_tail, preds_df.pred))
    print("balanced classification accuracy for rf model", balanced_accuracy_score(preds_df.true_tail, preds_df.pred))
    if plot: preds_histogram(preds_df)

    #measures = [DemographicParity(), PredictiveParity()]

    #evaluator = BiasEvaluator(dataset, measures)
    #evaluator.set_predictions_df(preds_df)
    #bias_eval = evaluator.evaluate_bias(bias_relations=bias_relations, bias_measures=measures)
    #d_parity, p_parity = bias_eval['demographic_parity'], bias_eval['predictive_parity']
    preds_df.to_csv("preds_df_"+model_name+str(idx)+".csv")


if __name__ == '__main__':

    from classifier import RFRelationClassifier
    from Measurement import DemographicParity, PredictiveParity
    from sklearn.metrics import balanced_accuracy_score, accuracy_score
    from visualization import preds_histogram
    from collections import Counter

    dataset = FB15k237()
    dataset_name = 'fb15k237'

    target_relation, bias_relations = suggest_relations(dataset_name)

    file_names = ['/Users/alacrity/Documents/uni/Fairness/fb15k237/distmult/replicates/replicate-00000/trained_model.pkl',
                  '/Users/alacrity/Documents/uni/Fairness/fb15k237/trans_e/replicates/replicate-00000/trained_model.pkl',
                  '/Users/alacrity/Documents/uni/Fairness/fb15k237/conve/replicates/replicate-00000/trained_model.pkl',
                  '/Users/alacrity/Documents/uni/Fairness/fb15k237/rotate/replicates/replicate-00000/trained_model.pkl',
                  '/Users/alacrity/Documents/uni/Fairness/fb15k237/complex/replicates/replicate-00000/trained_model.pkl'
                  ]
    model_names = ['distmult','transe','conve','rotate','complex']
    idx = 3
    #for file_name,model_name in zip(file_names, model_names):
    #    save_preds_df(file_name,model_name, idx=idx)
    # # save_preds_df('/Users/alacrity/Documents/uni/Fairness/fb15k237/conve/replicates/replicate-00000/trained_model.pkl',
    # #               'conve', idx=1000)

    for model_name in model_names:
        preds_df = pd.read_csv(f'./preds_dfs/preds_df_'+model_name+str(idx)+'.csv')
        measures = [DemographicParity(), PredictiveParity()]

        import time

        s = time.time()
        evaluator = BiasEvaluator(dataset, measures)
        evaluator.set_predictions_df(preds_df)
        bias_eval = evaluator.evaluate_bias(bias_relations=bias_relations, bias_measures=measures)
        e = time.time()
        print("calculating PPD and DPD for fb15k237 took", e-s, "seconds")

        d_parity, p_parity = bias_eval['demographic_parity'], bias_eval['predictive_parity']
        d_parity.to_csv(f'./preds_dfs/DPD_'+model_name+str(idx)+'.csv')
        p_parity.to_csv(f'./preds_dfs/PPD_'+model_name+str(idx)+'.csv')

        acc = accuracy_score(preds_df.pred, preds_df.true_tail)
        bacc = balanced_accuracy_score(y_pred=preds_df.pred, y_true=preds_df.true_tail)
        print(acc)
        print(bacc)

    fname = "/Users/alacrity/Documents/uni/Fairness/trained_model.pkl"
    # Trained Model Path
    # fname = '/local/scratch/kge_fairness/models/fb15k237/transe_openkeparams_alpha1/replicates/replicate-00000/trained_model.pkl'
    dataset = FB15k237()
    dataset_name = 'fb15k237'
    GENDER_RELATION = '/people/person/gender'
    PROFESSION_RELATION = '/people/person/profession'

    target_relation, bias_relations = suggest_relations(dataset_name)
    num_classes = 6


    rf = RFRelationClassifier(
        dataset=dataset,
        embedding_model_path=fname,
        target_relation=target_relation,
        num_classes=num_classes,
        batch_size=500,
        class_weight='balanced',
        max_depth=6,
        random_state=111,
        n_estimators=100,
    )

    rf.train()


    target_test_triplets = dataset.testing.get_triples_for_relations([target_relation])
    preds_df = pd.DataFrame({'entity': target_test_triplets[:, 0],
                             'relation': target_test_triplets[:, 1],
                             'true_tail': target_test_triplets[:, 2],
                             })
    target_relation = preds_df.relation.loc[0]

    preds_df = predict_relation_tails_rf(dataset, rf, target_test_triplets)
    preds_df = add_relation_values(dataset, preds_df, bias_relations)

    random_preds = [np.random.randint(num_classes) for __ in preds_df.pred]
    print("classification accuracy for random labels", accuracy_score(preds_df.true_tail,random_preds))
    print("balanced classification accuracy for random labels", balanced_accuracy_score(preds_df.true_tail,random_preds))

    print("classification accuracy for rf model", accuracy_score(preds_df.true_tail,preds_df.pred))
    print("balanced classification accuracy for rf model", balanced_accuracy_score(preds_df.true_tail,preds_df.pred))
    preds_histogram(preds_df)

    measures = [DemographicParity(), PredictiveParity()]

    evaluator = BiasEvaluator(dataset,measures)
    evaluator.set_predictions_df(preds_df)
    bias_eval = evaluator.evaluate_bias(bias_relations=bias_relations,bias_measures=measures)
    d_parity, p_parity = bias_eval['demographic_parity'], bias_eval['predictive_parity']

    # model = torch.load(fname, map_location=torch.device('cpu'))
    # df = pd.read_csv("top_hyperparam_accuracies.csv")
    #
    # param_grid = {"hdims":[[256,32],[256,32,32,16]],
    #               "batch_size": [64, 152, 200, 256],
    #               "lr":[0.1, 0.01, 0.001],
    #               "num_epochs":[10,20],
    # }
    #
    # accuracies = {
    #     "hdim":[],
    #     "batch_size":[],
    #     "lr":[],
    #     "num_epochs":[],
    #     "accuracy":[],
    #     "balanced_accuracy":[]
    # }
    # for hdim in param_grid["hdims"]:
    #     for batch_size in param_grid["batch_size"]:
    #         for lr in param_grid["lr"]:
    #             for num_epochs in param_grid["num_epochs"]:
    #
    #                 classifier = TargetRelationClassifier(
    #                     dataset=dataset,
    #                     embedding_model_path=fname,
    #                     target_relation=target_relation,
    #                     num_classes=num_classes,
    #                     hdims=hdim,
    #                     batch_size=batch_size,
    #                     lr=lr
    #                 )
    #
    #                 classifier.train(epochs=num_epochs)
    #
    #                 target_test_triplets = dataset.testing.get_triples_for_relations([target_relation])
    #                 preds_df = predict_relation_tails(dataset, classifier, target_test_triplets)
    #                 preds_df = add_relation_values(dataset, preds_df, bias_relations)
    #
    #                 acc = accuracy_score(preds_df.true_tail,preds_df.pred)
    #                 print("classification accuracy", acc)
    #                 balanced_acc = balanced_accuracy_score(preds_df.true_tail, preds_df.pred)
    #                 print("balanced classification accuracy",balanced_acc )
    #                 accuracies["hdim"].append(hdim)
    #                 accuracies["batch_size"].append(batch_size)
    #                 accuracies["lr"].append(lr)
    #                 accuracies["num_epochs"].append(num_epochs)
    #                 accuracies["accuracy"].append(acc)
    #                 accuracies["balanced_accuracy"].append(balanced_acc)
    #                 df = pd.DataFrame(accuracies)
    #                 df.to_csv("hyperparam_accuracies2.csv")
    #
    # random_preds = [np.random.randint(num_classes) for __ in preds_df.pred]
    # print("classification accuracy for random labels", accuracy_score(preds_df.true_tail,random_preds))
    # print("balanced classification accuracy for random labels", balanced_accuracy_score(preds_df.true_tail,random_preds))

