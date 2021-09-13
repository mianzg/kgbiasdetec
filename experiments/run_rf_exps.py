from Measurement import DemographicParity, PredictiveParity
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from matplotlib import pyplot as plt
import pandas as pd
from pykeen.datasets import FB15k237
from utils import suggest_relations
from classifier import TargetRelationClassifier, RFRelationClassifier
from predict_tails import predict_relation_tails, get_preds_df, add_relation_values
from collections import Counter

#fname = "/Users/alacrity/Documents/uni/Fairness/trained_model.pkl"
fname =  '/Users/alacrity/Documents/uni/Fairness/fb15k237/conve/replicates/replicate-00000/trained_model.pkl'

# Trained Model Path
# fname = '/local/scratch/kge_fairness/models/fb15k237/transe_openkeparams_alpha1/replicates/replicate-00000/trained_model.pkl'
dataset = FB15k237()
dataset_name = 'fb15k237'
GENDER_RELATION = '/people/person/gender'
PROFESSION_RELATION = '/people/person/profession'

target_relation, bias_relations = suggest_relations(dataset_name)
num_classes = 6

def read_hdims(a):
    a = a.strip('[').strip(']').split(',')
    return [int(aa) for aa in a]


accuracies = {
    "criterion": [],
    "batch_size": [],
    "n_ests": [],
    "max_features": [],
    "max_depth":[],
    "accuracy": [],
    "balanced_accuracy": []
}

for n_ests in [10,50,100]:
    for max_depth in [3,4,5]:
        for batch_size in [125,256,500]:
            for criterion in ['entropy', 'gini']:
                for max_features in ["auto"]:
                    rf = RFRelationClassifier(
                        dataset=dataset,
                        embedding_model_path=fname,
                        target_relation=target_relation,
                        num_classes=num_classes,
                        batch_size=batch_size,
                        class_weight='balanced',
                        max_depth=max_depth,
                        random_state=111,
                        n_estimators=n_ests,
                        max_features=max_features,
                        criterion=criterion
                    )

                    rf.train()

                    target_test_triplets = dataset.testing.get_triples_for_relations([target_relation])
                    preds_df = pd.DataFrame({'entity': target_test_triplets[:, 0],
                                             'relation': target_test_triplets[:, 1],
                                             'true_tail': target_test_triplets[:, 2],
                                             })
                    target_relation = preds_df.relation.loc[0]

                    preds_df = predict_relation_tails(dataset, rf, target_test_triplets)
                    preds_df = add_relation_values(dataset, preds_df, bias_relations)

                    acc = accuracy_score(preds_df.true_tail, preds_df.pred)
                    print("classification accuracy", acc)
                    balanced_acc = balanced_accuracy_score(preds_df.true_tail, preds_df.pred)
                    print("balanced classification accuracy", balanced_acc)
                    accuracies["n_ests"].append(n_ests)
                    accuracies["batch_size"].append(batch_size)
                    accuracies["max_depth"].append(max_depth)
                    accuracies["criterion"].append(criterion)
                    accuracies["accuracy"].append(acc)
                    accuracies["max_features"].append(max_features)
                    accuracies["balanced_accuracy"].append(balanced_acc)
                    new_df = pd.DataFrame(accuracies)
                    new_df.to_csv("rf_accs_conve.csv")