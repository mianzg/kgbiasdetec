from Measurement import *
import torch
import pandas as pd

class BiasEvaluator:
    """An evaluator for KG embedding bias detection"""
    def __init__(self, dataset, measures, preds_df=None):
        """
        dataset: instance of pykeen Dataset
        measures: list of instances of Measurement classes (see Measurement.py)
        preds_df: pandas.DataFrame, prediction dataframe from classification task required for certain bias measures (DPD, PPD)
        """
        self.binary_target = True
        self.measure_objects = {measure.name:measure for measure in measures}
        self.dataset = dataset
        self.measures = measures
        self.predictions = None
        if preds_df is not None:
            self.set_predictions_df(preds_df)

    def set_target_relation(self, relation):
        """
        relation: str, relation label e.g '/people/person/profession' (FB15k-237)
        """
        self.target_relation = relation

    def set_bias_relations(self, relations):
        """
        relations: a list str, e.g ['/people/person/gender', '/people/person/nationality'] (FB15k-237)
        """
        self.bias_relations = relations

    def set_trained_model(self, model):
        """
        model: instance of a class in pykeen.models, e.g pykeen.models.unimodal.trans_e.TransE
        """
        with torch.no_grad():
            self.trained_model = model

    def set_predictions_df(self, preds_df):
        """
        * predictions_df is a dataframe of entities, and the following columns:
            -'entity' = entity name
            -'pred' = predicted value for target relation
            -'true' = true value for target relation
            - relation_name = true value for relation
        preds_df: pd.DataFrame,
        """
        self.predictions = preds_df
    
    def filter_missing_relations(self, bias_relations):
        """
        Filter & only keep relations that are in the predictions dataframe
        bias_relations: list of str, relations of the interest to detect bias, e.g. ['/people/person/gender', '/people/person/nationality'] (FB15k-237)
        """
        rels_in_df = [rel for rel in bias_relations if rel in self.predictions.columns]
        # filter out relations that only have one tail
        rels_with_vals = [rel for rel in rels_in_df if len(set(self.predictions[rel])) > 1]
        return rels_with_vals

    def evaluate_bias(self, bias_relations=None, bias_measures=None, **kwargs):
        """
        Given a list of relations of interest to detect bias, namely bias_relations, and
        a list bias measurements, iterate over each measure to output bias calculation accordingly.
        Return a map with measurement name as key and measurement results as values. Measurement
        results are usually in the format of pd.DataFrame or dict of pd.DataFrame

        bias_relations: list of str, relations that we want to score for bias.
            They must already exist in the predictions DataFrame
        bias_measures: list of Measurement instances, by default it supports DemographicParity, PredictiveParity, TranslationalLikelihood
        """
        if 'require_preds_df' in kwargs.keys() and kwargs['require_preds_df']:
            if self.predictions is None:
                print("No predictions sorry, please set a predictions dataframe")
                return None
            else:
                bias_relations = self.filter_missing_relations(bias_relations)

        print("Evaluating ...")
        
        if bias_measures is None:
            bias_measures = self.measures

        bias_result = {}
        for measure in bias_measures:
            name = measure.get_name()
            print(f"Measure: {name}\n =============\n")
            bias_result[name] = measure.calculate(self, bias_relations)
        return bias_result

    def calculate_prediction_scores(self, relation):
        attributes = self.predictions[relation]
        attribute_set = list(set(attributes))
        accuracy,precision,recall = [],[],[]
        for attr in attribute_set:
            has_attr = self.predictions[self.predictions[relation] == attr]
            ytrue = has_attr.true_tail
            ypred = has_attr.pred
            accuracy.append(accuracy_score(y_true=ytrue,y_pred=ypred))
            precision.append(precision_score(y_true=ytrue, y_pred=ypred,average='micro'))
            recall.append(recall_score(y_true=ytrue, y_pred=ypred,average='macro'))
        df = pd.DataFrame({"relation_val":attribute_set, "precision":precision,"recall":recall,"accuracy":accuracy})
        return df
