"""
Bias Measurement Classes
Currently implemented : Demographic Parity, Predictive Parity, Translational Likelihood Bias (TLB)
"""
import torch
import pandas as pd
import numpy as np

from sklearn.metrics import precision_score, accuracy_score, recall_score
import pykeen.models  
import time 


class Measurement:
    def __init__(self):
        self.name = ""
        self.require_preds_df = True 

    def calculate(self, predictions, relation):
        pass

    def get_name(self):
        return self.name

class DemographicParity(Measurement):
    def __init__(self):
        super(DemographicParity, self).__init__()
        self.name = "demographic_parity"

    def calculate_one_relation(self, preds_df, relation):
        """
        Calculate demographic parity for each of the relation tail values
        a.k.a. Group fairness/statistical parity/equal acceptance rate
        A classifier satisfies this definition if subjects in both protected and unprotected groups
        have equal probability of being assigned to the positive predicted class : P(d=1|G=m)=P(d=1|G=f)

        Return:
        demographic_parity_distance: float
        """
        true_target_values = list(set(preds_df.true_tail.values))
        true_target_values = [trgt for trgt in true_target_values if trgt in set(preds_df.pred)]
        attributes = preds_df[relation]
        attribute_set = list(set(attributes))

        if -1 in attribute_set:
            attribute_set.remove(-1) 

        if len(attribute_set) == 0:
            return 0

        demographic_parity_distance = 0
        for attr in attribute_set:
            # sum demographic_parity_distance over all attributes a, by calculating the difference
            # P[y=target_val|a=relation_val] - p[y=target_val|a!=relation_val]
            # for all true target values
            for target_val in true_target_values:
                pred_current_target = (preds_df.pred == target_val)

                given_a = pred_current_target[attributes == attr]
                given_not_a = pred_current_target[attributes != attr]

                prob_y_given_a = sum(given_a)/len(given_a)

                prob_y_given_not_a = sum(given_not_a)/len(given_not_a)
                # Calculate the difference of the probabilities P(y=t|a) and P(y=t|not a),
                # We note that P(y=t|a) + P(y=t|~a) = P(y=t)
                # Therefore |P(y=t|a) - P(y=t|~a)| <= P(y=t)
                # moreover, sum(P(y=t) for all t) = 1
                # So sum(|P(y=t|a) - P(y=t|~a)|  for all t) <= 1
                demographic_parity_distance += abs(prob_y_given_a - prob_y_given_not_a)

        # Normalize the demographic parity distance score, to get a value between 0 and 1
        demographic_parity_distance = demographic_parity_distance/(len(attribute_set) + 1)
        return demographic_parity_distance

    def calculate(self, evaluator, bias_relations):
        """
        Calculate demographic parity distance of possibly biased relations, return a table of demographic parity distances(DPD)
        
        Param:
        =======
        evaluator: bias evaluator
        bias_relations: a list of possibly biased relations to be measured for DPD scores

        Return:
        =======
        dp_df: pandas.DataFrame, a table of DPD scores of input bias_relations
        """
        preds_df = evaluator.predictions
        bias_scores = []
        for r in bias_relations:
            print(f"{r}")
            bias_scores.append(self.calculate_one_relation(preds_df, r))
        dp_df = pd.DataFrame({"relations":bias_relations, "bias_scores":bias_scores})
        return dp_df

    def demographic_parity_for_target_attribute_pair(self, preds_df, relation, attr, target_val):
        attributes = preds_df[relation]
        pred_current_target = (preds_df.pred == target_val)

        given_a = pred_current_target[attributes == attr]
        given_not_a = pred_current_target[attributes != attr]

        prob_y_given_a = sum(given_a) / len(given_a)
        prob_y_given_not_a = sum(given_not_a) / len(given_not_a)
        return abs(prob_y_given_a - prob_y_given_not_a)

    def demographic_parity_for_target(self, preds_df, relation, target_val):
        attributes = preds_df[relation]
        attribute_set = list(set(attributes))
        pred_current_target = (preds_df.pred == target_val)
        DP = 0
        if -1 in attribute_set:
            attribute_set.remove(-1)

        if len(attribute_set) == 0:
            return 0

        for attr in attribute_set:
            given_a = pred_current_target[attributes == attr]
            given_not_a = pred_current_target[attributes != attr]

            prob_y_given_a = sum(given_a) / len(given_a)
            prob_y_given_not_a = sum(given_not_a) / len(given_not_a)
            DP += abs(prob_y_given_a - prob_y_given_not_a)
        return DP

class PredictiveParity(Measurement):
    def __init__(self):
        super(PredictiveParity, self).__init__()
        self.name = "predictive_parity"

    def calculate_one_relation(self, preds_df, rel):
        """
        Predictive parity (a.k.a. outcome test)
        A classifier satisfies this definition if both protected and unprotected groups
        have equal PPV – the probability of a subject with positive predictive value to
        truly belong to the positive class : P(Y=1|d=1,G=m)=P(Y=1|d=1,G=f)

        Return:
        predictive_parity_distance: float
        """
        attributes = preds_df[rel].values
        attribute_set = list(set(attributes))

        if -1 in attribute_set:
            attribute_set.remove(-1)

        if len(attribute_set) <= 1:
            return 0

        predictive_parity_distance = 0
        for attr in attribute_set:
            # sum over all attributes, i.e. tail values for the relation,
            # by calculating the difference
            # E[y=target_val|ytrue=target_val, a=relation_val] - E[y==target_val|ytrue=target_val, a!=relation_val]
            # for all target values
            given_a = preds_df[preds_df[rel] == attr]
            given_not_a = preds_df[np.logical_and(preds_df[rel] != attr,preds_df[rel] != -1)]

            precision_given_a = precision_score(given_a.true_tail, given_a.pred,average='micro')
            precision_given_not_a = precision_score(given_not_a.true_tail, given_not_a.pred, average='micro')
            predictive_parity_distance += abs(precision_given_a - precision_given_not_a)

        predictive_parity_distance = predictive_parity_distance/(len(attribute_set))
        return predictive_parity_distance
    
    def calculate(self, evaluator, bias_relations):
        """
        Calculate the predictiive parity distance of each possibly biased relation, return a table of predictive parity distances(PPD)
        
        Param:
        =======
        evaluator: bias evaluator
        bias_relations: a list of possibly biased relations to be measured for PPD scores

        Return:
        =======
        dp_df: pandas.DataFrame, a table of PPD scores of input bias_relations
        """
        preds_df = evaluator.predictions
        bias_scores = []
        for r in bias_relations:
            print(f"{r}")
            bias_scores.append(self.calculate_one_relation(preds_df, r))
        pp_df = pd.DataFrame({"relations":bias_relations, "bias_scores":bias_scores})
        return pp_df
    
class TranslationalLikelihood(Measurement):
    """
    Given a dataset D, its trained embeddings E, embedding's score function f, calculate the difference of f between
    new score and old score, where the new score is achieved by new head entity embedding. The new head embedding
    is calculated by translating it toward a value of bias relatioin

    Check the original paper at: https://arxiv.org/abs/1912.02761
    """
    def __init__(self):
        super(TranslationalLikelihood, self).__init__()
        self.name = "translational"
        self.require_preds_df = False
        
    def get_entities(self, dataset, relation):
        """
        Get all head and tail entities for the given relation
        Param:
        =======
        dataset: pykeen Dataset
        relation: str, a relation label in the above dataset parameter

        Return:
        =======
        heads: list, all head entities that has the input relation
        tails: list, all tail entites that the input relation links to
        """
        heads = []
        tails = []
        for triple in dataset.testing.get_triples_for_relations(relation):
            heads.append(triple[0])
            tails.append(triple[2])
        return heads, tails

    def get_embedding(self, dataset, model, label, is_rel):
        """
        Get the embedding vector for a relation or an entity
        Param:
        =======
        dataset: pykeen Dataset
        model: pykeen Model, trained embedding model
        label: str, the label of entity or relation
        is_rel: bool, True if getting the embedding for a relation, otherwiser getting the
                      embedding for an entity
        
        Return:
        =======
        embedding: the query embedding vector
        """
        if is_rel:
            id = dataset.relation_to_id[label]
            embedding = model.relation_embeddings.weight[id]
        else:
            id = dataset.entity_to_id[label]
            embedding = model.entity_embeddings.weight[id]
        return embedding

    def update_head_embedding(self, h, r, t_list, score_fn, lr):
        """
        One-step gradient descent update on the head entity embedding
        
        Param:
        =======
        h: int or tensor, head embedding
        r: int or tensor, relation embedding
        t_list: list of int or list of tensors, a list of tail embeddings
        score_fn: function, embedding's score function
        lr: float, learning rate of gradient descent step

        Return:
        =======
        new_em: the updated head entity embedding
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if type(h) is int:
            self.model.entity_embeddings.weight.requires_grad_(True)
            triple_0 = torch.LongTensor([[h, r, t_list[0]]]).to(device)
            triple_1 = torch.LongTensor([[h, r, t_list[1]]]).to(device)
            delta = score_fn(triple_0) - score_fn(triple_1)
            delta.backward()
            gradient = self.model.entity_embeddings.weight.grad[h]
            new_em = self.model.entity_embeddings.weight[h].detach() + lr*gradient
            self.model.entity_embeddings.weight = torch.nn.Parameter(self.model.entity_embeddings.weight.clone().detach(), requires_grad=False)
        elif type(h) is torch.Tensor:
            h = h.detach().requires_grad_(True)
            r = r.detach()
            t_list = [t.detach() for t in t_list]
            delta = score_fn(h, r, t_list[0]) - score_fn(h, r, t_list[1])
            delta.backward()
            gradient = h.grad #TODO
            new_em = h + lr*gradient
        else:
            raise ValueError("h must be either int (is using pre-defined pykeen models) or tensor")
        return new_em

    def calc_bias_on_instr_tail(self, h, new_h_em, instr_rel, instr_t, score_fn):
        """
        Calculated the difference of embedding score function on the same triple with the original and updated head entity embedding

        Param:
        =======
        h: torch.Tensor, original embedding of head entity
        new_h_em: torch.Tensor, new embeddiing of head entity
        instr_rel: torch.Tensor, embedding of instrumental relation
        instr_t: torch.Tensor, embedding of tail entity
        score_fn: function, the score function associated with embedding method

        Return:
        =======
        score_delta: score difference
        """
        if type(h) is int:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            oldscore = score_fn(torch.LongTensor([[h, instr_rel, instr_t]]).to(device))
            old_em = self.model.entity_embeddings.weight[h].clone()   
            self.model.entity_embeddings.weight[h] = new_h_em
            newscore = score_fn(torch.LongTensor([[h, instr_rel, instr_t]]).to(device))
            self.model.entity_embeddings.weight[h] = old_em
        elif type(h) is torch.Tensor:
            oldscore = score_fn(old_em, instr_rel, instr_t)
            newscore = score_fn(new_em, instr_rel, instr_t)
        else:
            raise ValueError("h must be either int (is using pre-defined pykeen models) or tensor")
        score_delta = newscore - oldscore
        return score_delta

    def calculate_relation(self, dataset, model, target_relation, bias_relation, score_fn=None):
        """
        Calculate the translational likelihood bias score on each head entity having the (possibly) biased relation(bias_relation),
        and save the result table to current directory. 

        Param:
        =======
        dataset: pykeen.Dataset, knowledge graph dataset e.g fb15k-237
        model: pykeen.models, an embedding model #TODO
        target_relation: str, the auxilary relation to measure the score for suspected bias relation (e.g. relation profession for measuring relation gender or ethnicity bias)
        bias_relation: str, the (possibly) bias relation to be measured
        score_fn: score function of the model

        Return:
        =======
        bias: pandas.DataFrame, bias scores for 
        """
        self.model = model
        _, instr_entities = self.get_entities(dataset, [target_relation])
        instr_entities = set(instr_entities)
        
        pre_def_models = (pykeen.models.TransE, 
                          pykeen.models.TransD,
                          pykeen.models.DistMult,
                          pykeen.models.ComplEx,
                          pykeen.models.ConvE,
                          pykeen.models.RotatE)

        if isinstance(model, pre_def_models):
            instr_rel_idx = dataset.relation_to_id[target_relation] 
            sensit_rel_idx = dataset.relation_to_id[bias_relation]
            sensit_heads, sensit_tails = self.get_entities(dataset, [bias_relation])
            # Binarize sensitive tail values: two most popular
            if len(set(sensit_tails)) > 2:
                most_pop_tail = {}
                for t in sensit_tails:
                    most_pop_tail[t] = most_pop_tail.setdefault(t, 0) + 1
                bi_sensit_tails = [i[0] for i in sorted(most_pop_tail.items(), key=lambda x: x[1], reverse=True)[:2]]
                sensit_heads = set([h for i, h in enumerate(sensit_heads) if sensit_tails[i] in bi_sensit_tails])
            elif len(set(sensit_tails)) < 2: 
                raise ValueError(f"The to-be-detect sensitive attribute {set(sensit_tails)} connects to \
                                   a tail entity having less than 2 value types. \
                                   Cannot be used for translational likelihood bias measurement.")
            else:
                bi_sensit_tails = list(set(sensit_tails))
            sensit_tails_idx = [dataset.entity_to_id[i] for i in bi_sensit_tails]
            score_fn = model.score_hrt
            
            bias = {"instrumental_entities":[], bi_sensit_tails[0]:[], bi_sensit_tails[1]:[]}
            print("Num of instrumental entities: {}".format(len(instr_entities)))
            for instr in instr_entities:
                start_time = time.time()
                instr_tail_idx = dataset.entity_to_id[instr]
                bias_score_0 = 0
                bias_score_1 = 0
                for h in sensit_heads:
                    try:
                        h_idx = dataset.entity_to_id[h]
                        new_em_0 = self.update_head_embedding(h_idx, sensit_rel_idx, sensit_tails_idx, score_fn, lr=1e-3)
                        sensit_tails_idx.reverse()
                        new_em_1 = self.update_head_embedding(h_idx, sensit_rel_idx, sensit_tails_idx, score_fn, lr=1e-3)
                        sensit_tails_idx.reverse()
                        bias_score_0 += self.calc_bias_on_instr_tail(h_idx, new_em_0, instr_rel_idx, instr_tail_idx, score_fn)
                        bias_score_1 += self.calc_bias_on_instr_tail(h_idx, new_em_1, instr_rel_idx, instr_tail_idx, score_fn)
                    except KeyError:
                        continue
                end_time = time.time()
                # print("Elapse ~{} mins".format((end_time-start_time)//60))
                bias_score_0 = bias_score_0/len(sensit_heads)
                bias_score_1 = bias_score_1/len(sensit_heads)
                bias["instrumental_entities"].append(instr) 
                bias[bi_sensit_tails[0]].append(bias_score_0.item())
                bias[bi_sensit_tails[1]].append(bias_score_1.item())
                
            bias = pd.DataFrame(bias)
        else:
            if score_fn is None:
                raise NotImplementedError("The model is not an instance of pykeen models, score_fn must be provided")
            # Get relevant embeddings
            sensit_rel_em = self.get_embedding(dataset, model, bias_relation, is_rel=True)
            instr_rel_em =  self.get_embedding(dataset, model, target_relation, is_rel=True)
            sensit_heads, sensit_tails = self.get_entities(dataset, [bias_relation]) #e.g heads: a list of people, tails: [male, female]
            
            if len(set(sensit_tails)) == 2:
                sensit_tails_em = [self.get_embedding(dataset, model, label, is_rel=False) for label in set(sensit_tails)]
                bi_sensit_tails = list(set(sensit_tails))
            elif len(set(sensit_tails)) > 2: # Binarize sensitive tail values: two most popular
                most_pop_tail = {}
                for t in sensit_tails:
                    most_pop_tail[t] = most_pop_tail.setdefault(t, 0) + 1
                bi_sensit_tails = [i[0] for i in sorted(most_pop_tail.items(), key=lambda x: x[1], reverse=True)[:2]]
                sensit_heads = [h for i, h in enumerate(sensit_heads) if sensit_tails[i] in bi_sensit_tails]
                sensit_tails_em = [self.get_embedding(dataset, model, label, is_rel=False) for label in bi_sensit_tails]
            else:
                raise ValueError(f"The to-be-detect sensitive attribute {set(sensit_tails)} connects to \
                                   a tail entity having less than 2 value types. \
                                   Cannot be used for translational likelihood bias measurement.") 
            for instr in instr_entities:
                instr_tail_em = self.get_embedding(dataset, model, instr, is_rel=False)
                bias_score_0 = 0
                bias_score_1 = 0
                for h in sensit_heads:
                    try:
                        old_em = self.get_embedding(dataset, model, h,False)
                        new_em_0 = self.update_head_embedding(old_em, sensit_rel_em, sensit_tails_em, score_fn, lr=1e-3)
                        sensit_tails_em.reverse()
                        new_em_1 = self.update_head_embedding(old_em, sensit_rel_em, sensit_tails_em, score_fn, lr=1e-3)
                        sensit_tails_em.reverse()
                        bias_score_0 += self.calc_bias_on_instr_tail(old_em, new_em_0, instr_rel_em, instr_tail_em, score_fn)
                        bias_score_1 += self.calc_bias_on_instr_tail(old_em, new_em_1, instr_rel_em, instr_tail_em, score_fn)
                    except KeyError:
                        continue
                bias_score_0 = bias_score_0/len(sensit_heads)
                bias_score_1 = bias_score_1/len(sensit_heads)
                bias[instr] = (bias_score_0, bias_score_1)
            bias = pd.DataFrame(bias, columns = bi_sensit_tails)
        bias.to_csv("./{}_{}.csv",format(bias_relation, str(model).split("(")[0]))
        return bias
        
    def calculate(self, evaluator, bias_relations):
        """
        Iterate over a list of relations of interest to calculate the translational likelihood bias
        metrics 
        
        evaluator: Evaluator (see BiasEvaluator.py)
        bias_relations: list of str, a list of relations needed to calculate bias scores
        """
        # Init
        dataset = evaluator.dataset
        bias_relations = evaluator.bias_relations
        target_rel = evaluator.target_relation
        trained_model = evaluator.trained_model 
        # Iterate over bias relations
        result = {}
        for r in bias_relations:
            print(f"{r}")
            result[r] = self.calculate_relation(dataset, trained_model, target_rel, r)
        return result
