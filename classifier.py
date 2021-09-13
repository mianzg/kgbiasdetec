import pprint
import operator
import numpy as np
import pandas as pd
from collections import Counter
from pykeen.datasets import FB15k237
##### Ignite
from ignite.contrib.handlers import ProgressBar
#from ignite.contrib.metrics import ROC_AUC
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Precision, Recall, RunningAverage
#### Torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
#### Internal Imports
from classifier_models import MLP

class TargetRelationClassifier:

    def __init__(self, dataset,
                 embedding_model_path,
                 target_relation,
                 num_classes,
                 batch_size=200,
                 lr = 0.01,
                 model_type='mlp',
                 **model_kwargs,
                 ):
        """"
        embedding_model_path : path to the kg embedding that will be used
        num_classes : number of labels to consider. The classifier will learn
            to predict the (num_classes -1) most frequent labels,
            and consider all the rest to be of class OTHER
        hidden_layer_sizes
        batch_size : the batch size used when training
        """
        self.OTHER = 41414141

        self._device = self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._dataset = dataset

        self.num_classes = num_classes

        self._batch_size = batch_size
        self.set_data_loaders(target_relation=target_relation)
        self.set_target_labels()

        self.binary = (num_classes == 2)
        self.set_loss(binary=self.binary)

        self._model = torch.load(embedding_model_path, map_location=self._device)
        self.set_classifier(model_type,**model_kwargs)

        self._optimizer = torch.optim.Adam(self._classifier.parameters(),lr=lr)#self._classifier.parameters())
        self._target = target_relation

    def set_classifier(self, type='mlp', **model_kwargs):
        output_layer_size = 1 if self.binary else self.num_classes
        if type == 'mlp' :
            if "hdims" in model_kwargs: hidden_layer_sizes = model_kwargs["hdims"]
            else: hidden_layer_sizes = [256,16]
            all_layer_dims = [self._model.embedding_dim] + hidden_layer_sizes + [output_layer_size]
            self._classifier = MLP(
                all_layer_dims)
        elif type == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            self._classifier = RandomForestClassifier(**model_kwargs)

    def attach_metrics(self):
        avg_loss = RunningAverage(output_transform=lambda x: x)
        avg_loss.attach(self._trainer, 'loss')

        accuracy = Accuracy()
        accuracy.attach(self._evaluator, 'accuracy')

        precision = Precision(average=False)
        precision.attach(self._evaluator, 'precision')

        recall = Recall(average=False)
        recall.attach(self._evaluator, 'recall')

        F1 = (precision * recall * 2 / (precision + recall)).mean()
        F1.attach(self._evaluator, 'F1')

    def set_loss(self, binary):
        if binary:
            # set binary cross entropy loss
            self._loss = nn.BCEWithLogitsLoss()
        else:
            self._loss = nn.CrossEntropyLoss()

    def predict_tails(self, heads, relation):
        if relation != self._target:
            print("predicting tails for wrong relation, prediction is for", self._target)
        heads = heads.to(self._device)
        head_embeddings = self._model.entity_embeddings(heads)
        yhat = self._classifier(head_embeddings).detach()
        if self.binary:
            return torch.round(torch.sigmoid(yhat)).cpu() # need to put in cpu if trained on gpu...
        else:
            return torch.argmax(yhat,1).cpu()

    def set_data_loaders(self,target_relation):
        relations = [target_relation]

        self.train_triples_factory = train_triples_factory = self._dataset.training.new_with_restriction(
            relations=relations
        )
        self._train_loader = DataLoader(
            dataset=train_triples_factory.create_lcwa_instances(),
            batch_size=self._batch_size, shuffle=True
        )
        self.test_triples_factory = test_triples_factory = self._dataset.testing.new_with_restriction(
            relations=relations
        )
        self._test_loader = DataLoader(
            dataset=test_triples_factory.create_lcwa_instances(),
            batch_size=self._batch_size,
            shuffle=False
        )
        self._train_loader.dataset.labels = self._train_loader.dataset.labels.reshape(
                                                    self._train_loader.dataset.labels.shape[0])
        self._test_loader.dataset.labels = self._test_loader.dataset.labels.reshape(
                                                    self._test_loader.dataset.labels.shape[0])

    def set_target_labels(self):
        train_triples = self.train_triples_factory.triples
        tails = train_triples[:,2]
        tails2keep = self.tails2keep(tails)
        tail_ids = [self._train_loader.dataset.entity_to_id[vl] for vl in tails2keep]
        self._target2int = {idval: k for k, idval in enumerate(tail_ids)}
        self._target2int[self.OTHER] = len(tail_ids)
        self.tailCounts = Counter([self._train_loader.dataset.entity_to_id[vl] for vl in tails])

    def train(self, epochs):

        self._trainer = Engine(self.process_function)
        self._evaluator = Engine(self.eval_function)

        self.attach_metrics()

        self._pbar = ProgressBar(persist=True, bar_format='')
        self._pbar.attach(self._trainer, ['loss'])

        @self._trainer.on(Events.EPOCH_COMPLETED)
        def log_test_results(engine):
            self._evaluator.run(self._test_loader)
            metrics = self._evaluator.state.metrics
            self._pbar.log_message(
                f'Epoch: {engine.state.epoch} \nMetrics:\n {pprint.pformat(metrics)}'
            )

        self._trainer.run(self._train_loader, max_epochs=epochs)

    def process_function(self, engine, batch):
        self._classifier.to(self._device)
        self._classifier.train()

        heads, tails = self.get_heads_tails(engine,batch)
        labels = torch.Tensor([self.target2label(tl) for tl in tails])#
        if self.binary:
            labels = torch.tensor(labels, dtype=torch.float, device=self._device
                                         ) # TODO: binary and multi requires different?
        else:
            labels = torch.tensor(labels, dtype=torch.long, device=self._device
                                         )
        embeddings = self._model.entity_embeddings(heads.to(self._device))
        logits = self._classifier(embeddings)

        ce_loss = self._loss(logits, labels)

        self._optimizer.zero_grad()
        ce_loss.mean().backward()
        self._optimizer.step()
        return ce_loss.item()

    def eval_function(self, engine, batch):
        self._classifier.eval()
        heads, tails = self.get_heads_tails(engine,batch)

        labels = self.targets2labels(tails)
        labels = labels.type(dtype=torch.int64).to(self._device)

        with torch.no_grad():
            embeddings = self._model.entity_embeddings(heads.to(self._device))

            return self._classifier.predict(embeddings), labels

    def get_heads_tails(self, engine, batch):
        """
        Split batch into heads and tails
        """
        data, targets = batch
        data, targets = data, targets
        heads = data[:, 0]

        tails_idx = (targets == 1).nonzero(as_tuple=False)[:, 0]
        tails = (targets == 1).nonzero(as_tuple=False)[:, 1]
        tails_list = [tl.item() for tl in tails]
        if heads.shape != tails.shape:
            # if the number of heads doesn't match the number of tails,
            # Choose one tail per head so
            #tails_list = self.make_one2one(tails_list, tails_idx, batch_size=len(heads))
            heads = self.increase_heads(tails, tails_idx,heads)
        return heads, tails_list

    def increase_heads(self, tails, tails_idx, heads):
        """"
        Make sure each entity corresponds to one tail
        """
        new_heads = np.zeros(tails.shape,)
        for idx,(tail, tail_idx) in enumerate(zip(tails, tails_idx)):
            # if current and previous tail belong to the same target
            new_heads[idx] = heads[tail_idx]
            # choose the more frequent tail
        return torch.LongTensor(new_heads)

    def make_one2one(self, tails, tails_idx, batch_size):
        """"
        Make sure each entity corresponds to one tail
        """
        new_tails = np.zeros(batch_size,)
        prev_idx = -1
        for tail, idx in zip(tails, tails_idx):
            # if current and previous tail belong to the same target
            if idx == prev_idx:
                cur_tail = new_tails[idx]
                # choose the more frequent tail
                if self.tailCounts[cur_tail] <= self.tailCounts[tail]:
                    new_tails[idx] = tail
            else:
                prev_idx = idx
                new_tails[idx] = tail
        return new_tails

    def tails2keep(self, tails):
        """
        Keep the num_classes -1 most frequent tails, relabel the rest as self.OTHER
        """
        # If there are less tail types than num_classes, don't do anything
        if len(set(tails)) <= self.num_classes:
            self.num_classes = len(set(tails))
            return tails
        tail_count = Counter(tails)
        # Choose which tails to keep
        keep = []
        for keep_tail in range(self.num_classes-1):
            # find the most frequent tail, and add to keep
            cur_max = max(tail_count.items(), key=operator.itemgetter(1))[0]
            keep.append(cur_max)
            # delete current max so we'll find a different one in the next iteration
            del tail_count[cur_max]
        return keep

    def targets2labels(self, targets):
        labels = []
        for tail in targets:
            if tail in self._target2int.keys():
                labels.append(int(self._target2int[tail]))
            else:
                labels.append(int(self._target2int[self.OTHER]))
        return torch.Tensor(labels)

    def target2label(self, target):
        if target in self._target2int.keys():
            return int(self._target2int[target])
        else:
            return int(self._target2int[self.OTHER])


class RFRelationClassifier:

    def __init__(self,dataset,
                 target_relation,
                 embedding_model_path,
                 batch_size,
                 num_classes=6,
                 **model_kwargs
                 ):

        self.OTHER = 41414141

        self._dataset = dataset
        self._device = self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_classes = num_classes

        self._batch_size = batch_size
        self.set_data_loaders(target_relation=target_relation)
        self.set_target_labels()

        self.binary = (num_classes == 2)

        self._model = torch.load(embedding_model_path, map_location=self._device)
        self.set_classifier(**model_kwargs)

        self._target = target_relation

    def set_classifier(self, **model_kwargs):
        from sklearn.ensemble import RandomForestClassifier
        self._classifier = RandomForestClassifier(warm_start = True,**model_kwargs)

    def predict_tails(self, heads, relation):
        if relation != self._target:
            print("predicting tails for wrong relation, prediction is for", self._target)
        heads = heads.to(self._device)
        head_embeddings = self._model.entity_embeddings(heads).detach().numpy()
        yhat = self._classifier.predict(head_embeddings)
        if self.binary:
            return torch.round(torch.sigmoid(yhat)).cpu()  # need to put in cpu if trained on gpu...
        else:
            return torch.argmax(yhat, 1).cpu()

    def set_data_loaders(self, target_relation):
        relations = [target_relation]

        self.train_triples_factory = train_triples_factory = self._dataset.training.new_with_restriction(
            relations=relations
        )
        self._train_loader = DataLoader(
            dataset=train_triples_factory.create_lcwa_instances(),
            batch_size=self._batch_size, shuffle=True
        )
        self.test_triples_factory = test_triples_factory = self._dataset.testing.new_with_restriction(
            relations=relations
        )
        self._test_loader = DataLoader(
            dataset=test_triples_factory.create_lcwa_instances(),
            batch_size=self._batch_size,
            shuffle=False
        )
        self._train_loader.dataset.labels = self._train_loader.dataset.labels.reshape(
            self._train_loader.dataset.labels.shape[0])
        self._test_loader.dataset.labels = self._test_loader.dataset.labels.reshape(
            self._test_loader.dataset.labels.shape[0])

    def set_target_labels(self):
        train_triples = self.train_triples_factory.triples
        tails = train_triples[:, 2]
        tails2keep = self.tails2keep(tails)
        tail_ids = [self._train_loader.dataset.entity_to_id[vl] for vl in tails2keep]
        self._target2int = {idval: k for k, idval in enumerate(tail_ids)}
        self._target2int[self.OTHER] = len(tail_ids)
        self.tailCounts = Counter([self._train_loader.dataset.entity_to_id[vl] for vl in tails])

    def train(self):
        for batch in self._train_loader:
            heads, tails = self.get_heads_tails(batch)
            heads = self._model.entity_embeddings(heads.to(self._device)).detach().numpy()
            labels = self.targets2labels(tails)
            labels = labels.type(dtype=torch.int64).to(self._device).detach().numpy()
            self._classifier.n_estimators += 11
            self._classifier.fit(heads, labels)

    def get_heads_tails(self, batch):
        """
        Split batch into heads and tails
        """
        data, targets = batch
        data, targets = data, targets
        heads = data[:, 0]

        tails_idx = (targets == 1).nonzero(as_tuple=False)[:, 0]
        tails = (targets == 1).nonzero(as_tuple=False)[:, 1]
        tails_list = [tl.item() for tl in tails]
        if heads.shape != tails.shape:
            # if the number of heads doesn't match the number of tails,
            # Choose one tail per head so
            # tails_list = self.make_one2one(tails_list, tails_idx, batch_size=len(heads))
            heads = self.increase_heads(tails, tails_idx, heads)
        return heads, tails_list

    def increase_heads(self, tails, tails_idx, heads):
        """"
        Make sure each entity corresponds to one tail
        """
        new_heads = np.zeros(tails.shape, )
        for idx, (tail, tail_idx) in enumerate(zip(tails, tails_idx)):
            # if current and previous tail belong to the same target
            new_heads[idx] = heads[tail_idx]
            # choose the more frequent tail
        return torch.LongTensor(new_heads)

    def make_one2one(self, tails, tails_idx, batch_size):
        """"
        Make sure each entity corresponds to one tail
        """
        new_tails = np.zeros(batch_size, )
        prev_idx = -1
        for tail, idx in zip(tails, tails_idx):
            # if current and previous tail belong to the same target
            if idx == prev_idx:
                cur_tail = new_tails[idx]
                # choose the more frequent tail
                if self.tailCounts[cur_tail] <= self.tailCounts[tail]:
                    new_tails[idx] = tail
            else:
                prev_idx = idx
                new_tails[idx] = tail
        return new_tails

    def tails2keep(self, tails):
        """
        Keep the num_classes -1 most frequent tails, relabel the rest as self.OTHER
        """
        # If there are less tail types than num_classes, don't do anything
        if len(set(tails)) <= self.num_classes:
            self.num_classes = len(set(tails))
            return tails
        tail_count = Counter(tails)
        # Choose which tails to keep
        keep = []
        for keep_tail in range(self.num_classes - 1):
            # find the most frequent tail, and add to keep
            cur_max = max(tail_count.items(), key=operator.itemgetter(1))[0]
            keep.append(cur_max)
            # delete current max so we'll find a different one in the next iteration
            del tail_count[cur_max]
        return keep

    def targets2labels(self, targets):
        labels = []
        for tail in targets:
            if tail in self._target2int.keys():
                labels.append(int(self._target2int[tail]))
            else:
                labels.append(int(self._target2int[self.OTHER]))
        return torch.Tensor(labels)

    def target2label(self, target):
        if target in self._target2int.keys():
            return int(self._target2int[target])
        else:
            return int(self._target2int[self.OTHER])


if __name__ == '__main__':

    fname = "/Users/alacrity/Documents/uni/Fairness/trained_model.pkl"
    # Trained Model Path
    #fname = '/local/scratch/kge_fairness/models/fb15k237/transe_openkeparams_alpha1/replicates/replicate-00000/trained_model.pkl'
    dataset = FB15k237()
    GENDER_RELATION = '/people/person/gender'
    PROFESSION_RELATION = '/people/pearson/profession'


    classifier = TargetRelationClassifier(
                              dataset=dataset,
                              embedding_model_path=fname,
                              target_relation=PROFESSION_RELATION,
                              num_classes=6,
                              hdims=[25,25,25]
                              )

    classifier.train(epochs=10)

    print("hi")
