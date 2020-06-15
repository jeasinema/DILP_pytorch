from __future__ import print_function, division, absolute_import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict, defaultdict
import pandas as pd
import os
from core.rules import RulesManager
from core.clause import Predicate
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter


class Agent(object):
    def __init__(self, rules_manager, ilp, cuda=False):
        self.cuda = False

        self.rules_manager = rules_manager

        self.rule_weights = OrderedDict() # dictionary from predicates to rule weights matrices
        self.__init__rule_weights()
        self.ground_atoms = rules_manager.all_grounds
        self.base_valuation = self.axioms2valuation(ilp.background)
        self.training_data = OrderedDict() # index to label
        self.__init_training_data(ilp.positive, ilp.negative)

    def __init__rule_weights(self):
        for predicate, clauses in self.rules_manager.all_clauses.items():
            # TODO: 2 clauses
            self.rule_weights[str(predicate)] = nn.ParameterList()
            for i in range(len(clauses)):
                self.rule_weights[str(predicate)].append(nn.Parameter(torch.randn(
                    len(clauses[i]))))
            # self.rule_weights[str(predicate)] = nn.Parameter(torch.randn(len(clauses[0]),
            #     len(clauses[1])).float())
        if self.cuda:
            for key in self.rule_weights.keys():
                self.rule_weights[key] = self.rule_weights[key].cuda()

    def get_definition(self, topk=2):
        log_text = ''
        result = defaultdict(list)
        for predicate, clauses in self.rules_manager.all_clauses.items():
            # each predicate
            rule_weights = self.rule_weights[str(predicate)]
            for weights in rule_weights:
                # each rule template
                tmp = F.softmax(weights.detach(), dim=-1)
                topk_index = torch.topk(tmp, min(topk, len(tmp)))[1]
                result[predicate].append(topk_index)
            log_text += '{}  \n'.format(str(predicate))
            for i in range(len(result[predicate][0])):
                # each topk program
                # TODO: 2 clauses
                log_text += "weight is {}  \n".format([w[i].item() for w in rule_weights])
                for j in range(len(result[predicate])):
                    # each rule template
                    log_text += '{}  \n'.format(str(
                        clauses[j][result[predicate][j][i]]))
            log_text += '{}  \n'.format('=======')

        return log_text, result

    def __init_training_data(self, positive, negative):
        for i, atom in enumerate(self.ground_atoms):
            if atom in positive:
                self.training_data[i] = 1.0
            elif atom in negative:
                self.training_data[i] = 0.0

    def axioms2valuation(self, axioms):
        '''
        :param axioms: list of Atoms, background knowledge
        :return: a single valuation vector [num_of_all_grounds]
        '''
        result = np.zeros(len(self.ground_atoms), dtype=np.float32)
        for i, atom in enumerate(self.ground_atoms):
            if atom in axioms:
                result[i] = 1.0
        return result

    def valuation2atoms(self, valuation, threshold=0.5):
        '''
        :param valuation: a single valuation vector [num_of_all_grounds]
        '''
        result = OrderedDict()
        for i, value in enumerate(valuation):
            if value >= threshold:
                result[self.ground_atoms[i]] = float(value)
        return result

    def deduction(self, state=None):
        '''
        :param state: list of list(of Atoms), extra facts input
        '''
        # takes background (and extra facts as state) as input and return a valuation of target ground atoms
        bs = 1 if not state else len(state)
        valuation = np.stack([self.base_valuation for _ in range(bs)])
        if state:
            state_valuation = np.stack([self.axioms2valuation(state[i])
                for i in range(bs)])
            valuation += state_valuation
        # normalization
        valuation = np.clip(valuation, 0, 1)

        valuation = torch.from_numpy(valuation)
        if self.cuda:
            valuation = valuation.cuda()
        for _ in range(self.rules_manager.program_template.forward_n):
            valuation = self.inference_step(valuation)
        return valuation

    def inference_step(self, valuation):
        '''
        :param valuation: a mini-batch valuation vector
            [batch_size, num_of_all_grounds]
        '''
        deduced_valuation = torch.zeros_like(valuation)
        #?? deduction_matrices = self.rules_manager.deducation_matrices[predicate]
        for predicate, matrix in self.rules_manager.deduction_matrices.items():
            deduced_valuation += Agent.inference_single_predicate(valuation, matrix,
                    self.rule_weights[str(predicate)])
        return torch.clamp(deduced_valuation+valuation, max=1)
        #?? return deduced_valuation+valuation - deduced_valuation*valuation

    @staticmethod
    def inference_single_predicate(valuation, deduction_matrices, rule_weights):
        '''
        :param valuation: a mini-batch valuation vector
            [batch_size, num_of_all_grounds]
        :param deduction_matrices: list of list of matrices
        :param rule_weights: list of tensor, shape (number_of_rule_temps, number_of_clauses_generated)
        :return:
        '''
        # TODO: 2 clauses
        result_valuations = [[] for _ in rule_weights]
        for i in range(len(result_valuations)):
            for matrix in deduction_matrices[i]:
                result_valuations[i].append(Agent.inference_single_clause(
                    valuation, matrix))

        c_p = None
        for i in range(len(result_valuations)):
            valuations = torch.stack(result_valuations[i])
            prob_rule_weights = F.softmax(rule_weights[i], dim=-1)[:, None, None]
            if c_p==None:
                c_p = torch.sum(prob_rule_weights*valuations, dim=0)
            else:
                c_p = prob_sum(c_p, torch.sum(prob_rule_weights*valuations,
                    dim=0))
        return c_p

    @staticmethod
    def inference_single_clause(valuation, X):
        '''
        The F_c in the paper
        :param valuation: a mini-batch valuation vector
            [batch_size, num_of_all_grounds]
        :param X: array, size (number)
        :return: tensor, size (number_of_ground_atoms)
        '''
        # e.g.
        # pred(X) :- A(X), B(X).
        #   then size(X) = (num_of_ground_atoms, 1, 2)
        # pred(X) :- A(X, Y), B(X).
        #   then size(X) = (num_of_ground_atoms, num_of_constants, 2)

        # TODO: 2 atoms
        # TODO: 2 arity
        X1 = X[:, :, 0]
        X2 = X[:, :, 1]
        Y1 = valuation.T[X1]
        Y2 = valuation.T[X2]
        Z = Y1*Y2
        return torch.max(Z, dim=1)[0].T

    def loss(self, batch_size=-1):
        labels = np.array(list(self.training_data.values()), dtype=np.float32)

        labels = torch.from_numpy(labels)
        if self.cuda:
            labels = labels.cuda()

        index = torch.from_numpy(np.array(list(self.training_data.keys()), dtype=np.int32)).long()
        if self.cuda:
            index = index.cuda()
        outputs = self.deduction()[0][index]
        if batch_size>0:
            index = torch.from_numpy(np.random.randint(0, len(labels), batch_size))
            if self.cuda:
                index = index.cuda()
            labels = labels[index]
            outputs = outputs[index]

        loss = -torch.mean(labels*torch.log(outputs+1e-10)+(1-labels)*torch.log(1-outputs+1e-10))

        return loss

    def grad(self):
        loss_value = self.loss(-1)
        weight_decay = 0.0
        regularization = 0
        for weights in self.__all_variables():
            weights = F.softmax(weights, dim=-1)
            regularization += torch.sum(torch.sqrt(weights))*weight_decay
        loss_value += regularization/len(self.__all_variables())
        return torch.autograd.grad(loss_value, self.__all_variables())

    def loss_torch(self):
        orig_loss_value = self.loss(-1)
        weight_decay = 0.0
        regularization = 0
        for weights in self.__all_variables():
            weights = F.softmax(weights, dim=-1)
            regularization += torch.sum(torch.sqrt(weights))*weight_decay
        aug_loss_value = orig_loss_value + regularization/len(self.__all_variables())
        return orig_loss_value, aug_loss_value

    def __all_variables(self):
        return [w for weights in self.rule_weights.values() for w in weights]

    def train(self, steps=300, name=None):
        """
        :param steps:
        :param name:
        :return: the loss history
        """
        if name:
            checkpoint_dir = "./model_pt/"+name
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            try:
                self.rule_weights = torch.load(checkpoint_prefix)
            except Exception as e:
                print(e)
            if self.cuda:
                for key in self.rule_weights.keys():
                    self.rule_weights[key] = self.rule_weights[key].cuda()

        losses = []

        optimizer = torch.optim.RMSprop(self.__all_variables(), lr=0.1)
        writer = SummaryWriter(checkpoint_dir)

        for i in range(steps):
            optimizer.zero_grad()
            loss_orig, loss_aug = self.loss_torch()
            loss_aug.backward()
            optimizer.step()
            loss_avg = float(loss_orig.item())

            losses.append(loss_avg)
            print("-"*20)
            print("step "+str(i)+" loss is "+str(loss_avg))
            writer.add_scalar('loss', loss_avg, i)
            if i%5==0:
                log_text, program = self.get_definition()
                writer.add_text('program', log_text, i)
                if name:
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    if self.cuda:
                        tmp = OrderedDict()
                        for key in self.rule_weights.keys():
                            tmp[key] = self.rule_weights[key].cpu()
                        torch.save(tmp, checkpoint_prefix)
                    else:
                        torch.save(self.rule_weights, checkpoint_prefix)

                    pd.Series(np.array(losses)).to_csv(name+".csv")
            print("-"*20+"\n")
        return losses


def prob_sum(x, y):
    return x + y - x*y
