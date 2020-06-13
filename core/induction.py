from __future__ import print_function, division, absolute_import
import numpy as np
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pandas as pd
# import tensorflow.contrib.eager as tfe
import os
from core.rules import RulesManager
from core.clause import Predicate
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter


class Agent(object):
    def __init__(self, rules_manager, ilp, cuda=False):
        self.cuda = False

        self.rules_manager = rules_manager

        self.rule_weights = nn.ParameterDict() # dictionary from predicates to rule weights matrices
        # self.rule_weights = OrderedDict() # dictionary from predicates to rule weights matrices

        self.__init__rule_weights()
        self.ground_atoms = rules_manager.all_grounds
        self.base_valuation = self.axioms2valuation(ilp.background)
        self.training_data = OrderedDict() # index to label
        self.__init_training_data(ilp.positive, ilp.negative)

    def __init__rule_weights(self):
        for predicate, clauses in self.rules_manager.all_clauses.items():
            self.rule_weights[str(predicate)] = nn.Parameter(torch.randn(len(clauses[0]),
                len(clauses[1])).float())
            if self.cuda:
                self.rule_weights[str(predicate)] = self.rule_weights[str(predicate)].cuda()
        # with tf.variable_scope("rule_weights", reuse=tf.AUTO_REUSE):
        #     for predicate, clauses in self.rules_manager.all_clauses.items():
        #         self.rule_weights[predicate] = tf.get_variable(predicate.name + "_rule_weights",
        #                                                        [len(clauses[0]), len(clauses[1])],
        #                                                        initializer=tf.random_normal_initializer,
        #                                                        dtype=tf.float32)

    def show_definition(self):
        log_text = ''
        for predicate, clauses in self.rules_manager.all_clauses.items():
            shape = self.rule_weights[str(predicate)].shape
            rule_weights = torch.reshape(self.rule_weights[str(predicate)], [-1])
            weights = torch.reshape(F.softmax(rule_weights, dim=-1)[:, None], shape)
            # shape = self.rule_weights[predicate].shape
            # rule_weights = tf.reshape(self.rule_weights[predicate] ,[-1])
            # weights = tf.reshape(tf.nn.softmax(rule_weights)[:, None], shape)

            # indexes = torch.nonzero(weights>0.05, as_tuple=True)
            topk = torch.topk(weights.reshape(-1), min(2, len(weights.reshape(-1))))[1]
            indexes = (topk/weights.shape[1], topk%weights.shape[1])
            # indexes = np.nonzero(weights>0.05)

            log_text += '{}  \n'.format(str(predicate))
            for i in range(len(indexes[0])):
                log_text += "weight is {}  \n".format(weights[indexes[0][i], indexes[1][i]])
                log_text += '{}  \n'.format(str(clauses[0][indexes[0][i]]))
                log_text += '{}  \n'.format(str(clauses[1][indexes[1][i]]))
            log_text += '{}  \n'.format('=======')
        return log_text

    def __init_training_data(self, positive, negative):
        for i, atom in enumerate(self.ground_atoms):
            if atom in positive:
                self.training_data[i] = 1.0
            elif atom in negative:
                self.training_data[i] = 0.0


    def axioms2valuation(self, axioms):
        '''
        :param axioms: list of Atoms, background knowledge
        :return: a valuation vector
        '''
        result = np.zeros(len(self.ground_atoms), dtype=np.float32)
        for i, atom in enumerate(self.ground_atoms):
            if atom in axioms:
                result[i] = 1.0
        return result

    def valuation2atoms(self, valuation):
        result = {}
        for i, value in enumerate(valuation):
            if value > 0.01:
                result[self.ground_atoms[i]] = float(value)
        return result

    def deduction(self):
        # takes background as input and return a valuation of target ground atoms
        valuation = self.base_valuation

        valuation = torch.from_numpy(valuation)
        if self.cuda:
            valuation = valuation.cuda()

        for _ in range(self.rules_manager.program_template.forward_n):
            valuation = self.inference_step(valuation)
        return valuation

    def inference_step(self, valuation):
        deduced_valuation = torch.zeros(len(self.ground_atoms))
        if self.cuda:
            deduced_valuation = deduced_valuation.cuda()
        # deduced_valuation = tf.zeros(len(self.ground_atoms))

        # deduction_matrices = self.rules_manager.deducation_matrices[predicate]
        for predicate, matrix in self.rules_manager.deduction_matrices.items():
            deduced_valuation += Agent.inference_single_predicate(valuation, matrix, self.rule_weights[str(predicate)])
            # deduced_valuation += Agent.inference_single_predicate(valuation, matrix, self.rule_weights[predicate])
        return deduced_valuation+valuation - deduced_valuation*valuation

    @staticmethod
    def inference_single_predicate(valuation, deduction_matrices, rule_weights):
        '''

        :param valuation:
        :param deduction_matrices: list of list of matrices
        :param rule_weights: list of tensor, shape (number_of_rule_temps, number_of_clauses_generated)
        :return:
        '''
        result_valuations = [[], []]
        for i in range(len(result_valuations)):
            for matrix in deduction_matrices[i]:
                result_valuations[i].append(Agent.inference_single_clause(valuation, matrix))

        c_p = [] # flattened
        for clause1 in result_valuations[0]:
            for clause2 in result_valuations[1]:
                c_p.append(torch.max(clause1, clause2))
                # c_p.append(tf.maximum(clause1, clause2))

        rule_weights = torch.reshape(rule_weights ,[-1])
        prob_rule_weights = F.softmax(rule_weights, dim=-1)[:, None]
        return torch.sum((torch.stack(c_p)*prob_rule_weights), dim=0)
        # rule_weights = tf.reshape(rule_weights ,[-1])
        # prob_rule_weights = tf.nn.softmax(rule_weights)[:, None]
        # return tf.reduce_sum((tf.stack(c_p)*prob_rule_weights), axis=0)

    @staticmethod
    def inference_single_clause(valuation, X):
        '''
        The F_c in the paper
        :param valuation:
        :param X: array, size (number)
        :return: tensor, size (number_of_ground_atoms)
        '''
        X1 = X[:, :, 0]
        X2 = X[:, :, 1]
        Y1 = valuation[X1]
        Y2 = valuation[X2]
        Z = Y1*Y2
        return torch.max(Z, dim=1)[0]
        # X1 = X[:, :, 0, None]
        # X2 = X[:, :, 1, None]
        # Y1 = tf.gather_nd(params=valuation, indices=X1)
        # Y2 = tf.gather_nd(params=valuation, indices=X2)
        # Z = Y1*Y2
        # return tf.reduce_max(Z, axis=1)

    def loss(self, batch_size=-1):
        labels = np.array(list(self.training_data.values()), dtype=np.float32)

        labels = torch.from_numpy(labels)
        if self.cuda:
            labels = labels.cuda()

        index = torch.from_numpy(np.array(list(self.training_data.keys()), dtype=np.int32)).long()
        if self.cuda:
            index = index.cuda()
        outputs = self.deduction()[index]
        # outputs = tf.gather(self.deduction(), np.array(list(self.training_data.keys()), dtype=np.int32))
        if batch_size>0:
            index = torch.from_numpy(np.random.randint(0, len(labels), batch_size))
            if self.cuda:
                index = index.cuda()
            labels = labels[index]
            outputs = outputs[index]
            # index = np.random.randint(0, len(labels), batch_size)
            # labels = labels[index]
            # outputs = tf.gather(outputs, index)

        loss = -torch.mean(labels*torch.log(outputs+1e-10)+(1-labels)*torch.log(1-outputs+1e-10))
        # loss = -tf.reduce_mean(labels*tf.log(outputs+1e-10)+(1-labels)*tf.log(1-outputs+1e-10))

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
        # with tfe.GradientTape() as tape:
        #     loss_value = self.loss(-1)
        #     weight_decay = 0.0
        #     regularization = 0
        #     for weights in self.__all_variables():
        #         weights = tf.nn.softmax(weights)
        #         regularization += tf.reduce_sum(tf.sqrt(weights))*weight_decay
        #     loss_value += regularization/len(self.__all_variables())
        # return tape.gradient(loss_value, self.__all_variables())

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
        return [weights for weights in self.rule_weights.values()]

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
                self.rule_weights = self.rule_weights.cuda()
        # str2weights = {str(key):value for key,value in self.rule_weights.items()}
        # if name:
        #     checkpoint = tfe.Checkpoint(**str2weights)
        #     checkpoint_dir = "./model/"+name
        #     checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        #     try:
        #         checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        #     except Exception as e:
        #         print(e)

        losses = []

        optimizer = torch.optim.RMSprop(self.rule_weights.values(), lr=0.1)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=0.5)
        writer = SummaryWriter(checkpoint_dir)

        for i in range(steps):
            optimizer.zero_grad()
            loss_orig, loss_aug = self.loss_torch()
            loss_aug.backward()
            optimizer.step()
            loss_avg = float(loss_orig.item())
            # grads = self.grad()
            # optimizer.apply_gradients(zip(grads, self.__all_variables()),
            #                           global_step=tf.train.get_or_create_global_step())
            # loss_avg = float(self.loss().numpy())

            losses.append(loss_avg)
            print("-"*20)
            print("step "+str(i)+" loss is "+str(loss_avg))
            writer.add_scalar('loss', loss_avg, i)
            if i%5==0:
                log_text = self.show_definition()
                writer.add_text('program', log_text, i)
                valuation_dict = self.valuation2atoms(self.deduction()).items()
                # for atom, value in valuation_dict:
                #     print(str(atom)+": "+str(value))
                if name:
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    torch.save(self.rule_weights, checkpoint_prefix)
                    # checkpoint.save(checkpoint_prefix)

                    pd.Series(np.array(losses)).to_csv(name+".csv")
            print("-"*20+"\n")
        return losses


#def prob_sum(x, y):
#    return x + y - x*y

class RLAgent(Agent):
    pass
