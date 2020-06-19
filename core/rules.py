from __future__ import print_function, division, absolute_import
import numpy as np
from itertools import product
from core.ilp import *
from core.clause import *
from collections import defaultdict
from itertools import zip_longest

class RulesManager():
    def __init__(self, language_frame, program_template):
        self.__language = language_frame
        self.program_template = program_template

        self.__predicate_mapping = {} # map from predicate to ground atom indices
        self.all_grounds = []
        self.__generate_grounds()
        self.all_clauses = defaultdict(list) # dictionary of (invented + target) predicate to list(2d) of lists of clause.
        self.__init_all_clauses()
        self.deduction_matrices =defaultdict(list) # dictionary of predicate to list of lists of deduction matrices.
        self.__init_deduction_matrices()

    def __init_all_clauses(self):
        intensionals = self.__language.target + self.program_template.auxiliary
        for intensional in intensionals:
            for i in range(len(self.program_template.rule_temps[intensional])):
                self.all_clauses[intensional].append(self.generate_clauses(intensional,
                    self.program_template.rule_temps[intensional][i]))

    def __init_deduction_matrices(self):
        for intensional, clauses in self.all_clauses.items():
            for row in clauses:
                row_matrices = []
                for clause in row:
                    row_matrices.append(self.generate_induction_matrix(clause))
                self.deduction_matrices[intensional].append(row_matrices)


    def generate_clauses(self, intensional, rule_template):
        base_variable = tuple(range(intensional.arity))
        head = (Atom(intensional,base_variable),)

        body_variable = tuple(range(intensional.arity+rule_template.variables_n))
        if rule_template.allow_intensional:
            # TODO: if target == action, we can use some heuristics here and remove the last union
            # as target/action cannot present in body
            predicates = list(set(self.program_template.auxiliary).union((self.__language.extensional)).union(set([intensional])))
        else:
            predicates = self.__language.extensional
        terms = []
        for predicate in predicates:
            # WARN: 0/1/2 arity, large arity will implicitly suggest more free variables,
            # thus also enlarge this for-loop
            body_variables = [body_variable for _ in range(predicate.arity)]
            terms += self.generate_body_atoms(predicate, *body_variables)
        # WARN: 2 atoms, result_tuples can be too many
        result_tuples = list(product(head, *[terms for _ in range(rule_template.atoms_n)]))
        # result_tuples = list(product(head, terms, terms)
        print('======= Total clauses: {} ======='.format(len(result_tuples)))
        pruned = self.prune([Clause(result[0], result[1:]) for result in result_tuples])
        print('======= Total clauses(pruned): {} ======='.format(len(pruned)))

    def find_index(self, atom):
        '''
        find index for a ground atom
        :param atom:
        :return:
        '''
        for term in atom.terms:
            assert isinstance(term, str)
        all_indexes = self.__predicate_mapping[atom.predicate]
        for index in all_indexes:
            if self.all_grounds[index] == atom:
                return index
        raise ValueError("didn't find {} in all ground atoms".format(atom))

    def generate_induction_matrix(self, clause):
        '''
        :param clause:
        :return: array of size (number_of_ground_atoms, max_satisfy_paris, 2)
        '''
        satisfy = []
        for atom in self.all_grounds:
            if clause.head.predicate == atom.predicate:
                satisfy.append(self.find_satisfy_by_head(clause, atom))
            else:
                satisfy.append([])
        X = np.empty(find_shape(satisfy), dtype=np.int32)
        fill_array(X, satisfy)
        return X

    def find_satisfy_by_head(self, clause, head):
        '''
        find combination of ground atoms that can trigger the clause to get a specific conclusion (head atom), for free variable, an existential quantifier is applied.
        :param clause:
        :param head:
        :return: list of tuples of indexes
        '''
        result = [] #list of paris of indexes
        free_body = clause.replace_by_head(head).body
        free_variables = set()
        for i in range(len(free_body)):
            free_variables = free_variables.union(free_body[i].variables)
        free_variables = list(free_variables)

        repeat_constatns = [self.__language.constants for _ in free_variables]
        # WARN: 0/1/2 arity, large arity will implicitly suggest more free variables,
        # thus also enlarge this for-loop
        all_constants_combination = product(*repeat_constatns)
        all_match = []
        for combination in all_constants_combination:
            all_match.append({free_variables[i]:constant for i,constant in enumerate(combination)})
        for match in all_match:
            result.append(tuple([self.find_index(i.replace(match)) for i in free_body]))
        return result

    def __generate_grounds(self):
        self.all_grounds.append(Atom(Predicate("Empty", 0), []))
        self.__predicate_mapping[Predicate("Empty", 0)] = [0]
        all_predicates = self.__language.extensional+self.__language.target+self.program_template.auxiliary
        for predicate in all_predicates:
            constant = self.__language.constants
            constants = [constant for _ in range(predicate.arity)]
            grounds = self.generate_body_atoms(predicate, *constants)
            start = len(self.all_grounds)
            self.all_grounds += grounds
            end = len(self.all_grounds)
            self.__predicate_mapping[predicate] = list(range(start, end))

    @staticmethod
    def prune(clauses):
        pruned = []
        def not_unsafe(clause):
            head_variables = set(clause.head.terms)
            body_variables = sum(tuple([clause.body[i].terms for i in
                range(len(clause.body))]), ())
            return head_variables.issubset(body_variables)

        def not_circular(clause):
            '''
            e.g. pred(X) :- pred(X),...
            '''
            return clause.head not in clause.body

        def not_duplicated(clause):
            '''
            e.g. pred(X) :- A(X),B(X) <-> pred(X) :- B(X),A(X)
            '''
            for pruned_caluse in pruned:
                if tuple(reversed(pruned_caluse.body)) == clause.body:
                    return False
                if str(clause)==str(pruned_caluse):
                    return False
            return True

        def not_recursive(clause):
            '''
            [optional]
            e.g. pred(X) :- A(X, Y),pred(Y)
            '''
            for body_atom in clause.body:
                if body_atom.predicate == clause.head.predicate:
                    return False
            return True

        def follow_order(clause):
            '''
            e.g. pred(Y) :- A(X)
            '''
            symbols = OrderedSet()
            for atom in clause.atoms:
                for term in atom.terms:
                    symbols.add(term)
            max_v = 0
            for term in symbols:
                if isinstance(term, int):
                    if term>=max_v:
                        max_v = term
                    else:
                        return False
            return True

        def no_insertion(clause):
            '''
            e.g. pred(X,Z) :- A(X),B(Z)
            '''
            symbols = OrderedSet()
            for atom in clause.atoms:
                for term in atom.terms:
                    symbols.add(term)
            symbols = list(symbols)
            if len(symbols) == max(symbols) - min(symbols)+1:
                return True
            else:
                return False

        def no_repeat(clause):
            '''
            e.g. pred(X,Y) :- A(X,Y), A(X, Y)
            Notice: sometimes a repeated clause will be the only valid one
            when atoms_n of the template is too large
            '''
            return len(clause.atoms) == len(set([str(i) for i in clause.atoms]))

        for clause in clauses:
            if follow_order(clause) and \
                not_unsafe(clause) and \
                no_insertion(clause) and \
                not_circular(clause) and \
                not_duplicated(clause):
                # not_recursive(clause):
                # no_repeat(clause):
                pruned.append(clause)
        return pruned


    @staticmethod
    def generate_body_atoms(predicate, *variables):
        '''
        :param predict_candidate: string, candiate of predicate
        :param variables: iterable of tuples of integers, candidates of variables at each position
        :return: tuple of atoms
        '''
        result_tuples = product((predicate,), *variables)
        atoms = [Atom(result[0], result[1:]) for result in result_tuples]
        return atoms

# from https://stackoverflow.com/questions/27890052
def find_shape(seq):
    try:
        len_ = len(seq)
    except TypeError:
        return ()
    shapes = [find_shape(subseq) for subseq in seq]
    return (len_,) + tuple(max(sizes) for sizes in zip_longest(*shapes,
                                                                fillvalue=1))

def fill_array(arr, seq):
    if arr.ndim == 1:
        try:
            len_ = len(seq)
        except TypeError:
            len_ = 0
        arr[:len_] = seq
        arr[len_:] = 0
    else:
        for subarr, subseq in zip_longest(arr, seq, fillvalue=()):
            fill_array(subarr, subseq)

import collections

class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)
