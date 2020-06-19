from __future__ import print_function, division, absolute_import
from core.clause import *
from core.ilp import *
from core.rules import *
from core.induction import *
from core.clause import str2atom,str2clause

def setup_predecessor():
    constants = [str(i) for i in range(10)]
    background = [Atom(Predicate("succ", 2), [constants[i], constants[i + 1]]) for i in range(9)]
    positive = [Atom(Predicate("predecessor", 2), [constants[i], constants[i+2]]) for i in range(8)]
    all_atom = [Atom(Predicate("predecessor", 2), [constants[i], constants[j]]) for i in range(10) for j in range(10)]
    negative = list(set(all_atom) - set(positive))

    language = LanguageFrame(Predicate("predecessor",2), [Predicate("succ",2)], constants)
    ilp = ILP(language, background, positive, negative)
    program_temp = ProgramTemplate([], {Predicate("predecessor", 2): [RuleTemplate(1, False, 3), RuleTemplate(0, False)]},
                                   4)
    man = RulesManager(language, program_temp)
    return man, ilp

def setup_planning():
    constants = ['blocka', 'blockb', 'blockc', 'sita', 'sitb']
    ontable = Predicate('ontable', 2)
    clear = Predicate('clear', 2)
    on = Predicate('on', 3)
    inprecond = Predicate('inprecond', 2)

    background = [
        Atom(ontable, ['blocka', 'sita']),
        Atom(on, ['blockb', 'blocka', 'sita']),
        Atom(clear, ['blockb', 'sita']),
        Atom(ontable, ['blockc', 'sitb']),
        Atom(ontable, ['blockb', 'sitb']),
        Atom(on, ['blocka', 'blockc', 'sitb']),
        Atom(clear, ['blockb', 'sitb']),
        Atom(clear, ['blocka', 'sitb']),
    ]
    positive = [
        Atom(inprecond, ['blockb', 'sita']),
        Atom(inprecond, ['blocka', 'sitb']),
    ]
    negative = [
        Atom(inprecond, ['blockb', 'sitb']),
        Atom(inprecond, ['blocka', 'sita']),
        Atom(inprecond, ['blockc', 'sitb']),
    ]
    language = LanguageFrame(inprecond, [ontable, clear, on], constants)
    ilp = ILP(language, background, positive, negative)
    program_temp = ProgramTemplate([], {
        # inprecond: [RuleTemplate(1, False), RuleTemplate(0, False),]
        # inprecond: [RuleTemplate(2, False), RuleTemplate(1, False), RuleTemplate(0, False),]
        inprecond: [RuleTemplate(1, False, 3),]
        }, 4)
    man = RulesManager(language, program_temp)
    return man, ilp

def setup_fizz():
    constants = [str(i) for i in range(10)]
    succ = Predicate("succ", 2)
    zero = Predicate("zero", 1)
    fizz = Predicate("fizz", 1)
    pred1 = Predicate("pred1", 2)
    pred2 = Predicate("pred2", 2)

    background = [Atom(succ, [constants[i], constants[i + 1]]) for i in range(9)]
    background.append(Atom(zero, "0"))
    positive = [Atom(fizz, [constants[i]]) for i in range(0, 10, 3)]
    all_atom = [Atom(fizz, [constants[i]]) for i in range(10)]
    negative = list(set(all_atom) - set(positive))
    language = LanguageFrame(fizz, [zero, succ], constants)
    ilp = ILP(language, background, positive, negative)
    program_temp = ProgramTemplate([pred1, pred2], {fizz: [RuleTemplate(1, True), RuleTemplate(1, False)],
                                                    pred1: [RuleTemplate(1, True),],
                                                    pred2: [RuleTemplate(1, True),],},
                                   10)
    man = RulesManager(language, program_temp)
    return man, ilp

def setup_even():
    constants = [str(i) for i in range(10)]
    succ = Predicate("succ", 2)
    zero = Predicate("zero", 1)
    target = Predicate("even", 1)
    pred = Predicate("pred", 2)
    background = [Atom(succ, [constants[i], constants[i + 1]]) for i in range(9)]
    background.append(Atom(zero, "0"))
    positive = [Atom(target, [constants[i]]) for i in range(0, 10, 2)]
    all_atom = [Atom(target, [constants[i]]) for i in range(10)]
    negative = list(set(all_atom) - set(positive))
    language = LanguageFrame(target, [zero, succ], constants)
    ilp = ILP(language, background, positive, negative)
    program_temp = ProgramTemplate([pred], {target: [RuleTemplate(1, True), RuleTemplate(1, False)],
                                            pred: [RuleTemplate(1, True),RuleTemplate(1, False)],
                                            },
                                   10)
    man = RulesManager(language, program_temp)
    return man, ilp

def start_DILP(task, name):
    # import tensorflow as tf
    # tf.enable_eager_execution()
    if task == "predecessor":
        man, ilp = setup_predecessor()
    elif task == "even":
        man, ilp = setup_even()
    elif task == 'planning':
        man, ilp = setup_planning()
    agent = Agent(man, ilp)
    return agent.train(name=name)[-1]

if __name__ == "__main__":
    # start_DILP("predecessor", "predecessor0")
    # start_DILP("even", "even0")
    start_DILP("planning", "planning10")
