from __future__ import print_function, division, absolute_import
from core.clause import *
from core.ilp import *
from core.rules import *
from core.induction import *
from core.clause import str2atom,str2clause
# from core.NTP import NeuralProver

def setup_predecessor():
    constants = [str(i) for i in range(10)]
    background = [Atom(Predicate("succ", 2), [constants[i], constants[i + 1]]) for i in range(9)]
    positive = [Atom(Predicate("predecessor", 2), [constants[i], constants[i+2]]) for i in range(8)]
    all_atom = [Atom(Predicate("predecessor", 2), [constants[i], constants[j]]) for i in range(10) for j in range(10)]
    negative = list(set(all_atom) - set(positive))

    language = LanguageFrame(Predicate("predecessor",2), [Predicate("succ",2)], constants)
    ilp = ILP(language, background, positive, negative)
    program_temp = ProgramTemplate([], {Predicate("predecessor", 2): [RuleTemplate(1, False), RuleTemplate(0, False)]},
                                   4)
    man = RulesManager(language, program_temp)
    return man, ilp

def setup_planning():
    constants = ['blocka', 'blockb', 'blockc', 'sita', 'sitb']
    ontable = Predicate('ontable', 2)
    clear = Predicate('clear', 2)
    # on = Predicate('on', 3)
    inprecond = Predicate('inprecond', 2)

    background = [
        Atom(ontable, ['blocka', 'sita']),
        Atom(clear, ['blocka', 'sita']),
        Atom(ontable, ['blockc', 'sitb']),
        # Atom(on, ['blocka', 'blockc', 'sitb']),
        # Atom(clear, ['blocka', 'sitb']),
    ]
    positive = [
        Atom(inprecond, ['blocka', 'sita']),
        # Atom(inprecond, ['blocka', 'sitb']),
    ]
    negative = [
        Atom(inprecond, ['blockc', 'sitb']),
    ]
    # language = LanguageFrame(inprecond, [ontable, clear, on], constants)
    language = LanguageFrame(inprecond, [ontable, clear], constants)
    ilp = ILP(language, background, positive, negative)
    program_temp = ProgramTemplate([], {
        inprecond: [RuleTemplate(1, False), RuleTemplate(0, False),]
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

def start_NTP(task, name=None):
    import tensorflow as tf
    from core.NTP import ProofState
    tf.enable_eager_execution()
    if task == "predecessor":
        man, ilp = setup_predecessor()
        ntp = NeuralProver.from_ILP(ilp, [str2clause("predecessor(X,Y):-s1(X,Z),s2(Z,Y)"),
                                          str2clause("predecessor(X,Y):-s3(X,X),s4(X,Y)"),
                                          str2clause("predecessor(X,Y):-s5(X,X),s6(Y,Y)"),
                                          str2clause("predecessor(X,Y):-s7(X,Y),s8(Y,Y)"),
                                          str2clause("predecessor(X,Y):-s9(Y,X)"),
                                          str2clause("predecessor(X,Y):-s10(Y,Z),s11(Z,X)"),
                                          str2clause("predecessor(X,Y):-s12(X,Y),s13(Z,X)"),
                                          str2clause("predecessor(X,Y):-s14(X,X),s15(Z,Y)"),
                                          str2clause("predecessor(X,Y):-s16(Y,Y),s17(Z,X)"),
                                          str2clause("predecessor(X,Y):-s18(Y,X),s19(Z,Y)"),
                                          ])
    if task == "even":
        man, ilp = setup_even()
        ntp = NeuralProver.from_ILP(ilp, [str2clause("predecessor(X,Y):-s(X,Z),s2(Z,Y)"),
                                          str2clause("even(Y):-p(X,Y),e(X)"),
                                          str2clause("even(X):-z(X)")])
    final_loss = ntp.train(ilp.positive,ilp.negative,2,3000)[-1]
    return final_loss

if __name__ == "__main__":
    # start_DILP("predecessor", "predecessor0")
    # start_DILP("even", "even0")
    start_DILP("planning", "planning0")
