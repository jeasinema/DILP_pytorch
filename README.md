# DILP-pytorch
This project aims to build a general framework for DataLog neural program systhesis.

* ∂ILP in "Learning Explanatory Rules from Noisy Data": https://www.jair.org/index.php/jair/article/view/11172

## Features

* arity of a predicate > 2
* number of clauses > 2 (by adding more `RuleTemplate`)
* number of atoms in the body of a cluase > 2 (by providing `atoms_n` to `RuleTemplate`)

Learnin/Inference may significantly slow down due to large arity/number of clauses/number of atoms.

## Dependencies

* numpy
* pytorch

## User Guide
* run main.py

# Ackonwledgement

This repo is effectively a translation of https://github.com/ZhengyaoJiang/GradientInduction.
