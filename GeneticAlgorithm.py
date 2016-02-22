# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 00:30:50 2016

In this GA, the genotype takes the form of a numpy array of L integers, each
representing a character of the 94-long ASCII alphabet.

A population consists of N genotypes.
"""

import numpy as np
from scipy.spatial.distance import hamming as hamm

target = 'Hello, World!'

L = len(target)
N = 250
mutation_rate = 0.01

def fittest_individual(p, fitnesses):
    best = np.argmax(fitnesses)
    fitness = fitnesses[best]
    string = geno_to_str(p[best])
    return fitness, string

def create_initialised_pop(N, L):
    return np.random.random_integers(32, 126, [N, L])

def Hamming_dist(p, target_g):
    """
    Fitness function, for evaluating similarity of two same-length strings.
    Actually 1 - Hamming_distance, s.t. 0 is perfectly unfit, 1 perfectly fit.
    """

    dists = np.empty(N)
    for i, geno in enumerate(p):
        dists[i] = 1 - hamm(geno, target_g)
    return dists

def reproduce(p, fitnesses):
    # Normalise fitnesses
    fitnesses = fitnesses/sum(fitnesses)

    children = np.empty([N, L], dtype=int)
    for child in range(N):
        parents = np.random.choice(range(N), size=2, p=fitnesses)
        cross = np.random.randint(L + 1)
        geno = np.concatenate((p[parents[0], :cross], p[parents[1], cross:]))
        children[child] = geno
    return children

def str_to_geno(s):
    geno = np.empty(len(s), dtype=int)

    for i, character in enumerate(s):
        geno[i] = ord(character)
    return geno

def geno_to_str(geno):
    s = ""

    for i, character in enumerate(geno):
        s += chr(character)
    return s

def mutate(p, rate):
    mutations = np.random.random([N, L])
    mask = np.greater(rate, mutations)
    substitute = create_initialised_pop(N, L)
    np.putmask(p, mask, substitute)
    return p


def main():
    p = create_initialised_pop(N, L)
    generation = 0
    max_fitness = 0

    while(max_fitness < 1):
        fitnesses = Hamming_dist(p, target_geno)
        max_fitness, string = fittest_individual(p, fitnesses)
        print generation, max_fitness, string

        p = reproduce(p, fitnesses)
        p = mutate(p, mutation_rate)

        generation += 1


target_geno = str_to_geno(target)

main()
