"""Minimal CoNLL-style entity-span scorer for BIO tag sequences.

Reads token gold predicted columns (blank lines delimit sentences) as written by
src.sequence. Exact matching of typed spans, including BIO boundary repair for I-X.
This is an independently implemented evaluator, not the original Perl conlleval.

Usage: python -m src.conlleval results/ner_conlleval_input.conll
"""
import sys, json
from .sequence import conlleval_scores


def read(path):
    gold=[];pred=[];g=[];p=[]
    for line in open(path,encoding='utf-8'):
        if not line.strip():
            if g:gold.append(g);pred.append(p);g=[];p=[]
            continue
        parts=line.strip().split()
        if len(parts)<3:raise ValueError('Expected token gold pred: '+line)
        g.append(parts[-2]);p.append(parts[-1])
    if g:gold.append(g);pred.append(p)
    return gold,pred


def evaluate(path):
    gold,pred=read(path)
    return conlleval_scores(gold,pred)[0]

if __name__=='__main__':
    data=evaluate(sys.argv[1])
    for typ,values in data.items():
        print(f'{typ:8s} P={values["precision"]*100:6.2f}% R={values["recall"]*100:6.2f}% F1={values["f1"]*100:6.2f}% '
              f'TP={values["TP"]} FP={values["FP"]} FN={values["FN"]}')