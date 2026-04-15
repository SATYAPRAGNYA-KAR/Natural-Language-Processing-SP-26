import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import re

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # raise NotImplementedError
    QWERTY_NEIGHBORS={
        'q':['w','a'], 'w':['q','e','a','s'], 'e':['w','r','s','d'], 'r':['e','t','d','f'],
        't':['r','y','f','g'], 'y':['t','u','g','h'], 'u':['y','i','h','j'], 'i':['u','o','j','k'],
        'o':['i','p','k','l'], 'p':['o','l'], 'a':['q','w','s','z'], 's':['a','d','w','e','z','x'],
        'd':['s','f','e','r','x','c'], 'f':['d','g','r','t','c','v'], 'g':['f','h','t','y','v','b'],
        'h':['g','j','y','u','b','n'], 'j':['h','k','u','i','n','m'], 'k':['j','l','i','o','m'],
        'l':['k','o','p'], 'z':['a','s','x'], 'x':['z','s','d','c'], 'c':['x','d','f','v'],
        'v':['c','f','g','b'], 'b':['v','g','h','n'], 'n':['b','h','j','m'], 'm':['n','j','k']
    }
    INFORMAL_MAP={
        'you':'u', 'are':'r', 'because':'bcz', 'great':'gr8', 'before':'b4', 'later':'l8r',
        'please':'plz', 'thanks':'thx', 'people':'ppl', 'really':'rly', 'with':'w/',
        'without':'w/o', 'your':'ur', 'though':'tho', 'through':'thru', 'okay':'ok',
        'about':'abt', 'probably':'prob', 'definitely':'def', 'actually':'tbh',
        'something':'smth', 'everyone':'evry1', 'anyone':'ne1', 'nothing':'nth',
        'everything':'evrythng', 'good':'gud', 'love':'luv', 'hate':'h8', 'see':'c',
        'too':'2', 'for':'4', 'be':'b', 'and':'&', 'at':'@'
    }
    CONTRACTION_EXPAND={
        "don't":"do not", "doesn't":"does not", "didn't":"did not", "won't":"will not",
        "can't":"cannot", "couldn't":"could not", "shouldn't":"should not",
        "wouldn't":"would not", "it's":"it is", "that's":"that is", "i'm":"i am",
        "i've":"i have", "i'll":"i will", "i'd":"i would", "they're":"they are",
        "we're":"we are", "you're":"you are", "isn't":"is not", "aren't":"are not",
        "wasn't":"was not", "weren't":"were not", "hasn't":"has not", "haven't":"have not"
    }
    CONTRACTION_COLLAPSE={
        "do not":"don't", "does not":"doesn't", "did not":"didn't", "will not":"won't",
        "can not":"can't", "could not":"couldn't", "should not":"shouldn't",
        "would not":"wouldn't", "it is":"it's", "that is":"that's", "i am":"i'm",
        "i have":"i've", "i will":"i'll", "i would":"i'd", "they are":"they're",
        "we are":"we're", "you are":"you're", "is not":"isn't", "are not":"aren't",
        "was not":"wasn't", "were not":"weren't", "has not":"hasn't", "have not":"haven't"
    }
    FILLERS=['like', 'basically', 'honestly', 'kind of', 'sort of', 'you know', 'i mean',
             'literally', 'actually', 'pretty much', 'low key', 'arguably', 'supposedly']
    text=example["text"]
    # Step 1: Contraction Expansion/Collapsing inconsistently
    if random.random()<0.5:
        for contraction,expansion in CONTRACTION_EXPAND.items():
            if contraction in text.lower():
                text=re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
    else:
        for expansion,contraction in CONTRACTION_COLLAPSE.items():
            if expansion in text.lower():
                text=re.sub(re.escape(expansion), contraction, text, flags=re.IGNORECASE)

    # Step 2: Clause Shuffling on Comma-separated Segments per Sentence
    sentences=re.split(r'(?<=[.!?])\s+', text)
    shuffled_sentences=[]
    for sent in sentences:
        clauses=[c.strip() for c in sent.split(',') if c.strip()]
        if len(clauses)>2 and random.random()<0.5:
            middle=clauses[1:-1]
            random.shuffle(middle)
            clauses=[clauses[0]]+middle+[clauses[-1]]
        shuffled_sentences.append(', '.join(clauses))
    text=' '.join(shuffled_sentences)

    # Step 3: Punctuation Disruption- Stripping and Reinserting Randomly
    if random.random()<0.4:
        text=re.sub(r'[,;]', '', text)
        words_for_punct=text.split()
        num_insertions=max(1,len(words_for_punct)//15)
        for _ in range(num_insertions):
            pos=random.randint(1,max(1, len(words_for_punct)-1))
            words_for_punct[pos]=words_for_punct[pos]+','
        text=' '.join(words_for_punct)

    # Step 4: Word-level Transformations with Filler Insertion
    words=word_tokenize(text)
    transformed=[]
    i=0
    while i<len(words):
        word=words[i]
        r=random.random()
        lower=word.lower()
        # Filler Insertion
        if r<0.06 and word.isalpha():
            filler=random.choice(FILLERS)
            transformed.append(filler)
            transformed.append(word)
        # Informal Map
        elif r<0.13 and lower in INFORMAL_MAP:
            transformed.append(INFORMAL_MAP[lower])
        # Synonym Replacement
        elif r<0.24 and word.isalpha():
            syns=wordnet.synsets(word)
            lemmas=[l.name().replace('_', ' ') for s in syns for l in s.lemmas() if l.name().lower()!=lower]
            transformed.append(lemmas[0] if lemmas else word)
        # Typo or Transposition
        elif r<0.34 and word.isalpha() and len(word)>2:
            transform_type=random.random()
            if transform_type<0.5:
                idx=random.randint(0, len(word)-1)
                ch=word[idx].lower()
                if ch in QWERTY_NEIGHBORS:
                    neighbor=random.choice(QWERTY_NEIGHBORS[ch])
                    word=word[:idx]+neighbor+word[idx+1:]
            else:
                idx=random.randint(0, len(word)-2)
                word=word[:idx]+word[idx+1]+word[idx]+word[idx+2:]
            transformed.append(word)
        else:
            transformed.append(word)
        i+=1
    example["text"]=TreebankWordDetokenizer().detokenize(transformed)
    return example

    ##### YOUR CODE ENDS HERE ######

    return example
