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
        'q':['w','a'],'w':['q','e','a','s'],'e':['w','r','s','d'],'r':['e','t','d','f'],
        't':['r','y','f','g'],'y':['t','u','g','h'],'u':['y','i','h','j'],'i':['u','o','j','k'],
        'o':['i','p','k','l'],'p':['o','l'],'a':['q','w','s','z'],'s':['a','d','w','e','z','x'],
        'd':['s','f','e','r','x','c'],'f':['d','g','r','t','c','v'],'g':['f','h','t','y','v','b'],
        'h':['g','j','y','u','b','n'],'j':['h','k','u','i','n','m'],'k':['j','l','i','o','m'],
        'l':['k','o','p'],'z':['a','s','x'],'x':['z','s','d','c'],'c':['x','d','f','v'],
        'v':['c','f','g','b'],'b':['v','g','h','n'],'n':['b','h','j','m'],'m':['n','j','k']
    }
    INFORMAL_MAP={
        'you':'u','are':'r','because':'bcz','great':'gr8','before':'b4','later':'l8r',
        'please':'plz','thanks':'thx','people':'ppl','really':'rly','with':'w/',
        'without':'w/o','your':'ur','though':'tho','through':'thru','okay':'ok',
        'about':'abt','probably':'prob','definitely':'def','actually':'tbh',
        'something':'smth','everyone':'evry1','anyone':'ne1','nothing':'nth',
        'everything':'evrythng','good':'gud','love':'luv','hate':'h8','see':'c',
        'too':'2','to':'2','for':'4','be':'b','and':'&','at':'@'
    }
    words=word_tokenize(example["text"])
    transformed=[]
    for word in words:
        r=random.random()
        lower=word.lower()
        if r<0.07 and lower in INFORMAL_MAP:
            transformed.append(INFORMAL_MAP[lower])
        elif r<0.18 and word.isalpha():
            syns=wordnet.synsets(word)
            lemmas=[l.name().replace('_',' ') for s in syns for l in s.lemmas() if l.name().lower()!=lower]
            transformed.append(lemmas[0] if lemmas else word)
        elif r<0.28 and word.isalpha() and len(word)>2:
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
    example["text"]=TreebankWordDetokenizer().detokenize(transformed)

    ##### YOUR CODE ENDS HERE ######

    return example
