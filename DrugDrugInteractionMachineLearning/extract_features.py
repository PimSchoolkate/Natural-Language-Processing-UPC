#! /usr/bin/python3

import sys
from os import listdir

from xml.dom.minidom import parse

from util.deptree import *
from nltk import ngrams
import patterns


## ------------------- 
## -- Convert a pair of drugs and their context in a feature vector

def extract_features(tree, entities, e1, e2):
    feats = set()

    # get head token for each gold entity
    tkE1 = tree.get_fragment_head(entities[e1]['start'], entities[e1]['end'])
    tkE2 = tree.get_fragment_head(entities[e2]['start'], entities[e2]['end'])

    if tkE1 is not None and tkE2 is not None:

        # features for tokens in between E1 and E2
        for tk in range(tkE1 + 1, tkE2):
            if not tree.is_stopword(tk):
                word = tree.get_word(tk)
                lemma = tree.get_lemma(tk).lower()
                tag = tree.get_tag(tk)
                feats.add(f"lib={lemma}")
                feats.add(f"wib={word}")
                feats.add(f"lpib={lemma}_{tag}")
                feats.add(f"pib={tag}")

                # feature indicating the presence of an entity in between E1 and E2
                if tree.is_entity(tk, entities):
                    feats.add("eib=1")

        if tkE2 - tkE1 == 2:
            feats.add(f"oneib={tree.get_tag(tkE1+1)}")

        # features about paths in the tree
        lcs = tree.get_LCS(tkE1, tkE2)

        ## Start Added features
        feats.add(f"lcs_stop={tree.is_stopword(lcs)}")
        feats.add(f"lcs_syntax={tree.get_rel(lcs)}")
        feats.add(f"lcs_tag={tree.get_tag(lcs)}")
        feats.add(f"lcs_lemma={tree.get_lemma(lcs)}")
        ## End Added features

        path1 = tree.get_up_path(tkE1, lcs)
        path1 = "<".join([tree.get_lemma(x) + "_" + tree.get_rel(x) for x in path1])
        feats.add("path1=" + path1)

        path2 = tree.get_down_path(lcs, tkE2)
        path2 = ">".join([tree.get_lemma(x) + "_" + tree.get_rel(x) for x in path2])
        feats.add("path2=" + path2)

        path = path1 + "<" + tree.get_lemma(lcs) + "_" + tree.get_rel(lcs) + ">" + path2
        feats.add("path=" + path)

        ## Start Added features
        path1 = tree.get_up_path(tkE1, lcs)
        # path1_words = "<".join([tree.get_lemma(x) for x in path1])
        path1_syntactic = "<".join([tree.get_tag(x) for x in path1])
        # feats.add(f"path1_words={path1_words}")
        feats.add(f"path1_syntactic={path1_syntactic}")
        ## End Added features

        ## Start Added features
        path2 = tree.get_down_path(lcs, tkE2)
        # path2_words = ">".join([tree.get_lemma(x) for x in path2])
        path2_syntactic = ">".join([tree.get_tag(x) for x in path2])
        # feats.add(f"path2_words={path2_words}")
        feats.add(f"path2_syntactic={path2_syntactic}")
        ## End Added features

        ## Start Added features
        # path_words = path1_words + "<" + tree.get_lemma(lcs) + ">" + path2_words
        path_syntactic = path1_syntactic + "<" + tree.get_lemma(lcs) + ">" + path2_syntactic
        # feats.add(f"path_words={path_words}")
        feats.add(f"path_syntactic={path_syntactic}")
        ## End Added features

        ## Start Added Features
        feats.add(f"distance={tkE2 - tkE1}")

        general_clue_lemmas = ['ingest','correspond','react','present','regulate','represent','counteract','antagonize',
                               'stimulate','describe','protect','augment','anaesthetise','deplete','blunt','switch',
                               'share','dictate','select','bear','exaggerate','accord','shorten','wait','threaten',
                               'necessitate','warn','anticipate','expose','exert','continue','withdraw','term',
                               'alkalinize','fold','halogenate','sensitize','vasoconstrict','lengthen','recognize',
                               'progress','depolarise','precipitate','propose','watch','isrecommend','postmarket',
                               'last','stand']

        advise_clue_lemmas = ['present','advise','undertake','switch','seem','dictate','select','bear','hear','accord',
                              'wait','threaten','warn','deplete','exert','continue','exceed','conflict','outweigh',
                              'anaesthetise','precipitate','watch','function','beadminister','isrecommend','interrupt']
        effect_clue_lemmas = ['present','regulate','counteract','antagonize','stimulate','protect','augment',
                              'cecectomize','blunt','exaggerate','necessitate','expose','term','halogenate','sensitize',
                              'vasoconstrict','lengthen','shorten','describe','recognize','progress','depolarise',
                              'weaken','propose','postmarket','stand']
        interaction_clue_lemmas = ['anaesthetise','exist','pose','depend','threaten','nondepolarize','kill']
        mechanism_clue_lemmas = ['ingest','phosphorylate','correspond','react', 'match', 'present', 'share', 'promote',
                                 'empty','evidence','anticipate','alkalinize','market','fold','hydroxylate','fall',
                                 'propose','threaten','average','represent','desire','leave','denote','last']

        for node in tree.get_nodes():
            if tree.get_lemma(node) in general_clue_lemmas:
                feats.add("gen_clue=1")
                feats.add(f"gen_loc={find_clue_position(tkE1, tkE1, node)}")
            if tree.get_lemma(node) in advise_clue_lemmas:
                feats.add("adv_clue=1")
                feats.add(f"adv_loc={find_clue_position(tkE1, tkE1, node)}")
            if tree.get_lemma(node) in effect_clue_lemmas:
                feats.add("eff_clue=1")
                feats.add(f"eff_loc={find_clue_position(tkE1, tkE1, node)}")
            if tree.get_lemma(node) in interaction_clue_lemmas:
                feats.add("int_clue=1")
                feats.add(f"int_loc={find_clue_position(tkE1, tkE1, node)}")
            if tree.get_lemma(node) in mechanism_clue_lemmas:
                feats.add("mec_clue=1")
                feats.add(f"mec_loc={find_clue_position(tkE1, tkE1, node)}")


        ## Features taken from Louis van Langendonck and Enric Reverter
        verb_list = ['reduce', 'induce', 'recommend', 'administer', 'produce', 'use', 'enhance', 'give', 'result',
                     'receive', 'potentiate', 'coadminister'
            , 'decrease', 'cause', 'inhibit', 'report', 'take', 'contain']
        pattern_list = [('JJ', 'NN', 'IN'), ('DT', 'NN', 'IN'), ('DT', 'JJ', 'NN'), ('IN', 'DT', 'NN'),
                        ('NN', 'CC', 'NN'), ('IN', 'JJ', 'NN'),
                        ('JJ', 'NNS', ','), ('NN', 'NN', 'NNS'), ('-LRB-', 'NN', 'NN'), ('NN', 'IN', 'DT'),
                        ('IN', 'NN', 'IN')]
        pos_pattern = []
        for tk in (tree.get_nodes()):
            pos_pattern.append(tree.get_tag(tk))
            if 'VB' in tree.get_tag(tk):
                feats.add("verb=" + tree.get_lemma(tk))
                if tree.get_lemma(tk) in verb_list:
                    feats.add("typical_verb=True")
        threegrams = ngrams(pos_pattern, 3)
        for grams in threegrams:
            if grams in pattern_list:
                feats.add("typical_threegram=True")

    return feats


def find_clue_position(e1, e2, n):
    d1 = n - e1
    d2 = n - e2
    if d1 < 0:
        return "before"
    if d1 > 0 and d2 < 0:
        return "between"
    if d2 > 0:
        return "after"


## --------- MAIN PROGRAM -----------
## --
## -- Usage:  extract_features targetdir
## --
## -- Extracts feature vectors for DD interaction pairs from all XML files in target-dir
## --

# directory with files to process
# datadir = sys.argv[1]

datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir):

    # parse XML file, obtaining a DOM tree
    tree = parse(datadir + "/" + f)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences:
        sid = s.attributes["id"].value  # get sentence id
        stext = s.attributes["text"].value  # get sentence text
        # load sentence entities
        entities = {}
        ents = s.getElementsByTagName("entity")
        for e in ents:
            id = e.attributes["id"].value
            offs = e.attributes["charOffset"].value.split("-")
            entities[id] = {'start': int(offs[0]), 'end': int(offs[-1])}

        # there are no entity pairs, skip sentence
        if len(entities) <= 1: continue

        # analyze sentence
        analysis = deptree(stext)

        # for each pair in the sentence, decide whether it is DDI and its type
        pairs = s.getElementsByTagName("pair")
        for p in pairs:
            # ground truth
            ddi = p.attributes["ddi"].value
            if (ddi == "true"):
                dditype = p.attributes["type"].value
            else:
                dditype = "null"
            # target entities
            id_e1 = p.attributes["e1"].value
            id_e2 = p.attributes["e2"].value
            # feature extraction

            feats = extract_features(analysis, entities, id_e1, id_e2)
            # resulting vector with sid, id_e1, and id_e2
            print(sid, id_e1, id_e2, dditype, "\t".join(feats), sep="\t")
