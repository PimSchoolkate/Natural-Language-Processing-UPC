#! /usr/bin/python3

from lib2to3.pgen2 import token
import sys
import re
from os import listdir
import string

from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize

import syllables  
   
## --------- tokenize sentence ----------- 
## -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    ## word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(txt):
        ## keep track of the position where each token should appear, and
        ## store that information with the token
        #txt = txt.translate(str.maketrans("","", string.punctuation))

        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)

    ## tks is a list of triples (word,start,end)
    return tks


## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans) :
   (form,start,end) = token
   for (spanS,spanE,spanT) in spans :
      if start==spanS and end<=spanE : return "B-"+spanT
      elif start>=spanS and end<=spanE : return "I-"+spanT

   return "O"
 
## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence

def extract_features(tokens) :

   # for each token, generate list of features and add it to the result
   result = []
   for k in range(0,len(tokens)):
      tokenFeatures = []
      t = tokens[k][0]

      tokenFeatures.append("form="+t)
      tokenFeatures.append("suf3="+t[-3:])
      

      # v1.1 : length of word
      tokenFeatures.append(f"length={len(t)}")
      # v.1.1.1.2: checking if the whole word is uppercase
      tokenFeatures.append(f"isUpper={t.isupper()}")
      # v.1.1.1.2.1: 
      tokenFeatures.append(f"isLower={t.islower()}")

      # v.1.1.1.2.1.2:
      tokenFeatures.append(f"isfirstLetterUppercase={t[0].isupper()}")

      # v.1.1.1.2.1.1: checking for special characters
      dash = any(c in "-" for c in t)
      #slash = any(c in "/" for c in t)
      tokenFeatures.append(f"hasDash={dash}")
      #tokenFeatures.append(f"hasSlash={slash}")

      # v.1.1.2.1.1.1.
      tokenFeatures.append(f"syllables={syllables.estimate(t)}")

      #v.1.1.1.2.1.1.1.1: ending with s
      isplural = t.endswith('s')
      tokenFeatures.append(f"isPlural={isplural}")

      """
      # v.1.1.2.1.1.1.1.2 Word shape
      # encode word
      w = ""
      for c in t:
         if c.isupper() and 'X' not in w:
            w = w +"X"
         elif c.islower() and "x" not in w:
            w = w+"x"
         elif c.isnumeric() and "o" not in w:
            w = w+"o"
         elif "O" not in w:
            w= w +"O"
      tokenFeatures.append("wordShape="+w)
      """
      # Is drug found ?
      # v.1.1.2.1.1.1.1.3 
      entityFound = False
      file = open("DrugBank.txt", encoding="utf8")
      for line in file:
         line = line.strip().split('|')
         if t == line[0]:
            entityFound = True
            entitytype = line[1]
      
      if entityFound == True:
         tokenFeatures.append(f"entityFound={entityFound}")
         tokenFeatures.append(f"entitytype={entitytype}")


      if k>0 :
         tPrev = tokens[k-1][0]
         # v1.1.1: adding length of previous word
         tokenFeatures.append(f"lengthPrev={len(tPrev)}")

         
         #v1 adding suffix 2 words prior
         if k>1: 
            tPrev2 = tokens[k-2][0]
            tokenFeatures.append("formPrev2="+tPrev2)
            tokenFeatures.append("suf3Prev2="+tPrev2[-3:])
         tokenFeatures.append("formPrev="+tPrev)
         tokenFeatures.append("suf3Prev="+tPrev[-3:])
         
      else :
         tokenFeatures.append("BoS")

      if k<len(tokens)-1 :
         tNext = tokens[k+1][0]
         # v1.1.1. adding length of next word
         tokenFeatures.append(f"lengthNext={len(tNext)}")

         #v1 adding suffix 2 words posterior
         if k<len(tokens)-2:
            tNext2 = tokens[k+2][0]
            tokenFeatures.append("formPrev2="+tNext2)
            tokenFeatures.append("suf3Prev2="+tNext2[-3:])
           
         tokenFeatures.append("formNext="+tNext)
         tokenFeatures.append("suf3Next="+tNext[-3:])
        

         

      else:
         tokenFeatures.append("EoS")
    
      result.append(tokenFeatures)
    
   return result


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --


# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir) :
   
   # parse XML file, obtaining a DOM tree
   tree = parse(datadir+"/"+f)
   
   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for s in sentences :
      sid = s.attributes["id"].value   # get sentence id
      spans = []
      stext = s.attributes["text"].value   # get sentence text
      entities = s.getElementsByTagName("entity")
      for e in entities :
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ =  e.attributes["type"].value
         spans.append((int(start),int(end),typ))
         

      # convert the sentence to a list of tokens
      tokens = tokenize(stext)
      # extract sentence features
      features = extract_features(tokens)

      # print features in format expected by crfsuite trainer
      for i in range (0,len(tokens)) :
         # see if the token is part of an entity
         tag = get_tag(tokens[i], spans) 
         print (sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

      # blank line to separate sentences
      print()
