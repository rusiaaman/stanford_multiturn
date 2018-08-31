import json
import keras
import numpy as np
import re
from dateutil import parser
from nltk import word_tokenize
from tqdm import tqdm
from collections import defaultdict

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from difflib import SequenceMatcher

from config import Config

class Data():
    def process(self):
        path = Config.path
        with open(path+'kvret_train_public.json') as f:
            self.dat=json.load(f)

        with open(path+'kvret_dev_public.json') as f:
            self.valid_dat=json.load(f)

        with open(path+'kvret_test_public.json') as f:
            self.test_dat=json.load(f)

        # collection of kb documents
        doc_kb=[]
        for d in tqdm(self.dat):
            try:
                for item in d['scenario']['kb']['items']:
                    [doc_kb.append(t)  for t in item.values()]
            except TypeError:
                continue

        doc_colnames=[]
        for d in tqdm(self.dat):
            try:
                item = d['scenario']['kb']['column_names']
                [doc_colnames.append(t)  for t in item]
            except TypeError:
                continue
            


        # Vocabulary for the databases
        from keras.preprocessing.text import Tokenizer
        self.tokenizer=Tokenizer(filters="")

        self.tokenizer.fit_on_texts(doc_kb)

        self.tokenizer.fit_on_texts(["<SOS>","<EOS>"])

        Config.DB_VOCAB_LEN = len(self.tokenizer.word_index)+1

        self.tokenizer.fit_on_texts(all_texts(self.valid_dat))

        self.tokenizer.fit_on_texts(all_texts(self.dat))

        self.tokenizer.fit_on_texts(all_texts(self.test_dat))


        Config.all_columns_wi={'address': 5,
         'agenda': 0,
         'date': 6,
         'distance': 8,
         'event': 7,
         'friday': 14,
         'location': 2,
         'monday': 10,
         'party': 9,
         'poi': 1,
         'poi_type': 19,
         'room': 15,
         'saturday': 4,
         'sunday': 12,
         'thursday': 16,
         'time': 17,
         'today': 13,
         'traffic_info': 11,
         'tuesday': 3,
         'wednesday': 18}

        self.all_columns = {int(v):w for w,v in Config.all_columns_wi.items()}
        Config.NUM_COL = len(self.all_columns)
        Config.CONV_VOCAB_LEN = len(self.tokenizer.word_index)+1




    # Converting rules based db to desired output first
    def results_to_vector(self,bs_output,pred_intent,operation,kb_data,kb_intent,kb_columns):
        MAX_QUERIES = Config.MAX_QUERIES
        NUM_COL = Config.NUM_COL
        MAX_ENTITY_LENGTH = Config.MAX_ENTITY_LENGTH
        CONV_VOCAB_LEN = Config.CONV_VOCAB_LEN
        NUM_INTENTS = Config.NUM_INTENTS
        OPERATOR_LEN = Config.OPERATOR_LEN
        MAX_DB_RESULTS = Config.MAX_DB_RESULTS
        THRESHOLD=Config.THRESHOLD
        
        assert bs_output.shape == (NUM_COL,MAX_ENTITY_LENGTH,CONV_VOCAB_LEN)
        assert operation.shape == (NUM_COL,OPERATOR_LEN)
        pred_intent = np.argmax(pred_intent) if max(pred_intent)>THRESHOLD else None
        kb_intent = np.argmax(kb_intent)
        output=np.zeros((MAX_DB_RESULTS,NUM_COL,MAX_ENTITY_LENGTH,CONV_VOCAB_LEN))
        if pred_intent is None:
            return output
        q=bs_output
        op = operation
        op_conf =  np.max(op,axis=-1)
        op_classes = np.argmax(op,axis=-1) 
        op_classes = [_q if _q_conf>THRESHOLD else None for _q,_q_conf in zip(op_classes,op_conf)]

        q_ents = np.argmax(q,axis=-1)
        q_confs = np.max(q,axis=-1)
        q_mask = np.array(q_confs>THRESHOLD,dtype='float32')
        q_ents = q_mask*q_ents
        q_words = [" ".join([self.tokenizer.index_word[_q] for _q in __q if _q!=0]) for __q in q_ents]
        # Now that q_words and op_classes are known
        bs={}
        operations = {}
        for j,ent in enumerate(q_words):
            if ent is None or ent=="": continue
            bs[self.all_columns[j]]=ent
            operations[self.all_columns[j]] = op_classes[j]
        result,confidence = kb_results(kb_data,kb_intent,kb_columns,pred_intent,bs,operations)
        result=np.array(result)
        confidence=np.array(confidence)
        result = result[np.argsort(confidence)[-1::-1]]
        confidence = confidence[np.argsort(confidence)[-1::-1]]
        final_result=[kb_data[_i] for _i,(c,r) in enumerate(zip(confidence,result)) if c>=THRESHOLD and r==1]
        confidence=[confidence[_i] for _i,(c,r) in enumerate(zip(confidence,result)) if c>=THRESHOLD and r==1]
        kb_result = np.zeros((MAX_DB_RESULTS,NUM_COL,MAX_ENTITY_LENGTH,CONV_VOCAB_LEN))
        for j,r in enumerate(final_result):
            if j==MAX_DB_RESULTS: break
            for k,v in r.items():
                kb_result[j,Config.all_columns_wi[k]] = to_categorical(pad_sequences(self.tokenizer.texts_to_sequences([v]),
                                                                padding='post',truncating='post',maxlen=MAX_ENTITY_LENGTH)\
                                                                 ,num_classes=CONV_VOCAB_LEN)*confidence[j]
        output = kb_result
        return output

def sim(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def close(a,b):
    return sim(a,b)>=0.50


def num_op(s1,op,s2):
    if isinstance(s1, (int, float)):
        num1 = s1
    else:
        m1=re.search(r'([0-9]+)',s1)
        if m1 is None:
            try:
                num1=parser.parse(s1)
            except ValueError:
                return False,None
        else:
          num1=int(m1.group())
           
    if isinstance(s2,(int,float)):
        num2 = s2
    else:
        m2=re.search(r'([0-9]+)',s2)
        if m2 is None:
            try:
                num2=parser.parse(s2)
            except ValueError:
                return False ,None     
        else:
          num2=int(m2.group())
    try:
        if op=='equal to':
            return num1==num2,num1
        if op=='greater than':
            return num1>num2,num1
        if op=='less than':
            return num1<num2,num1
    except TypeError:
        return False,num1

def kb_results(kb_data,kb_intent,columns,pred_kb_intent,belief_state,operation):
    """This function gets the kb_data, column names and intent for which the kb is received. 
    If intent identified by the bot is nto same as kb_intent no results will be returned.
    
    intents types: {'schedule', 'weather', 'navigate'}
    
    operation should have same keys as belief state with following possible values:
    str, =, >, <, minimum, maximum  indexed from 0 to 5
    Use None for all the values not numerical. If not None, operation would be performend
    """
    #defaultdict(<class 'set'>, {'navigate': {'poi', 'distance', 'poi_type', 'traffic_info', 'address'},, 'weather': {'thursday', 'sunday', 'today', 'friday', 'wednesday', 'tuesday', 'saturday', 'location', 'monday'}})
    if pred_kb_intent!=kb_intent:
        return [],[]
    if kb_data is None:
        return [],[]
    results = [None for _ in range(len(kb_data))]
    confidence = np.ones(len(kb_data))
    # column names possiblity: {'room', 'party', 'event', 'agenda', 'date', 'time'}  
    # Note that date and time are immutable and non-comparable in current dialog, so they are treated as strings
    col_types = defaultdict(lambda: 'str')
    if any(k not in columns for k in belief_state.keys()):
        return [],[]
    for k in belief_state.keys():
        if belief_state.get(k) is None or operation.get(k) is None:
            print(k)
            return [],[]
        min_idx = None
        min_val = float('Inf')
        max_idx = None
        max_val = -float('Inf')
        for i,items in enumerate(kb_data):
            if results[i] == 0: continue
            if col_types[k]=='str':
                if items.get(k) is None:
                    results[i]=0
                    continue
                results[i]=0
                if operation[k]==0 and close(belief_state[k],items[k]):
                    # Doing string comparison
                    results[i]=1
                    confidence[i] = confidence[i]*sim(belief_state.get(k),items.get(k))
                elif operation[k]==1:
                    #Doing equal comparison extracting the first number
                    if num_op(belief_state[k],'equal to',items[k])[0]:
                        results[i]=1
                elif operation[k]==2:
                    #Doing greater than comparison extracting the first number
                    if num_op(belief_state[k],'less than',items[k])[0]:
                        results[i]=1
                elif operation[k]==3:
                    #Doing less than comparison extracting the first number
                    if num_op(belief_state[k],'greater than',items[k])[0]:
                        results[i]=1
                elif operation[k]==4:
                    #Doing mimum comparison extracting the first number
                    res,val = num_op(items[k],'less than',(min_val))
                    if res:
                        results[i]=1
                        if min_idx is not None:
                            results[min_idx] = 0
                        min_val = val
                        min_idx = i
                elif operation[k]==5:
                    #Doing maximum comparison extracting the first number
                    res,val = num_op(items[k],'greater than',(max_val))
                    if res:
                        results[i]=1
                        if max_idx is not None:
                            results[max_idx] = 0
                        max_val = val
                        max_idx = i

           
    return np.array(results),np.array(confidence)


def tokenize(t):
    return word_tokenize(t)



def all_dict(d):
    texts=[]
    texts.append(" ".join(list(d.keys())))
    for v in d.values():
        if isinstance(v,str):
            texts.append(v) 
        elif isinstance(v,list):
            texts.append(" ".join(all_texts(v)))
        elif isinstance(v,dict):
            texts.append(" ".join(all_dict(v)))
        else:
            try:
                texts.append(str(v))
            except:
                raise Exception(f'type of v is {type(v)}')
    return texts

def all_texts(data):
    texts = []
    for d in data:
        if isinstance(d,dict):
            texts.append(" ".join(all_dict(d)))
        elif isinstance(d,list):
            texts.append(" ".join(all_texts(d)))
        elif isinstance(d,str):
            texts.append(d)
        else:
            try:
                texts.append(str(d))
            except:
                raise Exception(f'type of d is {type(d)}')
    return texts


