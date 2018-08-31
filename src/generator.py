import numpy as np

from config import Config

def softmax(w, t = 1.0):
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist


class Generator(object):
    def __init__(self,all_columns_wi,data_pool):
        self.nav_cols=['distance','traffic_info','poi_type','address','poi']
        self.intents = ['schedule', 'weather', 'navigate']
        self.all_columns_wi=all_columns_wi
        self.data_pool=data_pool

    def input_generator(self,batch_size,data):
        MAX_QUERIES = Config.MAX_QUERIES
        NUM_COL = Config.NUM_COL
        MAX_ENTITY_LENGTH = Config.MAX_ENTITY_LENGTH
        CONV_VOCAB_LEN = Config.CONV_VOCAB_LEN
        NUM_INTENTS = Config.NUM_INTENTS
        OPERATOR_LEN = Config.OPERATOR_LEN
        MAX_DB_RESULTS = Config.MAX_DB_RESULTS
        batch_data1=[]
        batch_data2=[]
        batch_data3=[]
        target=[]
        random_dat = [data[i] for i in np.random.permutation(len(data))]
        ij=0
        while True:
            ij+=1
            for d in random_dat:
                kb_intent = d['scenario']['task']['intent']
                if kb_intent!='navigate': continue
                kb_col_names = d['scenario']['kb']['column_names']
                kb_data = d['scenario']['kb']['items']
                true_vec_intent = np.zeros(NUM_INTENTS)
                true_vec_intent[self.intents.index(kb_intent)]=1.0
                pred_intent = np.array([0,0,1])#softmax(np.random.normal(size=NUM_INTENTS,loc=100,scale=5))
                bs_input = np.zeros((NUM_COL,MAX_ENTITY_LENGTH,CONV_VOCAB_LEN))
                operation = np.zeros((NUM_COL,OPERATOR_LEN))
                num_cols_to_have = np.random.randint(NUM_COL)+1
                num_ents_to_have = [np.random.randint(MAX_ENTITY_LENGTH)+1 for _ in range(num_cols_to_have)]
                for ii in range(num_cols_to_have):
                    col_idx = self.all_columns_wi[self.nav_cols[np.random.randint(5)]]
                    for j in range(num_ents_to_have[ii]):
                        ix=(col_idx,j)
                        bs_input[ix] = softmax(np.random.normal(size=CONV_VOCAB_LEN,loc=100,scale=5))
                    operation[col_idx] = softmax(np.random.normal(size=OPERATOR_LEN,loc=100,scale=5))

                batch_data1.append([bs_input])
                batch_data2.append([operation])
                batch_data3.append([pred_intent])
                target.append([self.data_pool.results_to_vector(bs_input,pred_intent,operation,kb_data,true_vec_intent,kb_col_names)])
                
                if len(batch_data1)==batch_size:
                    yield [np.array(batch_data1),np.array(batch_data3),np.array(batch_data2)],np.array(target)
                    batch_data1=[]
                    batch_data2=[]
                    batch_data3=[]
                    target=[]