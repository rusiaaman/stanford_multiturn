from keras.layers import Input,Dense,LSTM,Embedding,TimeDistributed, RepeatVector, Concatenate,Reshape
from keras.layers import Lambda
from keras.models import Model

import keras.backend as K
import tensorflow as tf

# model for db

def get_model():
    MAX_QUERIES = Config.MAX_QUERIES
    NUM_COL = Config.NUM_COL
    MAX_ENTITY_LENGTH = Config.MAX_ENTITY_LENGTH
    CONV_VOCAB_LEN = Config.CONV_VOCAB_LEN
    NUM_INTENTS = Config.NUM_INTENTS
    OPERATOR_LEN = Config.OPERATOR_LEN
    MAX_DB_RESULTS = Config.MAX_DB_RESULTS
    
    bs_input = Input(shape=(MAX_QUERIES,NUM_COL,MAX_ENTITY_LENGTH,CONV_VOCAB_LEN))
    intent_input = Input(shape=(MAX_QUERIES,NUM_INTENTS,))
    operation_input = Input(shape=(MAX_QUERIES,NUM_COL,OPERATOR_LEN))

    bs_proc = TimeDistributed(TimeDistributed(TimeDistributed(Dense(10,activation='sigmoid'))))(bs_input)
    LSTM_bs_emb = TimeDistributed(TimeDistributed(LSTM(50,return_sequences=False,return_state=False)))(bs_proc)
    rep_intent_input = TimeDistributed(RepeatVector(NUM_COL))(intent_input)
    print(LSTM_bs_emb.shape)
    all_steps = Concatenate(axis=-1)([LSTM_bs_emb,operation_input,rep_intent_input])
    all_steps = Lambda(lambda x: tf.reshape(x,shape=(-1,MAX_QUERIES,NUM_COL*(50+OPERATOR_LEN+NUM_INTENTS))))(all_steps)
    encoder_lstm = Dense(50,activation='relu')(all_steps)
    encoder_lstm = TimeDistributed(RepeatVector(MAX_DB_RESULTS))(encoder_lstm)

    decoder_lstm1 = TimeDistributed(LSTM(50,return_sequences=True))(encoder_lstm)

    decoder_lstm1 = Dense(NUM_COL*50,activation='relu')(decoder_lstm1)
    decoder_lstm1 = Lambda(lambda x: tf.reshape(x,shape=(-1,MAX_QUERIES,MAX_DB_RESULTS,NUM_COL,50)))(decoder_lstm1)


    decoder_lstm2 = TimeDistributed(Lambda(lambda x: K.tile(K.expand_dims(x,axis=-2),[1,1,1,MAX_ENTITY_LENGTH,1])))(decoder_lstm1)
    decoder_lstm3 = TimeDistributed(TimeDistributed(TimeDistributed(LSTM(10,return_sequences=True))))(decoder_lstm2)

    out = TimeDistributed(TimeDistributed(TimeDistributed(TimeDistributed(Dense(CONV_VOCAB_LEN,activation='softmax')))))(decoder_lstm3)
    db_model = Model(inputs=[bs_input,intent_input,operation_input],outputs=[out])

    db_model.summary()

    db_model.compile(optimizer='adam',loss='categorical_crossentropy')
    return db_model