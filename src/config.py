class Config(object):
    path='../kvret_dataset_public/'
    MAX_QUERIES = 1
    THRESHOLD = 0.5
    MAX_DB_RESULTS = 5
    MAX_ENTITY_LENGTH = 10
    OPERATOR_LEN = 6
    NUM_INTENTS = 3
    EMBEDDING_SIZE=50
    BATCH_SIZE=8
    EPOCHS=30
    LOAD_MODEL=True
    CHECKPOINT='./db_model.h5'
    def __init__():
    	pass
