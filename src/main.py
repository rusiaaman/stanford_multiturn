import time

from preprocess import Data
from model import get_model
from trainer import train
from config import Config
from generator import Generator

if __name__=='__main__':
    data = Data()
    data.process()
    model=get_model()
    Gen = Generator(Config.all_columns_wi,data)
    train(model,data,epochs=Config.EPOCHS,batch_size=Config.BATCH_SIZE,generator=Gen.input_generator)
    model.save('model'+str(time.time())+'.h5')
    print("FINISHED TRAINING")
