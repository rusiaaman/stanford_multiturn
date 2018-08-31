import psutil
import keras



def memory_exhausted():
    if memory_percent_available()<=10:
        print("Memory Exhausted")
        exit()
class memCall(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        memory_exhausted()
        
def memory_percent_available():
    return psutil.virtual_memory().available/psutil.virtual_memory().total*100


def train(db_model,data,generator,epochs=10,batch_size=1):
	checkpoint=keras.callbacks.ModelCheckpoint('./tmp.h5',save_best_only=True)

	db_model.fit_generator(generator(batch_size,data.dat),\
		validation_data=generator(batch_size,data.valid_dat),\
		steps_per_epoch=100,epochs=epochs,validation_steps=50,\
		callbacks=[checkpoint])
	return db_model
