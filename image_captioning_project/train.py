import SceneDesc
import sys

def train(epoch):
    sd = SceneDesc.scenedesc()
    batch_size = 512
    model = sd.create_model()
    model.fit_generator(sd.data_process(batch_size=batch_size), steps_per_epoch=sd.no_samples/batch_size, epochs=epoch, verbose=2, callbacks=None)
    model.save('RNN_Train_weights/Model.h5', overwrite=True)
    model.save_weights('RNN_Train_weights/Weights.h5',overwrite=True)
 
if __name__=="__main__":
    train(int(sys.argv[1]))