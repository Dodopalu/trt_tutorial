import segmentation_models as sm
import tensorflow as tf
from keras2onnx import convert_keras
from engine import *

# Disable eager execution in tensorflow 2 is required.
tf.compat.v1.disable_eager_execution()
# Set learning phase to Test.
tf.compat.v1.keras.backend.set_learning_phase(0)
 
hdf5_path = 'unet.hdf5'
onnx_path = 'unet.onnx'
engine_name = 'unet.plan'
batch_size = 1
CHANNEL = 3
HEIGHT = 224
WIDTH = 224
 
 
model = sm.Unet()
model._layers[0].batch_input_shape = (None, 224,224,3)
model = tf.keras.models.clone_model(model)
model.save(hdf5_path)

onx = convert_keras(model, onnx_path)
with open(onnx_path, "wb") as f:
    f.write(onx.SerializeToString())
 

shape = [batch_size, HEIGHT, WIDTH, CHANNEL]
engine = build_engine(onnx_path, shape=shape)
save_engine(engine, engine_name)
