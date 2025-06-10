import numpy as np
import tensorrt as trt
import os
import time
import h5py
from onnx import ModelProto
# import torch
# from torch import nn
# from model import BraggNN
# from dataset import BraggNNDataset
# from matplotlib import pyplot as plt
# plt.style.use('classic')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pycuda.driver as cuda
import pycuda.autoinit 

def load_engine(trt_runtime, plan_path):
    with open(plan_path, 'rb') as f:
        engine_data = f.read()
    print("Loading TensorRT engine from: ", plan_path)
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

def allocate_buffers(engine, batch_size, data_type):

   """
   This is the function to allocate buffers for input and output in the device
   Args:
      path : The path to the TensorRT engine. 
      batch_size : The batch size for execution time.
      data_type: The type of the data for input and output, for example trt.float32. 
   
   Output:
      h_input_1: Input in the host.
      d_input_1: Input in the device. 
      h_output_1: Output in the host. 
      d_output_1: Output in the device. 
      stream: CUDA stream.

   """

   # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
   h_input_1 = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(data_type))
   h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(data_type))
    # Allocate device memory for inputs and outputs.
   d_input_1 = cuda.mem_alloc(h_input_1.nbytes)

   d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
   stream = cuda.Stream()
   return h_input_1, d_input_1, h_output, d_output, stream 

def load_images_to_buffer(pics, pagelocked_buffer):
   preprocessed = np.asarray(pics).ravel()
   np.copyto(pagelocked_buffer, preprocessed) 

def do_inference(engine, pics_1, h_input_1, d_input_1, h_output, d_output, stream, batch_size, width):
   """
   This is the function to run the inference
   Args:
      engine : Path to the TensorRT engine 
      pics_1 : Input images to the model.  
      h_input_1: Input in the host         
      d_input_1: Input in the device 
      h_output_1: Output in the host 
      d_output_1: Output in the device 
      stream: CUDA stream
      batch_size : Batch size for execution time
      height: Height of the output image
      width: Width of the output image
   
   Output:
      The list of output images

   """

   load_images_to_buffer(pics_1, h_input_1)

   with engine.create_execution_context() as context:
       # Transfer input data to the GPU.
       cuda.memcpy_htod_async(d_input_1, h_input_1, stream)

       # Run inference.

       # context.profiler = trt.Profiler()
       context.execute(batch_size, bindings=[int(d_input_1), int(d_output)])

       # Transfer predictions back from the GPU.
       cuda.memcpy_dtoh_async(h_output, d_output, stream)
       # Synchronize the stream
       stream.synchronize()
       # Return the host output.
       out = h_output.reshape((batch_size, 1, width))
       # print(type(out))
       return out

print("Loading TensorRT engine...")

def create_engine(TRT_LOGGER, onnx_path, shape):

    batch_size = shape[0]

    with (
        trt.Builder(TRT_LOGGER) as builder, 
        builder.create_network(1) as network, 
        builder.create_builder_config() as config, 
        trt.OnnxParser(network, TRT_LOGGER) as parser
    ):
        builder.max_batch_size = batch_size
        config.set_flag(trt.BuilderFlag.FP16)
        config.max_workspace_size = (1 << 33)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = [batch_size] + [dim.dim_value for dim in network.get_input(0).shape]
        engine = builder.build_engine(network, config)
    return engine

with h5py.File("./valid-ds-psz11-new.h5", 'r') as fp:
    patch = fp['patch'][:]
    ploc  = fp['ploc'][:] - 0.5
    
input_tensor = patch

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

#engine = load_engine(trt_runtime, '0center-gpu-opset11_16384_FP16_TRT7.plan')


#build engine instead of loading
onnx_path = '0center-gpu-opset11.onnx'
engine_name = '0center-gpu-opset11_16384_FP16_TRT7.plan'
model = ModelProto()
batch_size = 16384
with open(onnx_path, "rb") as f:
    model.ParseFromString(f.read())

d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value

shape = [batch_size , d0, d1, d2]
print(shape)

engine = create_engine(TRT_LOGGER, onnx_path, shape)

print("TensorRT engine loaded.")
print("Engine input shape: ", engine.get_binding_shape(0))
print("Engine output shape: ", engine.get_binding_shape(1))
print("Allocating buffers...")
batch_size = 16384
h_input_1, d_input_1, h_output, d_output, stream = allocate_buffers(engine, batch_size, trt.float32)
print("Buffers allocated.")

iters = 20
total_time = 0.0

chunks = 0
if batch_size < 13799:
    chunks = len(input_tensor) // batch_size
    split_tensor = np.split(input_tensor[:batch_size*chunks], chunks, axis=0)
    print("Each chunk in the split tensor has shape: ", split_tensor[0].shape)

last_tensor = input_tensor[batch_size*chunks:]
shape = np.shape(last_tensor)
print(shape)
padded_array = np.zeros((batch_size, 1, 11, 11))
padded_array[:shape[0],:shape[1]] = last_tensor
last = padded_array

print("Last has shape: ", last.shape)


for i in range(iters):
    pred_list = np.empty((batch_size * (chunks + 1), 1, 2), dtype=np.float32)
    
    start_time = time.time()
    
    if batch_size < 13799:
        k = 0
        for j in range(chunks):
            pred_list[batch_size * j:batch_size*(j+1)] = (do_inference(engine, split_tensor[j], h_input_1, d_input_1, h_output, d_output, stream, batch_size, 2))
            k = j

        pred_list[batch_size*(k+1):batch_size*(k+2)] = (do_inference(engine, last, h_input_1, d_input_1, h_output, d_output, stream, batch_size, 2))
        
    else: 
        pred_list = (do_inference(engine, last, h_input_1, d_input_1, h_output, d_output, stream, batch_size, 2))
        
        
    end_time = time.time()
    
    print(end_time - start_time)
    
    # disregard the first iteration in our time measurement
    if i != 0:
        total_time += (end_time - start_time)
    
    pred_list = np.reshape(pred_list, (len(pred_list), 2))
    
print("Inference completed.")
print("Total time for inference: ", total_time)
print(pred_list[:10] + 0.5)
print()
print(ploc[:10] + 0.5)
    
# 95th percentile should be around 0.65 here
l2norm_ml = np.sqrt(np.sum((ploc - pred_list[:13799])**2, axis=1)) * 11
print("Error in pixel of %d samples for TRT: Avg: %.3f, 50th: %.3f, 75th: %.3f, 95th: %.3f, 99.5th: %.3f" % ((l2norm_ml.shape[0], l2norm_ml.mean(), ) + tuple(np.percentile(l2norm_ml, (50, 75, 95, 99.5))) )) 

print("TRT: ", total_time/iters)