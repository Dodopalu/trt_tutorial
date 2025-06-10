import numpy as np
import io
from matplotlib import pyplot as plt

from torch import nn
import torch.onnx
import torch



path = 'ResNet20.th'
model = torch.jit.load(path)

input_tensor = torch.randn(None, 3, 32, 32)  # Example input tensor for ResNet20

# Export the model
torch.onnx.export(model,               # model being run
                  input_tensor,                         # model input (or a tuple for multiple inputs)
                  "0center-gpu-opset11.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})