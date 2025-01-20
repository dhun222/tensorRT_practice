import torch
import torch.onnx as torch_onnx
import argparse

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--onnx', type=str, default='unet.onnx')
  parser.add_argument('--batch_size', type=int, default=1)

  a = parser.parse_args()

  onnx_path = a.onnx
  input_shape = (3, 256, 256)
  dummy_input = torch.zeros(a.batch_size, *input_shape)
  model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)
  model.train(False)
  
  inputs = ['input']
  outputs = ['output']
  dynamic_axes = {'input': {0: 'batch'}, 'output':{0:'batch'}}
  torch.onnx.export(model, dummy_input, onnx_path, input_names=inputs, output_names=outputs, dynamic_axes=dynamic_axes)

if __name__=='__main__':
  main()
