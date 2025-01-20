trtexec --onnx=unet.onnx --saveEngine=unet_fp16.engine --inputIOFormats=fp32:chw --outputIOFormats=fp32:chw --fp16
