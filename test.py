import onnxruntime as ort

# set providers to ['TensorrtExecutionProvider', 'CUDAExecutionProvider'] with TensorrtExecutionProvider having the higher priority.

sess = ort.InferenceSession('model.onnx', providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])