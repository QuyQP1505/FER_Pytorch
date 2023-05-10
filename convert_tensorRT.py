import torch
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import onnx
 
# Logger to capture errors, warnings, 
# and other information during the build and inference phases
TRT_LOGGER = trt.Logger()

def build_engine(onnx_file_path):
    
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
     
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    builder.max_workspace_size = 1 << 30
    
    # we have only one image in batch
    builder.max_batch_size = 1
    
    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True

    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine!")
 
    return engine, context


def main():

    # check that the model converted fine
    ONNX_FILE_PATH = 'resnet.onnx'
    onnx_model = onnx.load(ONNX_FILE_PATH)
    onnx.checker.check_model(onnx_model)

    # initialize TensorRT engine and parse ONNX model
    engine, context = build_engine(ONNX_FILE_PATH)

    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
            # allocate memory for output
            output_shape = engine.get_binding_shape(binding)
            output_size = trt.volume(output_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize
            device_output = cuda.mem_alloc(output_size)
            # create a stream in which to copy inputs/outputs and run inference
            stream = cuda.Stream()

    # pre-process the image
    input_image = preprocess_image("/media/data/Project_Only/MLOps/face-rec-avenger/data/01_raw/robert_downey_jr/test/robert_downey_jr36.png")

    # copy input to device
    cuda.memcpy_htod_async(device_input, input_image, stream)

    # execute model
    context.execute_async_v2(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)

    # allocate buffers to hold output
    output_data = np.empty_like(output_shape, dtype=np.float32)

    # copy output from device to host
    cuda.memcpy_dtoh_async(output_data, device_output, stream)

    # wait for the stream to finish
    stream.synchronize()

    # post-process output
    postprocess(output_data)


if __name__ == '__main__':
    main()