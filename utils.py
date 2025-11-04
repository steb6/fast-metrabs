from pathlib import Path
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from loguru import logger


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class Runner:
    def __init__(self, engine_path):
        logger.info(f'Loading {Path(engine_path).stem} engine...')

        G_LOGGER = trt.Logger(trt.Logger.ERROR)  # TODO PUT ERROR
        trt.init_libnvinfer_plugins(G_LOGGER, '')
        runtime = trt.Runtime(G_LOGGER)

        with open(engine_path, 'rb') as f:
            buf = f.read()
            engine = runtime.deserialize_cuda_engine(buf)

        # Create execution context
        context = engine.create_execution_context()

        # Prepare buffers using new TensorRT API (set_input_shape, set_tensor_address, execute_async_v3)
        # Identify input and output tensors by iterating over all I/O tensors
        inputs = []
        outputs = []
        input_names = []
        output_names = []

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            # Determine if this tensor is an input or output
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                input_names.append(tensor_name)
            else:
                output_names.append(tensor_name)

        # Allocate device memory for inputs
        # We will set shapes dynamically per inference, so allocate a reasonable max size
        # For now, get the shape from the engine (if static) or use a default profile shape
        for tensor_name in input_names:
            shape = engine.get_tensor_shape(tensor_name)
            # Replace -1 (dynamic dims) with 1 for allocation purposes
            shape = tuple([d if d > 0 else 1 for d in shape])
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            inputs.append(HostDeviceMem(host_mem, device_mem))

        # Allocate device memory for outputs
        # Output shapes may depend on input shapes, so we query after setting input shapes
        for tensor_name in output_names:
            shape = engine.get_tensor_shape(tensor_name)
            shape = tuple([d if d > 0 else 1 for d in shape])
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            outputs.append(HostDeviceMem(host_mem, device_mem))

        # Store
        self.stream = cuda.Stream()
        self.context = context
        self.engine = engine
        self.inputs = inputs
        self.outputs = outputs
        self.input_names = input_names
        self.output_names = output_names

        self.warmup()
        logger.success(f'{Path(engine_path).stem} engine loaded')

    def warmup(self):
        args = [np.random.rand(*inp.host.shape).astype(inp.host.dtype) for inp in self.inputs]
        self(*args)

    def __call__(self, *args):
        # Copy input data to host buffers
        for i, x in enumerate(args):
            x = x.ravel()
            np.copyto(self.inputs[i].host, x)

        # Set input shapes dynamically (if needed)
        for i, tensor_name in enumerate(self.input_names):
            actual_shape = args[i].shape
            self.context.set_input_shape(tensor_name, actual_shape)

        # Verify all shapes are specified
        assert self.context.all_binding_shapes_specified, "Not all input shapes are specified"

        # Reallocate output buffers if shapes changed (query actual output shapes from context)
        for i, tensor_name in enumerate(self.output_names):
            shape = self.context.get_tensor_shape(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            size = trt.volume(shape)
            # Check if reallocation is needed
            if self.outputs[i].host.size != size:
                self.outputs[i].host = cuda.pagelocked_empty(size, dtype)
                self.outputs[i].device = cuda.mem_alloc(self.outputs[i].host.nbytes)

        # Copy inputs to device
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        # Set tensor addresses (new API)
        for i, tensor_name in enumerate(self.input_names):
            self.context.set_tensor_address(tensor_name, int(self.inputs[i].device))
        for i, tensor_name in enumerate(self.output_names):
            self.context.set_tensor_address(tensor_name, int(self.outputs[i].device))

        # Execute inference (new API: execute_async_v3)
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy outputs back to host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        # Synchronize
        self.stream.synchronize()

        res = [out.host for out in self.outputs]

        return res