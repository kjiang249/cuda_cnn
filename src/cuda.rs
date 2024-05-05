// This is the skeleton for the CUDA implementation

use crate::cnn::*;
use rustacuda::function::BlockSize;
use rustacuda::{device, launch, stream};
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

// Fields need to be ordered this way so the DeviceBoxes are
// dropped before the Context. Otherwise the drop will panic.

pub struct CudaContext {
    conv_layer: DeviceBox<ConvLayer>,
    output_layer: DeviceBox<OutputLayer>,
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaContext {
    pub fn init(cnn: &Cnn) -> Result<Self, Box<dyn Error>> {
        // Err("Not implemented")?

        // Initialize the CUDA context
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

        let conv_layer = DeviceBox::new(&cnn.conv_layer).unwrap();
        let output_layer = DeviceBox::new(&cnn.output_layer).unwrap();

        let ptx = CString::new(include_str!("../kernel/kernel.ptx"))?;
        let module = Module::load_from_string(&ptx)?;
        let stream = Stream::new(StreamFlags::DEFAULT, None)?; //kj not sure which flags to use
        return Ok(CudaContext {
            conv_layer,
            output_layer,
            module,
            stream,
            _context: _ctx,
        });
    }

    pub fn compute(&mut self, input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {
        // Err("Not implemented")?
        let mut input_matrix = DeviceBox::new(input).unwrap();
        let mut conv_output = DeviceBox::new(&ConvOutput([[[0.0; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE])).unwrap();
        let mut output = DeviceBox::new(&OutputVec([0.0; OUT_LAYER_SIZE])).unwrap();
        let mut elementwise_multi_output = DeviceBox::new(&OutputLayer([[0.0; OUT_NEURON_DIM];OUT_LAYER_SIZE])).unwrap();
        let mut partial_add_output = DeviceBox::new(&PartialSumOutput([[0.0; 200];OUT_LAYER_SIZE])).unwrap();        
        

        let module = &self.module;
        let stream = &self.stream;

        unsafe {
            //launch the kernel

            let conv_result = launch!(module.convolution_layer<<<(1, 1, 10), (32, 32, 1), 0, stream>>>(
                input_matrix.as_device_ptr(),
                self.conv_layer.as_device_ptr(),
                conv_output.as_device_ptr()
            ));
            conv_result?;

            let relu_result = launch!(module.relu_layer<<<(1, 1, 10), (32, 32, 1), 0, stream>>>(
                conv_output.as_device_ptr()
            ));
            relu_result?;

            let elementwise_multi_result = launch!(module.elementwise_multiply<<<(8, 1, 10), (512, 1, 1), 0, stream>>>(
                conv_output.as_device_ptr(),
                self.output_layer.as_device_ptr(),
                elementwise_multi_output.as_device_ptr()
            ));
            elementwise_multi_result?;

            let partial_add_result = launch!(module.partial_add<<<(1, 1, 10), (200, 1, 1), 0, stream>>>(
                elementwise_multi_output.as_device_ptr(),
                partial_add_output.as_device_ptr()
            ));
            partial_add_result?;

            let output_result = launch!(module.final_add<<<(1, 1, 1), (10, 1, 1), 0, stream>>>(
                partial_add_output.as_device_ptr(),
                output.as_device_ptr()
            ));
            output_result?;
        }

        self.stream.synchronize()?;

        let mut out_host = OutputVec([0.0; OUT_LAYER_SIZE]);
        output.copy_to(&mut out_host)?;

        Ok(out_host)

    }
}
