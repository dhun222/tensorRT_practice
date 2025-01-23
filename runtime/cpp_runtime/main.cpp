#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <memory>
#include <numeric>
#include <chrono>
#include <cuda_runtime_api.h>
#include "util.h"
#include "NvInfer.h"
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

bool infer(nvinfer1::IExecutionContext *context, const size_t input_size, const float* input_buffer, const size_t output_size, float* output_buffer)
{
    // Allocate CUDA memory
    void* input_mem{nullptr};
    if (cudaMalloc(&input_mem, input_size) != cudaSuccess)
    {
        std::cout << "ERROR: input cuda memory allocation failed, size = " << input_size << " bytes" << std::endl;
        return false;
    }
    void* output_mem{nullptr};
    if (cudaMalloc(&output_mem, output_size) != cudaSuccess)
    {
        std::cout << "ERROR: output cuda memory allocation failed, size = " << output_size << " bytes" << std::endl;
        return false;
    }

    // Copy image into input binding memory
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cout << "ERROR: cuda stream creation failed." << std::endl;
        return false;
    }

    if (cudaMemcpyAsync(input_mem, input_buffer, input_size, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        std::cout << "ERROR: CUDA memory copy of input failed, size = " << input_size << " bytes" << std::endl;
        return false;
    }

    // Inference
    void* bindings[] = {input_mem, output_mem};
    bool status = context->enqueueV2(bindings, stream, nullptr);
    if (!status) {
        std::cout << "ERROR: Inference failed" << std::endl;
        return false;
    }

    // Copy prediction into output buffer
    if (cudaMemcpyAsync(output_buffer, output_mem, output_size, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        std::cout << "ERROR: CUDA memory copy of output failed, size = " << output_size << " bytes" << std::endl;
        return false;
    }
    cudaStreamSynchronize(stream);

    cudaFree(input_mem);
    cudaFree(output_mem);

    return true;
}

int main()
{
    const int32_t batch = 1;
    const int32_t width = 256;
    const int32_t height = 256;
    
    // Read input data from file
    // Normalization process makes slightly different result from python version which can make the prediction also different. 
    std::string input_filename = "../../data/image1.ppm";
    auto input_dims = nvinfer1::Dims4{batch, 3, height, width};
    auto input_image{util::RGBImageReader(input_filename, input_dims)};
    input_image.read();
    auto input_buffer = input_image.process();

    // Write normalized input image to compare it with python version. 
    util::ImageWriter test("test.ppm", input_dims);
    test.setBuffer(input_buffer.get());
    test.write();

    // Define the metadata for the input and output data
    auto input_size = util::getMemorySize(input_dims, sizeof(float));
    nvinfer1::Dims output_dims = nvinfer1::Dims4{batch, 1, height, width};
    auto output_size = util::getMemorySize(output_dims, sizeof(float));
    std::unique_ptr<float> output_buffer{new float[output_size]};

    // create runtime
    util::UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};

    //------------------------------------------------ fp16 engine ---------------------------------------------------------------
    std::cout << "fp16 engine" << std::endl;

    // load engine from file
    std::string enginePath = "../../unet_fp16.engine";
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (engineFile.fail()) {
        std::cout << "Failed to load engine" << std::endl;
    }

    engineFile.seekg(0, engineFile.end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if (!engineFile) {
        std::cout << "Error loading engine file: " << enginePath << std::endl;
        return 0;
    }

    // Init the engine and create context. 
    nvinfer1::ICudaEngine* engine_fp16 = runtime->deserializeCudaEngine(engineData.data(), fsize);
    nvinfer1::IExecutionContext* context_fp16 = engine_fp16-> createExecutionContext();
    if (!context_fp16) {
        std::cout << "Execution context not created" << std::endl;
        return 0;
    }

    std::cout << "engine succesfully loaded" << std::endl;

    // Define the input, output binding
    auto input_idx_fp16 = engine_fp16->getBindingIndex("input");
    if (input_idx_fp16 == -1) {
        std::cout << "Failed to get binding index" << std::endl;
        return false;
    }
    if (engine_fp16->getBindingDataType(input_idx_fp16) != nvinfer1::DataType::kFLOAT) {
        std::cout << "Input datatype not match" << std::endl;
        return false;
    }
    context_fp16->setBindingDimensions(input_idx_fp16, input_dims);

    auto output_idx = engine_fp16->getBindingIndex("output");
    if (output_idx == -1) {
        std::cout << "Failed to get binding index" << std::endl;
        return false;
    }
    if (engine_fp16->getBindingDataType(output_idx) != nvinfer1::DataType::kFLOAT) {
        std::cout << "Output datatype not match ->" << std::endl;
        std::cout << static_cast<int>(engine_fp16->getBindingDataType(output_idx)) << std::endl;
        return false;
    }

    // Infer to save the output image as file
    infer(context_fp16, input_size, input_buffer.get(), output_size, output_buffer.get());

    std::string output_file_fp16 = "output_fp16.ppm";
    auto output_image_fp16(util::ArgmaxImageWriter(output_file_fp16, output_dims));
    output_image_fp16.process(output_buffer.get());
    output_image_fp16.write();

    std::cout << "Output image written" << std::endl;

    // Repeatedly do the inference and measure the time. 
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < 1000 * 10; i ++) {
        infer(context_fp16, input_size, input_buffer.get(), output_size, output_buffer.get());
    }
    auto end = std::chrono::system_clock::now();

    std::cout << "fp16 result" << std::endl;
    std::cout << "total time(s): " << std::setprecision(3) << static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) / 1000 << std::endl;
    std::cout << "time per infer(ms): " << std::setprecision(3) << static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 10000 / 1000 << std::endl;

    context_fp16->destroy();
    engine_fp16->destroy();

    std::cout << "fp16 engine destroyed" << std::endl << std::endl;
    //---------------------------------------------- fp16 engine end -------------------------------------------------------------

    //------------------------------------------------ int8 engine ---------------------------------------------------------------
    std::cout << "int8 engine" << std::endl;

    // Load engine from file. 
    std::string enginePath_int8 = "../../unet_int8.engine";
    std::ifstream engineFile_int8(enginePath_int8, std::ios::binary);
    if (engineFile_int8.fail()) {
        std::cout << "Failed to load engine" << std::endl;
    }

    engineFile_int8.seekg(0, engineFile_int8.end);
    auto fsize_int8 = engineFile_int8.tellg();
    engineFile_int8.seekg(0, engineFile_int8.beg);

    std::vector<char> engineData_int8(fsize_int8);
    engineFile_int8.read(engineData_int8.data(), fsize_int8);
    if (!engineFile_int8) {
        std::cout << "Error loading engine file: " << enginePath_int8 << std::endl;
        return 0;
    }

    // Init the engie and create context. 
    nvinfer1::ICudaEngine* engine_int8 = runtime->deserializeCudaEngine(engineData_int8.data(), fsize_int8);
    // create execution context
    nvinfer1::IExecutionContext* context_int8 = engine_int8-> createExecutionContext();
    if (!context_int8) {
        std::cout << "Execution context not created" << std::endl;
        return 0;
    }

    std::cout << "engine succesfully loaded" << std::endl;

    // Define the input, output binding
    auto input_idx_int8 = engine_int8->getBindingIndex("input");
    if (input_idx_int8 == -1) {
        std::cout << "Failed to get binding index" << std::endl;
        return false;
    }
    if (engine_int8->getBindingDataType(input_idx_int8) != nvinfer1::DataType::kFLOAT) {
        std::cout << "Input datatype not match" << std::endl;
        return false;
    }
    context_int8->setBindingDimensions(input_idx_int8, input_dims);

    auto output_idx_int8 = engine_int8->getBindingIndex("output");
    if (output_idx_int8 == -1) {
        std::cout << "Failed to get binding index" << std::endl;
        return false;
    }
    if (engine_int8->getBindingDataType(output_idx_int8) != nvinfer1::DataType::kFLOAT) {
        std::cout << "Output datatype not match ->" << std::endl;
        std::cout << static_cast<int>(engine_int8->getBindingDataType(output_idx_int8)) << std::endl;
        return false;
    }

    // Infer to save the output image as file
    infer(context_int8, input_size, input_buffer.get(), output_size, output_buffer.get());

    std::string output_file_int8 = "output_int8.ppm";
    auto output_image_int8(util::ArgmaxImageWriter(output_file_int8, output_dims));
    output_image_int8.process(output_buffer.get());
    output_image_int8.write();

    std::cout << "Output image written" << std::endl;

    // Repeatedly do the inference and measure the time. 
    auto start_int8 = std::chrono::system_clock::now();
    for (int i = 0; i < 1000 * 10; i ++) {
        infer(context_int8, input_size, input_buffer.get(), output_size, output_buffer.get());
    }
    auto end_int8 = std::chrono::system_clock::now();

    std::cout << "int8 result" << std::endl;
    std::cout << "total time(s): " << std::setprecision(3) << static_cast<float>(std::chrono::duration_cast<std::chrono::milliseconds>(end_int8 - start_int8).count()) / 1000 << std::endl;
    std::cout << "time per infer(ms): " << std::setprecision(3) << static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end_int8 - start_int8).count()) / 10000 / 1000 << std::endl;

    context_int8->destroy();
    engine_int8->destroy();

    std::cout << "int8 engine destroyed" << std::endl;
    //---------------------------------------------- int8 engine end -------------------------------------------------------------
    return 0;
}