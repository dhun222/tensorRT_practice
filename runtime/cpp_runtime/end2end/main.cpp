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


int main(int argc, char* argv[])
{
    std::vector<std::string> input_files;
    std::string output_path;

    // parse arguments
    bool f_input_files = false;
    for (int i = 1; i < argc; i ++) {
        if (std::string("--input_files") == argv[i])
            f_input_files = true;
        else if(std::string("--output_path") == argv[i]) {
            i++;
            output_path = argv[i];
            f_input_files = false;
        }
        else if(f_input_files)
            input_files.push_back(argv[i]);
    }

    if (input_files.size() == 0 || output_path.empty()) {
        std::cout << "usage: --input_files <input file names> --output_path <directory for output files>" << std::endl;
        return 0;
    }

    const int32_t n_input = input_files.size();
    const int32_t width = 256;
    const int32_t height = 256;
    
    std::cout << "Input files: " << std::endl;
    for (int i = 0; i < n_input; i ++) {
        std::cout << "   " << input_files[i] << std::endl;
    }
    std::cout << "Output path: " << std::endl;
    std::cout << "   " << output_path << std::endl;
    
    // Read input data from file
    // Normalization process makes slightly different result from python version which can make the prediction also different. 
    auto input_image{util::BatchRGBImageReader(input_files, n_input, 3, height, width)};
    std::unique_ptr<float> input_buffer{input_image.read()};

    // Define the metadata for the input and output data
    nvinfer1::Dims4 input_dims(1, 3, height, width);
    auto input_size = util::getMemorySize(input_dims, sizeof(float));
    nvinfer1::Dims output_dims = nvinfer1::Dims4{1, 1, height, width};
    auto output_size = util::getMemorySize(output_dims, sizeof(float));
    std::unique_ptr<float> output_buffer{new float[output_size]};

    // create runtime
    util::UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};

    std::cout << "fp16 engine" << std::endl;

    // load engine from file
    std::string enginePath = "unet_fp16.engine";
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (engineFile.fail()) {
        std::cout << "Failed to load engine" << std::endl;
        return 0;
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
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), fsize);
    nvinfer1::IExecutionContext* context = engine-> createExecutionContext();
    if (!context) {
        std::cout << "Execution context not created" << std::endl;
        return 0;
    }

    std::cout << "engine succesfully loaded" << std::endl;

    // Define the input, output binding
    auto input_idx = engine->getBindingIndex("input");
    if (input_idx == -1) {
        std::cout << "Failed to get binding index" << std::endl;
        return false;
    }
    if (engine->getBindingDataType(input_idx) != nvinfer1::DataType::kFLOAT) {
        std::cout << "Input datatype not match" << std::endl;
        return false;
    }
    context->setBindingDimensions(input_idx, input_dims);

    auto output_idx = engine->getBindingIndex("output");
    if (output_idx == -1) {
        std::cout << "Failed to get binding index" << std::endl;
        return false;
    }
    if (engine->getBindingDataType(output_idx) != nvinfer1::DataType::kFLOAT) {
        std::cout << "Output datatype not match" << std::endl;
        return false;
    }

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

    // Create cuda stream
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cout << "ERROR: cuda stream creation failed." << std::endl;
        return false;
    }
    for (int i = 0; i < n_input; i++) {
        // Copy image into input binding memory
        size_t volume = height * width;
        if (cudaMemcpyAsync(input_mem, &(input_buffer.get()[i * 3 * volume]), input_size, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
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
        if (cudaMemcpyAsync(output_buffer.get(), output_mem, output_size, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
            std::cout << "ERROR: CUDA memory copy of output failed, size = " << output_size << " bytes" << std::endl;
            return false;
        }
        cudaStreamSynchronize(stream);
        
        // Write the output data to file
        std::string output_file = "output_" + std::to_string(i) + ".ppm";
        auto output_image(util::SingleChannelImageWriter(output_file, height, width));
        output_image.write(output_buffer.get());
    }

    cudaFree(input_mem);
    cudaFree(output_mem);



    return 0;
}