/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TENSORRT_SAMPLE_COMMON_UTILS_H
#define TENSORRT_SAMPLE_COMMON_UTILS_H
#include <memory>
#include <string>
#include <vector>

#include "NvInfer.h"

namespace util
{

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <typename T>
using UniquePtr = std::unique_ptr<T, util::InferDeleter>;

size_t getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size);

class BatchRGBImageReader
{
public:
    BatchRGBImageReader(const std::vector<std::string>& file_names, const int b, const int c, const int h, const int w);
    size_t volume() const;
    float* read();
private:
    std::vector<std::string> mFilenames;
    void _process(uint8_t* buffer, int b);
    float* mBuffers;
    int mB;
    int mC;
    int mH;
    int mW;
    int mVolume;
};

class SingleChannelImageWriter
{
public:
    SingleChannelImageWriter(const std::string& filename, const int h, const int w);
    void write(const float* buffer);
private:
    void _process(const float* buffer);
    std::string mFilename;
    int mH;
    int mW;
    std::unique_ptr<uint8_t> mBuffer;
};

}; // namespace util

#endif // TENSORRT_SAMPLE_COMMON_UTILS_H
