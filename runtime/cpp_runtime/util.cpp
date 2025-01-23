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

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <string>
#include <cmath>

#include "NvInfer.h"
#include "util.h"

namespace util
{

size_t getMemorySize(const nvinfer1::Dims& dims, const int32_t elem_size)
{
    return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
}

ImageBase::ImageBase(const std::string& filename, const nvinfer1::Dims& dims)
    : mDims(dims)
{
    assert(4 == mDims.nbDims);
    assert(1 == mDims.d[0]);
    mPPM.filename = filename;
}

size_t ImageBase::volume() const
{
    return mDims.d[3] /* w */ * mDims.d[2] /* h */ * 3;
}

void ImageBase::read()
{
    std::ifstream infile(mPPM.filename, std::ifstream::binary);
    if (!infile.is_open())
    {
        std::cerr << "ERROR: cannot open PPM image file: " << mPPM.filename << std::endl;
    }
    infile >> mPPM.magic >> mPPM.w >> mPPM.h >> mPPM.max;

    infile.seekg(1, infile.cur);
    mPPM.buffer.resize(volume());
    infile.read(reinterpret_cast<char*>(mPPM.buffer.data()), volume());
    infile.close();
}

void ImageBase::write()
{
    std::ofstream outfile(mPPM.filename, std::ofstream::binary);
    if (!outfile.is_open())
    {
        std::cerr << "ERROR: cannot open PPM image file: " << mPPM.filename << std::endl;
    }
    outfile << mPPM.magic << " " << mPPM.w << " " << mPPM.h << " " << mPPM.max << std::endl;
    outfile.write(reinterpret_cast<char*>(mPPM.buffer.data()), volume());
    outfile.close();
}

RGBImageReader::RGBImageReader(const std::string& filename, const nvinfer1::Dims& dims)
    : ImageBase(filename, dims)
{
}

std::unique_ptr<float> RGBImageReader::process() const
{
    const int C = mDims.d[1];
    const int H = mDims.d[2];
    const int W = mDims.d[3];
    auto buffer = std::unique_ptr<float>{new float[volume()]};

    // Get p10 and p99 and clip
    std::vector<double> list(volume());
    for (int i = 0; i < volume(); i ++) {
        list[i] = static_cast<double>(mPPM.buffer[i]);
    }
    std::sort(list.begin(), list.end());
    float p10 = list[int(0.10 * volume())];
    float p99 = list[int(0.99 * volume())];

    float diff = p99 - p10;
    for (int i = 0; i < volume(); i ++) {
        if (static_cast<double>(mPPM.buffer[i]) > p99)
            list[i] = 255;
        else if (static_cast<double>(mPPM.buffer[i]) < p10)
            list[i] = 0;
        else
            list[i] = (static_cast<double>(mPPM.buffer[i]) - p10) / diff * 255;
    }

    // Calculate the mean and std
    double sum = 0;
    for (int i = 0; i < volume(); i++)
    {
        sum += list[i];
    }

    double mean = sum / volume();

    double std = 0;
    for (int i = 0; i < volume(); i++)
    {
        std += pow(mean - list[i], 2);
    }
    std = std / volume();
    std = sqrt(std);

    if (mPPM.h == H && mPPM.w == W)
    {
        for (int c = 0; c < C; c++)
        {
            for (int j = 0, HW = H * W; j < HW; ++j)
            {
                buffer.get()[c * HW + j] = (static_cast<float>((list[j * C + c] - mean) / std));
            }
        }
    }
    else
    {
        assert(!"Specified dimensions don't match PPM image");
    }

    return buffer;
}

ArgmaxImageWriter::ArgmaxImageWriter(const std::string& filename, const nvinfer1::Dims& dims)
    : ImageBase(filename, dims)
{
}

void ArgmaxImageWriter::process(const float* buffer)
{
    mPPM.magic = "P6";
    mPPM.w = mDims.d[3];
    mPPM.h = mDims.d[2];
    mPPM.max = 255;
    mPPM.buffer.resize(volume());
    for (int j = 0, HW = mPPM.h * mPPM.w; j < HW; ++j)
    {
        if (buffer[j] > 0.5) {
            mPPM.buffer.data()[j*3] = mPPM.max;
            mPPM.buffer.data()[j*3+1] = mPPM.max;
            mPPM.buffer.data()[j*3+2] = mPPM.max;
        }
        else {
            mPPM.buffer.data()[j*3] = 0;
            mPPM.buffer.data()[j*3+1] = 0;
            mPPM.buffer.data()[j*3+2] = 0;

        }
    }
}

ImageWriter::ImageWriter(const std::string& filename, const nvinfer1::Dims& dims)
    :ImageBase(filename, dims)
{
}

void ImageWriter::setBuffer(const float *buffer)
{
    mPPM.buffer.resize(volume());
    mPPM.magic = "P6";
    mPPM.w = mDims.d[3];
    mPPM.h = mDims.d[2];
    mPPM.max = 255;

    float max = 0;
    float min = 99999;
    for (int i = 0; i < volume(); i ++) {
        if (buffer[i] > max)
            max = buffer[i];
        if (buffer[i] < min)
            min = buffer[i];
    }

    for (int c = 0; c < 3; c ++) {
        for(int i = 0, HW = mPPM.h * mPPM.w; i < HW; i ++) {
            mPPM.buffer.data()[i * 3 + c] = static_cast<uint8_t>(buffer[c * HW + i] / 5 * 255);
        }
    }
}

}; // namespace util
