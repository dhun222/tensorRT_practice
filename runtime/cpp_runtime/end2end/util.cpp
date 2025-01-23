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
/*
ImageBase::ImageBase(const std::string& filename, const int c, const int h, const int w)
{
    mPPM.filename = filename;
    mC = c;
    mH = h;
    mW = w;
}

size_t ImageBase::volume() const
{
    return mC * mH * mW;
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

void ImageBase::write(const std::vector<uint8_t>& buffer)
{
    std::ofstream outfile(mPPM.filename, std::ofstream::binary);
    if (!outfile.is_open())
    {
        std::cerr << "ERROR: cannot open PPM image file: " << mPPM.filename << std::endl;
    }
    outfile << mPPM.magic << " " << mPPM.w << " " << mPPM.h << " " << mPPM.max << std::endl;
    outfile.write(reinterpret_cast<char*>(buffer.data()), volume());
    outfile.close();
}
// ImageBase
*/

// BatchRGBImageReader
BatchRGBImageReader::BatchRGBImageReader(const std::vector<std::string>& filenames, const int b, const int c, const int h, const int w)
{
    mB = b;
    mC = c;
    mH = h;
    mW = w;
    mVolume = c * h * w;
    mFilenames = filenames;
}

void BatchRGBImageReader::_process(uint8_t* buffer, int b)
{
    // Get p10 and p99 and clip
    std::vector<double> list(mVolume);
    for (int i = 0; i < mVolume; i ++) {
        list[i] = static_cast<double>(buffer[i]);
    }
    std::sort(list.begin(), list.end());
    float p10 = list[int(0.10 * mVolume)];
    float p99 = list[int(0.99 * mVolume)];

    float diff = p99 - p10;
    for (int i = 0; i < mVolume; i ++) {
        if (static_cast<double>(buffer[i]) > p99)
            list[i] = 255;
        else if (static_cast<double>(buffer[i]) < p10)
            list[i] = 0;
        else
            list[i] = (static_cast<double>(buffer[i]) - p10) / diff * 255;
    }

    // Calculate the mean and std
    double sum = 0;
    for (int i = 0; i < mVolume; i++)
    {
        sum += list[i];
    }

    double mean = sum / mVolume;

    double std = 0;
    for (int i = 0; i < mVolume; i++)
    {
        std += pow(mean - list[i], 2);
    }
    std = std / mVolume;
    std = sqrt(std);

    int B = mVolume * b;
    for (int c = 0; c < mC; c++)
    {
        for (int j = 0, HW = mH * mW; j < HW; ++j)
        {
            mBuffers[B + c * HW + j] = (static_cast<float>((list[j * mC + c] - mean) / std));
        }
    }
}

float* BatchRGBImageReader::read()
{
    mBuffers = new float[mB * mVolume];

    std::unique_ptr<uint8_t> tempBuffer{new uint8_t[mVolume]};
    for (int b = 0; b < mB; b ++) {
        std::ifstream infile(mFilenames[b], std::ifstream::binary);
        if (!infile.is_open())
        {
            std::cerr << "ERROR: cannot open PPM image file: " << mFilenames[b] << std::endl;
        }
        std::string magic;
        int w;
        int h;
        int max;
        infile >> magic >> w >> h >> max;
        if (magic != "P6" || w != mW || h != mH || max != 255) {
            std::cout << "File format does not match!!" << std::endl;
            return nullptr;
        }

        infile.seekg(1, infile.cur);
        infile.read(reinterpret_cast<char*>(tempBuffer.get()), mVolume);
        _process(tempBuffer.get(), b);
        infile.close();
    }

    return mBuffers;
}
// BatchRGBImageReader end

// SIngleCHannelImageWriter
SingleChannelImageWriter::SingleChannelImageWriter(const std::string& filename, const int h, const int w)
{
    mFilename = filename;
    mBuffer.reset(new uint8_t[3 * h * w]);
    mW = w;
    mH = h;
}


void SingleChannelImageWriter::_process(const float* buffer)
{
    std::string magic = "P6";
    int max = 255;
    for (int j = 0, HW = mH * mW; j < HW; ++j)
    {
        if (buffer[j] > 0.5) {
            mBuffer.get()[j*3] = max;
            mBuffer.get()[j*3+1] = max;
            mBuffer.get()[j*3+2] = max;
        }
        else {
            mBuffer.get()[j*3] = 0;
            mBuffer.get()[j*3+1] = 0;
            mBuffer.get()[j*3+2] = 0;
        }
    }
}

void SingleChannelImageWriter::write(const float* buffer)
{
    _process(buffer);
    std::ofstream outfile(mFilename, std::ofstream::binary);
    if (!outfile.is_open())
    {
        std::cerr << "ERROR: cannot write PPM image file: " << mFilename << std::endl;
    }
    outfile << "P6" << " " << mW << " " << mH << " " << 255 << std::endl;
    outfile.write(reinterpret_cast<char*>(mBuffer.get()), 3 * mH * mW);
    outfile.close();
}
// SingleChannelImageWriter end

}; // namespace util
