#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <stdio.h>
#include <assert.h>

#include "cuda_runtime.h"

using Texel = uint8_t;
static constexpr int MAX_TEXEL = int(std::numeric_limits<Texel>::max());

static constexpr bool SAMPLE_NORMALIZED = false;

__global__ void testForWidth_kernel(dim3 texDim, cudaTextureObject_t src_tex, Texel* dst_arr, int dst_elemPitch) {

    // Get the integer sample coordinates.
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    // We're done if we're not within the texture dimensions.
    if (x >= texDim.x) return;
    if (y >= texDim.y) return;
    if (z >= texDim.z) return;

    const float texNormalizeX = SAMPLE_NORMALIZED ? float(texDim.x) : 1.0f;
    const float texNormalizeY = SAMPLE_NORMALIZED ? float(texDim.y) : 1.0f;
    const float texNormalizeZ = SAMPLE_NORMALIZED ? float(texDim.z) : 1.0f;

    // Convert to floating point texture coordinates at the texel center.
    const float u = (float(x) + 0.5f) / texNormalizeX;
    const float v = (float(y) + 0.5f) / texNormalizeY;
    const float w = (float(z) + 0.5f) / texNormalizeZ;

    // Fetch the normalized texel value
    const float fTex = tex3D<float>(src_tex, u, v, w);

    // Convert it to the non-normalized integer format.
    const Texel iTex = static_cast<Texel>(float(MAX_TEXEL) * fTex);

    // Stick it in the output array.
    const size_t flatIdx = (y + z * texDim.y) * dst_elemPitch + x;
    dst_arr[flatIdx] = iTex;
}

#define ENSURE(expr) do { if (expr) break; printf("Error: %s\n", #expr); std::abort(); } while (false)
#define CUDA_ENSURE(expr) ENSURE(cudaSuccess == (expr))

static bool test(dim3 texDim) {

    const cudaExtent elemExtent = make_cudaExtent(texDim.x, texDim.y, texDim.z);
    const cudaExtent byteExtent = make_cudaExtent(texDim.x * sizeof(Texel), texDim.y, texDim.z);

    // Create source data (on the host), fill it with runs of increasing [0...255] values.
    std::vector<uint8_t> src_hostArr(static_cast<size_t>(texDim.x) * texDim.y * texDim.z);
    std::iota(std::begin(src_hostArr), std::end(src_hostArr), 0);

    // Create a CUDA array and copy the data to it.
    const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<Texel>();
    cudaArray_t src_cudaArr{};
    CUDA_ENSURE(cudaMalloc3DArray(&src_cudaArr, &channelDesc, elemExtent, 0));

    cudaMemcpy3DParms memcpy3DParms{};
    memcpy3DParms.srcPtr = make_cudaPitchedPtr(src_hostArr.data(), texDim.x * sizeof(Texel), texDim.x, texDim.y);
    memcpy3DParms.dstArray = src_cudaArr;
    memcpy3DParms.extent = elemExtent;
    memcpy3DParms.kind = cudaMemcpyDefault;
    CUDA_ENSURE(cudaMemcpy3D(&memcpy3DParms));

    // Create a CUDA texture object to access the CUDA array.
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = src_cudaArr;
    
    cudaTextureDesc texDesc{};
    texDesc.filterMode = cudaFilterModeLinear;    
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = SAMPLE_NORMALIZED;
    
    cudaTextureObject_t src_cudaTex{};
    CUDA_ENSURE(cudaCreateTextureObject(&src_cudaTex, &resDesc, &texDesc, nullptr));

    // Create some CUDA memory to put the kernel result in.
    cudaPitchedPtr dst_deviceArr{};
    CUDA_ENSURE(cudaMalloc3D(&dst_deviceArr, byteExtent));
    ENSURE(0 == dst_deviceArr.pitch % sizeof(Texel));
    const int dst_elemPitch = int(dst_deviceArr.pitch / sizeof(Texel));    

    // Launch the kernel and wait for it to finish.
    const dim3 blockDim{ 32, 32, 1 };
    const dim3 gridDim{
        (texDim.x + blockDim.x - 1) / blockDim.x,
        (texDim.y + blockDim.y - 1) / blockDim.y,
        (texDim.z + blockDim.z - 1) / blockDim.z
    };

    testForWidth_kernel<<<gridDim, blockDim>>>(texDim, src_cudaTex, static_cast<Texel*>(dst_deviceArr.ptr), dst_elemPitch);
    CUDA_ENSURE(cudaDeviceSynchronize());

    // Copy the result from the CUDA device memory back to host memory.
    std::vector<Texel> dst_hostArr(src_hostArr.size());
    memcpy3DParms = {};
    memcpy3DParms.srcPtr = dst_deviceArr;
    memcpy3DParms.dstPtr = make_cudaPitchedPtr(dst_hostArr.data(), texDim.x * sizeof(Texel), texDim.x, texDim.y);
    memcpy3DParms.extent = byteExtent;
    memcpy3DParms.kind = cudaMemcpyDefault;
    CUDA_ENSURE(cudaMemcpy3D(&memcpy3DParms));   

    // Compare the source and destination to verify we sampled correctly.
    bool allEqual = true;
    for (size_t i = 0; i < src_hostArr.size(); ++i) {
        const Texel src = src_hostArr[i];
        const Texel dst = dst_hostArr[i];

        if (src != dst) {
            allEqual = false;
            break;
        }
    }


    // Clean up
    CUDA_ENSURE(cudaFree(dst_deviceArr.ptr));
    CUDA_ENSURE(cudaDestroyTextureObject(src_cudaTex));    
    CUDA_ENSURE(cudaFreeArray(src_cudaArr));

    return allEqual;
}

std::vector<bool> testAxis(int axis) {

    const unsigned int otherSize = 1;
    const unsigned int maxSize = 16384;

    std::vector<bool> results;
    results.reserve(maxSize + 1);
    results.push_back(false); // zero texture size is N/A

    bool hasFailed = false;

    for (unsigned int size = 1; size <= maxSize; ++size) {
        
        dim3 texDim{ otherSize, otherSize, otherSize };
        (&texDim.x)[axis] = size;

        const bool ok = test(texDim);
        results.push_back(ok);

        if (!ok && !hasFailed) {
            printf("First failure on %c axis at size: %d\n", 'X' + axis, size);
            hasFailed = true;
        }

        if (0 == size % 1000)
            printf("Done %c axis at size: %d\n", 'X' + axis, size);
    }

    return results;
}

int main() {

    const auto xAxisResults = testAxis(0);
    const auto yAxisResults = testAxis(1);
    const auto zAxisResults = testAxis(2);

    std::string csvStr;
    csvStr.reserve(xAxisResults.size() * 3 * 23);

    for (size_t i = 0; i < xAxisResults.size(); ++i) {
        csvStr.append(xAxisResults[i] ? "1" : "0");
        csvStr.append(",");
        csvStr.append(yAxisResults[i] ? "1" : "0");
        csvStr.append(",");
        csvStr.append(zAxisResults[i] ? "1" : "0");
        csvStr.append(",\n");
    }

    FILE* const csvFile = fopen("cuda_texture_filtering.csv", "wt");
    ENSURE(csvFile != nullptr);
    ENSURE(csvStr.size() == fwrite(csvStr.data(), 1, csvStr.size(), csvFile));
    ENSURE(0 == fclose(csvFile));
    
    return 0;    
}
