#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <stdio.h>
#include <assert.h>

#include "cuda_runtime.h"

using Texel = uint16_t;
static constexpr int MAX_TEXEL = int(std::numeric_limits<Texel>::max());

static constexpr bool SAMPLE_NORMALIZED = true;

__global__ void testForWidth_kernel(dim3 texDim, cudaTextureObject_t src_tex, cudaSurfaceObject_t dst_surf, int rank) {

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
    const float fTex = 1 == rank ? tex1D<float>(src_tex, u) :
                       2 == rank ? tex2D<float>(src_tex, u, v) :
                                   tex3D<float>(src_tex, u, v, w);
     
    // Convert it to the non-normalized integer format.
    const Texel iTex = static_cast<Texel>(float(MAX_TEXEL) * fTex);
    
    // Write it out.
    if (1 == rank)
        surf1Dwrite<Texel>(iTex, dst_surf, x * sizeof(Texel), cudaBoundaryModeTrap);
    else if (2 == rank)
        surf2Dwrite<Texel>(iTex, dst_surf, x * sizeof(Texel), y, cudaBoundaryModeTrap);
    else
        surf3Dwrite<Texel>(iTex, dst_surf, x * sizeof(Texel), y, z, cudaBoundaryModeTrap);
}

#define ENSURE(expr) do { if (expr) break; printf("Error: %s\n", #expr); std::abort(); } while (false)
#define CUDA_ENSURE(expr) ENSURE(cudaSuccess == (expr))

static bool test(dim3 texDim) {

    const int rank = 1 == texDim.z ? 1 == texDim.y ? 1 : 2 : 3;
        
    const cudaExtent elemExtent = make_cudaExtent(texDim.x, texDim.y, texDim.z);
    const cudaExtent elemDim{ elemExtent.width, rank > 1 ? elemExtent.height : 0, rank > 2 ? elemExtent.depth : 0 };
    const cudaExtent byteExtent = make_cudaExtent(texDim.x * sizeof(Texel), texDim.y, texDim.z);

    // Create source data (on the host), fill it with runs of increasing [0...255] values.
    std::vector<Texel> src_hostArr(static_cast<size_t>(texDim.x) * texDim.y * texDim.z);
    std::iota(std::begin(src_hostArr), std::end(src_hostArr), 0);

    // Create a CUDA array and copy the data to it.
    const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<Texel>();
    cudaArray_t src_cudaArr{};
    CUDA_ENSURE(cudaMalloc3DArray(&src_cudaArr, &channelDesc, elemDim, 0));

    cudaMemcpy3DParms memcpy3DParms{};
    memcpy3DParms.srcPtr = make_cudaPitchedPtr(src_hostArr.data(), texDim.x * sizeof(Texel), texDim.x, texDim.y);
    memcpy3DParms.dstArray = src_cudaArr;
    memcpy3DParms.extent = elemExtent;
    memcpy3DParms.kind = cudaMemcpyDefault;
    CUDA_ENSURE(cudaMemcpy3D(&memcpy3DParms));

    // Create a CUDA texture object to access the CUDA array.
    cudaResourceDesc resTexDesc{};
    resTexDesc.resType = cudaResourceTypeArray;
    resTexDesc.res.array.array = src_cudaArr;
    
    cudaTextureDesc texDesc{};
    texDesc.filterMode = cudaFilterModeLinear;    
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = SAMPLE_NORMALIZED;
    
    cudaTextureObject_t src_cudaTex{};
    CUDA_ENSURE(cudaCreateTextureObject(&src_cudaTex, &resTexDesc, &texDesc, nullptr));

    // Create some CUDA memory to put the kernel result in.
    cudaArray_t dst_cudaArr{};
    CUDA_ENSURE(cudaMalloc3DArray(&dst_cudaArr, &channelDesc, elemDim, cudaArraySurfaceLoadStore));
    
    // Create a CUDA surface to write the kernel result through.
    cudaResourceDesc resSurfDesc{};
    resSurfDesc.resType = cudaResourceTypeArray;
    resSurfDesc.res.array.array = dst_cudaArr;
    
    cudaSurfaceObject_t dst_cudaSurf{};
    CUDA_ENSURE(cudaCreateSurfaceObject(&dst_cudaSurf, &resSurfDesc));

    // Launch the kernel and wait for it to finish.
    const dim3 blockDim{ 32, 32, 1 };
    const dim3 gridDim{
        (texDim.x + blockDim.x - 1) / blockDim.x,
        (texDim.y + blockDim.y - 1) / blockDim.y,
        (texDim.z + blockDim.z - 1) / blockDim.z
    };

    testForWidth_kernel<<<gridDim, blockDim>>>(texDim, src_cudaTex, dst_cudaSurf, rank);
    CUDA_ENSURE(cudaDeviceSynchronize());

    // Copy the result from the CUDA device memory back to host memory.
    std::vector<Texel> dst_hostArr(src_hostArr.size());
    memcpy3DParms = {};
    memcpy3DParms.srcArray = dst_cudaArr;
    memcpy3DParms.dstPtr = make_cudaPitchedPtr(dst_hostArr.data(), texDim.x * sizeof(Texel), texDim.x, texDim.y);
    memcpy3DParms.extent = elemExtent;
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
    CUDA_ENSURE(cudaDestroySurfaceObject(dst_cudaSurf));
    CUDA_ENSURE(cudaFreeArray(dst_cudaArr));
    CUDA_ENSURE(cudaDestroyTextureObject(src_cudaTex));
    CUDA_ENSURE(cudaFreeArray(src_cudaArr));

    return allEqual;
}

std::vector<bool> testAxis(int axis) {

    const unsigned int otherSize = 1;
    const unsigned int maxSize = 4300;// 16384;

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
