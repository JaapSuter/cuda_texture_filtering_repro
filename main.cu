#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <stdio.h>
#include <assert.h>

#include "cuda_runtime.h"

__global__ void testForWidth_kernel(dim3 texDim, cudaTextureObject_t src_tex, float* dst_arr, int rank) {

    // Get the integer sample coordinates.
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    // We're done if we're not within the texture dimensions.
    if (x >= texDim.x) return;
    if (y >= texDim.y) return;
    if (z >= texDim.z) return;

    const float texNormalizeX = float(texDim.x);
    const float texNormalizeY = float(texDim.y);
    const float texNormalizeZ = float(texDim.z);

    // Convert to floating point texture coordinates at the texel center.
    const float u = (float(x) + 0.5f) / texNormalizeX;
    const float v = (float(y) + 0.5f) / texNormalizeY;
    const float w = (float(z) + 0.5f) / texNormalizeZ;

    // Fetch the normalized texel value
    const float fTex = 1 == rank ? tex1D<float>(src_tex, u) :
                       2 == rank ? tex2D<float>(src_tex, u, v) :
                                   tex3D<float>(src_tex, u, v, w);
     
    
    // Write it out.
	if (1 == rank)
		dst_arr[x] = fTex;
}

#define ENSURE(expr) do { if (expr) break; printf("Error: %s\n", #expr); std::abort(); } while (false)
#define CUDA_ENSURE(expr) ENSURE(cudaSuccess == (expr))

static bool test(dim3 texDim) {

    const int rank = 1 == texDim.z ? 1 == texDim.y ? 1 : 2 : 3;
        
    const cudaExtent elemExtent = make_cudaExtent(texDim.x, texDim.y, texDim.z);
    const cudaExtent elemDim{ elemExtent.width, rank > 1 ? elemExtent.height : 0, rank > 2 ? elemExtent.depth : 0 };
    const cudaExtent byteExtent = make_cudaExtent(texDim.x * sizeof(float), texDim.y, texDim.z);

    // Create source data (on the host), fill it with runs of increasing [0...255] values.
    std::vector<float> src_hostArr(static_cast<size_t>(texDim.x) * texDim.y * texDim.z);
    std::iota(std::begin(src_hostArr), std::end(src_hostArr), 0.0f);

    // Create a CUDA array and copy the data to it.
    const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray_t src_cudaArr{};
    CUDA_ENSURE(cudaMalloc3DArray(&src_cudaArr, &channelDesc, elemDim, 0));

    cudaMemcpy3DParms memcpy3DParms{};
    memcpy3DParms.srcPtr = make_cudaPitchedPtr(src_hostArr.data(), texDim.x * sizeof(float), texDim.x, texDim.y);
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
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = int(true);
    
    cudaTextureObject_t src_cudaTex{};
    CUDA_ENSURE(cudaCreateTextureObject(&src_cudaTex, &resTexDesc, &texDesc, nullptr));

    // Create some CUDA memory to put the kernel result in.
	float* dst_cudaArr = nullptr;
	CUDA_ENSURE(cudaMalloc(&dst_cudaArr, texDim.x));
	
    // Launch the kernel and wait for it to finish.
    const dim3 blockDim{ 32, 32, 1 };
    const dim3 gridDim{
        (texDim.x + blockDim.x - 1) / blockDim.x,
        (texDim.y + blockDim.y - 1) / blockDim.y,
        (texDim.z + blockDim.z - 1) / blockDim.z
    };

    testForWidth_kernel<<<gridDim, blockDim>>>(texDim, src_cudaTex, dst_cudaArr, rank);
    CUDA_ENSURE(cudaDeviceSynchronize());

    // Copy the result from the CUDA device memory back to host memory.
	std::vector<float> dst_hostArr(size_t(texDim.x));
    memcpy3DParms = {};
	memcpy3DParms.srcPtr = make_cudaPitchedPtr(dst_cudaArr, texDim.x * sizeof(float), texDim.x, texDim.y);
    memcpy3DParms.dstPtr = make_cudaPitchedPtr(dst_hostArr.data(), texDim.x * sizeof(float), texDim.x, texDim.y);
    memcpy3DParms.extent = elemExtent;
    memcpy3DParms.kind = cudaMemcpyDefault;
    CUDA_ENSURE(cudaMemcpy3D(&memcpy3DParms));   

    // Compare the source and destination to verify we sampled correctly.
    bool allEqual = true;
    for (size_t i = 0; i < src_hostArr.size(); ++i) {
        const float src = src_hostArr[i];
        const float dst = dst_hostArr[i];

        if (src != dst) {
            allEqual = false;
            break;
        }
    }


    // Clean up    
    CUDA_ENSURE(cudaFree(dst_cudaArr));
    CUDA_ENSURE(cudaDestroyTextureObject(src_cudaTex));
    CUDA_ENSURE(cudaFreeArray(src_cudaArr));

    return allEqual;
}


int main() {

	const bool ok = test(dim3{ 10, 1, 1 });    
	if (!ok)
		throw "what the heck";
    return 0;    
}
