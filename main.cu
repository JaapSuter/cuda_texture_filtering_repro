#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <stdio.h>
#include <assert.h>

#include "cuda_runtime.h"

__global__ void testForWidth_kernel(int texWidth, cudaTextureObject_t src_tex, float* dst_arr) {

    // Get the integer sample coordinates.
    const int x = threadIdx.x + blockIdx.x * blockDim.x;

    // We're done if we're not within the texture dimensions.
    if (x >= texWidth) return;

    // Convert to floating point texture coordinates at the texel center.
    const float u = (float(x) + 0.5f) / float(texWidth);
	
    // Fetch the normalized texel value
	const float fTex = tex1D<float>(src_tex, u);     

	// Write it out.
	dst_arr[x] = fTex;
}

#define ENSURE(expr) do { if (expr) break; printf("Error: %s\n", #expr); std::abort(); } while (false)
#define CUDA_ENSURE(expr) ENSURE(cudaSuccess == (expr))

static void test(int width) {

    const cudaExtent elemExtent = make_cudaExtent(width, 1, 1);
	const cudaExtent elemDim{ elemExtent.width, 0, 0 };
	const cudaExtent byteDim{ elemExtent.width * sizeof(float), 1, 1 };
    
    // Create source data (on the host), fill it with runs of increasing values.
    std::vector<float> src_hostArr(static_cast<size_t>(width));
    std::iota(std::begin(src_hostArr), std::end(src_hostArr), 0.0f);

    // Create a CUDA array and copy the data to it.
    const cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaArray_t src_cudaSwizzledArr{};
    CUDA_ENSURE(cudaMalloc3DArray(&src_cudaSwizzledArr, &channelDesc, elemDim, 0));

    cudaMemcpy3DParms memcpy3DParms{};
    memcpy3DParms.srcPtr = make_cudaPitchedPtr(src_hostArr.data(), width * sizeof(float), width, 1);
    memcpy3DParms.dstArray = src_cudaSwizzledArr;
    memcpy3DParms.extent = elemExtent;
    memcpy3DParms.kind = cudaMemcpyDefault;
    CUDA_ENSURE(cudaMemcpy3D(&memcpy3DParms));

    // Create a CUDA texture object to access the CUDA array.
    cudaResourceDesc resTexDesc{};
    resTexDesc.resType = cudaResourceTypeArray;
    resTexDesc.res.array.array = src_cudaSwizzledArr;
    
    cudaTextureDesc texDesc{};
    texDesc.filterMode = cudaFilterModeLinear;    
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = int(true);
    
    cudaTextureObject_t src_cudaTex{};
    CUDA_ENSURE(cudaCreateTextureObject(&src_cudaTex, &resTexDesc, &texDesc, nullptr));

    // Create some CUDA memory to put the kernel result in.
	float* dst_cudaArr = nullptr;
	CUDA_ENSURE(cudaMalloc(&dst_cudaArr, width * sizeof(float)));
	
    // Launch the kernel and wait for it to finish.
    const dim3 blockDim{ 32, 32, 1 };
    const dim3 gridDim{ (width + blockDim.x - 1) / blockDim.x, 1, 1 };

    testForWidth_kernel<<<gridDim, blockDim>>>(width, src_cudaTex, dst_cudaArr);
    CUDA_ENSURE(cudaDeviceSynchronize());

    // Copy the result from the CUDA device memory back to host memory.
	std::vector<float> dst_hostArr(static_cast<size_t>(width));
    memcpy3DParms = {};
	memcpy3DParms.srcPtr = make_cudaPitchedPtr(dst_cudaArr, width * sizeof(float), width, 1);
    memcpy3DParms.dstPtr = make_cudaPitchedPtr(dst_hostArr.data(), width * sizeof(float), width, 1);
    memcpy3DParms.extent = byteDim;
    memcpy3DParms.kind = cudaMemcpyDefault;
    CUDA_ENSURE(cudaMemcpy3D(&memcpy3DParms));   

    // Compare the source and destination to verify we sampled correctly.
    for (size_t i = 0; i < src_hostArr.size(); ++i) {
        const float src = src_hostArr[i];
        const float dst = dst_hostArr[i];

		ENSURE(src == dst);
    }


    // Clean up    
    CUDA_ENSURE(cudaFree(dst_cudaArr));
    CUDA_ENSURE(cudaDestroyTextureObject(src_cudaTex));
    CUDA_ENSURE(cudaFreeArray(src_cudaSwizzledArr));
}

int main() {

	const int width = 100;
	test(width);
    return 0;    
}
