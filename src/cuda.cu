#include "cuda.cuh"
#include "helper.h"
#include "cuda_runtime.h"

#include <cstring>
#include <cmath>
#include <device_launch_parameters.h>
#include <windows.h>

///
/// Host Variables
///
unsigned int cuda_pixel_contrib_count;                      // The number of contributors d_pixel_contrib_colours and d_pixel_contrib_depth have been allocated for
unsigned int cuda_particles_count;                          
Particle* cuda_particles;
unsigned int* cuda_pixel_contribs;
unsigned int* cuda_pixel_index;
unsigned char* cuda_pixel_contrib_colours;
float* cuda_pixel_contrib_depth;
CImage cuda_output_image;

int cuda_output_image_width;                                // Host storage of the output image dimensions (width)
int cuda_output_image_height;                               // Host storage of the output image dimensions (height)

///
/// Device Variables
///
__device__ Particle* d_particles;                           // Device pointer to a list of particles
__device__ unsigned int* d_pixel_contribs;                  // Device pointer to a histogram of the number of particles contributing to each pixel
__device__ unsigned int* d_pixel_index;                     // Device pointer to an index of unique offsets for each pixels contributing colours
__device__ unsigned char* d_pixel_contrib_colours;          // Device pointer to storage for each pixels contributing colours
__device__ float* d_pixel_contrib_depth;                    // Device pointer to storage for each pixels contributing colours' depth
__device__ unsigned char* d_output_image_data;              // Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device

__constant__ int D_OUTPUT_IMAGE_WIDTH;                      // Device storage of the output image dimensions (width)
__constant__ int D_OUTPUT_IMAGE_HEIGHT;                     // Device storage of the output image dimensions (height)
__constant__ unsigned int D_PARTICLES_COUNT;                // Number of particles in d_particles


///
/// Provided sorting algorithm.
///
void regular_sort_pairs(float* keys_start, unsigned char* colours_start, const int first, const int last) {
    // Based on https://www.tutorialspoint.com/explain-the-quick-sort-technique-in-c-language
    int i, j, pivot;
    float depth_t;
    unsigned char color_t[4];
    if (first < last) {
        pivot = first;
        i = first;
        j = last;
        while (i < j) {
            while (keys_start[i] <= keys_start[pivot] && i < last)
                i++;
            while (keys_start[j] > keys_start[pivot])
                j--;
            if (i < j) {
                // Swap key
                depth_t = keys_start[i];
                keys_start[i] = keys_start[j];
                keys_start[j] = depth_t;
                // Swap color
                memcpy(color_t, colours_start + (4 * i), 4 * sizeof(unsigned char));
                memcpy(colours_start + (4 * i), colours_start + (4 * j), 4 * sizeof(unsigned char));
                memcpy(colours_start + (4 * j), color_t, 4 * sizeof(unsigned char));
            }
        }
        // Swap key
        depth_t = keys_start[pivot];
        keys_start[pivot] = keys_start[j];
        keys_start[j] = depth_t;
        // Swap color
        memcpy(color_t, colours_start + (4 * pivot), 4 * sizeof(unsigned char));
        memcpy(colours_start + (4 * pivot), colours_start + (4 * j), 4 * sizeof(unsigned char));
        memcpy(colours_start + (4 * j), color_t, 4 * sizeof(unsigned char));
        // Recurse
        regular_sort_pairs(keys_start, colours_start, first, j - 1);
        regular_sort_pairs(keys_start, colours_start, j + 1, last);
    }
}


void iterative_sort_pairs(float* keys_start, unsigned char* colours_start, const int first, const int last)
{
    // Based on the iterative version of quicksort

    int stack[32]; // Stack to simulate recursion
    int top = -1;

    stack[++top] = first;
    stack[++top] = last;

    while (top >= 0) {
        // Pop indices from the stack
        int high = stack[top--];
        int low = stack[top--];

        int i = low, j = high;
        float pivot = keys_start[(low + high) / 2];

        while (i <= j) {
            while (keys_start[i] < pivot)
                i++;
            while (keys_start[j] > pivot)
                j--;

            if (i <= j) {
                // Swap key
                float depth_t = keys_start[i];
                keys_start[i] = keys_start[j];
                keys_start[j] = depth_t;

                // Swap color
                unsigned char color_t[4];
                for (int k = 0; k < 4; ++k) {
                    color_t[k] = colours_start[4 * i + k];
                    colours_start[4 * i + k] = colours_start[4 * j + k];
                    colours_start[4 * j + k] = color_t[k];
                }

                i++;
                j--;
            }
        }

        if (low < j) {
            stack[++top] = low;
            stack[++top] = j;
        }
        if (i < high) {
            stack[++top] = i;
            stack[++top] = high;
        }
    }
}

///
/// Provided sorting algorithm.
///
__device__ void cuda_sort_pairs(float* keys_start, unsigned char* colours_start, const int first, const int last) {
    // Based on the iterative version of quicksort

    int stack[32]; // Stack to simulate recursion
    int top = -1;

    stack[++top] = first;
    stack[++top] = last;

    while (top >= 0) {
        // Pop indices from the stack
        int high = stack[top--];
        int low = stack[top--];

        int i = low, j = high;
        float pivot = keys_start[(low + high) / 2];

        while (i <= j) {
            while (keys_start[i] < pivot)
                i++;
            while (keys_start[j] > pivot)
                j--;

            if (i <= j) {
                // Swap key
                float depth_t = keys_start[i];
                keys_start[i] = keys_start[j];
                keys_start[j] = depth_t;

                // Swap color
                unsigned char color_t[4];
                for (int k = 0; k < 4; ++k) {
                    color_t[k] = colours_start[4 * i + k];
                    colours_start[4 * i + k] = colours_start[4 * j + k];
                    colours_start[4 * j + k] = color_t[k];
                }

                i++;
                j--;
            }
        }

        if (low < j) {
            stack[++top] = low;
            stack[++top] = j;
        }
        if (i < high) {
            stack[++top] = i;
            stack[++top] = high;
        }
    }
}

///
/// CUDA Stage 1 Implementation (1/2)
/// Outer loop is parallelized via a kernel.
/// Inner loops are unrolled and collapsed.
/// Provides the best performance out of two implemented approaches. Used in cuda_stage1.
///
__global__ void cuda_stage1_outer_parallel_inner_unrolled(Particle* particles, unsigned int* pixel_contribs) {
    // Compute the index for the current thread
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < D_PARTICLES_COUNT)
    {
        // Compute bounding box [inclusive-inclusive]
        int x_min = (int)roundf(particles[index].location[0] - particles[index].radius);
        int y_min = (int)roundf(particles[index].location[1] - particles[index].radius);
        int x_max = (int)roundf(particles[index].location[0] + particles[index].radius);
        int y_max = (int)roundf(particles[index].location[1] + particles[index].radius);
        // Clamp bounding box to image bounds
        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= D_OUTPUT_IMAGE_WIDTH ? D_OUTPUT_IMAGE_WIDTH - 1 : x_max;
        y_max = y_max >= D_OUTPUT_IMAGE_HEIGHT ? D_OUTPUT_IMAGE_HEIGHT - 1 : y_max;

        // Collapse the loops over x and y
#pragma unroll
        for (int xy = 0; xy < ((x_max - x_min + 1) * (y_max - y_min + 1)); ++xy) {
            int x = x_min + (xy % (x_max - x_min + 1));
            int y = y_min + (xy / (x_max - x_min + 1));

            const float x_ab = (float)x + 0.5f - particles[index].location[0];
            const float y_ab = (float)y + 0.5f - particles[index].location[1];
            const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
            if (pixel_distance <= particles[index].radius) {
                const unsigned int pixel_offset = y * D_OUTPUT_IMAGE_WIDTH + x;
                atomicAdd(&pixel_contribs[pixel_offset], 1);
            }
        }
    }
}

///
/// CUDA Stage 1 Implementation (2/2)
/// Outer loop is parallelized via a kernel.
///
__global__ void cuda_stage1_outer_parallel(Particle* particles, unsigned int* pixel_contribs) {
    // Compute the index for the current thread
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < D_PARTICLES_COUNT)
    {
        // Compute bounding box [inclusive-inclusive]
        int x_min = (int)roundf(particles[index].location[0] - particles[index].radius);
        int y_min = (int)roundf(particles[index].location[1] - particles[index].radius);
        int x_max = (int)roundf(particles[index].location[0] + particles[index].radius);
        int y_max = (int)roundf(particles[index].location[1] + particles[index].radius);
        // Clamp bounding box to image bounds
        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= D_OUTPUT_IMAGE_WIDTH ? D_OUTPUT_IMAGE_WIDTH - 1 : x_max;
        y_max = y_max >= D_OUTPUT_IMAGE_HEIGHT ? D_OUTPUT_IMAGE_HEIGHT - 1 : y_max;
        // For each pixel in the bounding box, check that it falls within the radius
        for (int x = x_min; x <= x_max; ++x) {
            for (int y = y_min; y <= y_max; ++y) {
                const float x_ab = (float)x + 0.5f - particles[index].location[0];
                const float y_ab = (float)y + 0.5f - particles[index].location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= particles[index].radius) {
                    const unsigned int pixel_offset = y * D_OUTPUT_IMAGE_WIDTH + x;
                    atomicAdd(&pixel_contribs[pixel_offset], 1);
                }
            }
        }
    }
}

///
/// CUDA Stage 3 Implementation (1/2)
/// Outer loop is parallelized via a kernel.
/// Inner loop is optimized by rearranging operations to improve cache locality
/// Provides the best performance out of two implemented approaches. Used in cuda_stage3.
///
__global__ void cuda_stage3_order_rearranged_outer_parallel(unsigned int* pixel_index, unsigned char* pixel_contrib_colours, unsigned char* output_image_data)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < D_OUTPUT_IMAGE_WIDTH * D_OUTPUT_IMAGE_HEIGHT) {
        for (unsigned int j = pixel_index[index]; j < pixel_index[index + 1]; ++j) {
            // Get the color and opacity values
            const unsigned char* color = &pixel_contrib_colours[j * 4];
            const float opacity = (float)color[3] / (float)255;

            // Compute the blended color channels
            const float src_r = (float)color[0] * opacity;
            const float src_g = (float)color[1] * opacity;
            const float src_b = (float)color[2] * opacity;
            const float dest_r = output_image_data[(index * 3) + 0];
            const float dest_g = output_image_data[(index * 3) + 1];
            const float dest_b = output_image_data[(index * 3) + 2];
            const float blended_r = src_r + dest_r * (1 - opacity);
            const float blended_g = src_g + dest_g * (1 - opacity);
            const float blended_b = src_b + dest_b * (1 - opacity);

            // Store the blended color channels in the output image
            output_image_data[(index * 3) + 0] = (unsigned char)blended_r;
            output_image_data[(index * 3) + 1] = (unsigned char)blended_g;
            output_image_data[(index * 3) + 2] = (unsigned char)blended_b;
        }
    }
}

///
/// CUDA Stage 3 Implementation (2/2)
/// Outer loop is parallelized via a kernel.
///
__global__ void cuda_stage3_outer_parallel(unsigned int* pixel_index, unsigned char* pixel_contrib_colours, unsigned char* output_image_data)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < D_OUTPUT_IMAGE_WIDTH * D_OUTPUT_IMAGE_HEIGHT) {
        for (unsigned int j = pixel_index[index]; j < pixel_index[index + 1]; ++j) {
            const float opacity = (float)pixel_contrib_colours[j * 4 + 3] / (float)255;
            output_image_data[(index * 3) + 0] = (unsigned char)((float)pixel_contrib_colours[j * 4 + 0] * opacity + (float)output_image_data[(index * 3) + 0] * (1 - opacity));
            output_image_data[(index * 3) + 1] = (unsigned char)((float)pixel_contrib_colours[j * 4 + 1] * opacity + (float)output_image_data[(index * 3) + 1] * (1 - opacity));
            output_image_data[(index * 3) + 2] = (unsigned char)((float)pixel_contrib_colours[j * 4 + 2] * opacity + (float)output_image_data[(index * 3) + 2] * (1 - opacity));
        }
    }
}

__global__ void cuda_stage2_last_parallel(float* pixel_contrib_depth, unsigned char* pixel_contrib_colours, unsigned int* pixel_index) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < D_OUTPUT_IMAGE_WIDTH * D_OUTPUT_IMAGE_HEIGHT) {
        // Pair sort the colours which contribute to a single pigment
        cuda_sort_pairs(
            pixel_contrib_depth,
            pixel_contrib_colours,
            pixel_index[index],
            pixel_index[index + 1] - 1
        );

        __syncthreads();
    }
}

void serial_stage2() {
    // TODO: Delete if Stage 2 Changes.
	// Copying to host memory, as Stage 2 does not use CUDA
    CUDA_CALL(cudaMemcpy(cuda_pixel_contribs, d_pixel_contribs, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost))

    // Exclusive prefix sum across the histogram to create an index
    cuda_pixel_index[0] = 0;
    for (int i = 0; i < cuda_output_image.width * cuda_output_image.height; ++i) {
        cuda_pixel_index[i + 1] = cuda_pixel_index[i] + cuda_pixel_contribs[i];
    }

    // Recover the total from the index
    const unsigned int TOTAL_CONTRIBS = cuda_pixel_index[cuda_output_image.width * cuda_output_image.height];
    if (TOTAL_CONTRIBS > cuda_pixel_contrib_count) {
        // (Re)Allocate colour storage
        if (cuda_pixel_contrib_colours)
        {
            free(cuda_pixel_contrib_colours);
            CUDA_CALL(cudaFree(d_pixel_contrib_colours));
        }
        if (cuda_pixel_contrib_depth)
        {
            free(cuda_pixel_contrib_depth);
            CUDA_CALL(cudaFree(d_pixel_contrib_depth));
        }

        cuda_pixel_contrib_colours = (unsigned char*)malloc(TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
        cuda_pixel_contrib_depth = (float*)malloc(TOTAL_CONTRIBS * sizeof(float));
        cuda_pixel_contrib_count = TOTAL_CONTRIBS;

        CUDA_CALL(cudaMemcpy(d_pixel_index, cuda_pixel_index, ((cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int)), cudaMemcpyHostToDevice))
    	CUDA_CALL(cudaMalloc(&d_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char)))
    	CUDA_CALL(cudaMalloc(&d_pixel_contrib_depth, TOTAL_CONTRIBS * sizeof(float)))
    }

    // Reset the pixel contributions
    memset(cuda_pixel_contribs, 0, cuda_output_image.width * cuda_output_image.height * sizeof(unsigned int));
    CUDA_CALL(cudaMemset(d_pixel_contribs, 0, cuda_output_image.width * cuda_output_image.height * sizeof(unsigned int)));

    // Store colours according to index
    // For each particle, store a copy of the colour/depth in cuda_pixel_contribs for each contributed pixel
    for (unsigned int i = 0; i < cuda_particles_count; ++i) {
        // Compute bounding box [inclusive-inclusive]
        int x_min = (int)roundf(cuda_particles[i].location[0] - cuda_particles[i].radius);
        int y_min = (int)roundf(cuda_particles[i].location[1] - cuda_particles[i].radius);
        int x_max = (int)roundf(cuda_particles[i].location[0] + cuda_particles[i].radius);
        int y_max = (int)roundf(cuda_particles[i].location[1] + cuda_particles[i].radius);
        // Clamp bounding box to image bounds
        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= cuda_output_image.width ? cuda_output_image.width - 1 : x_max;
        y_max = y_max >= cuda_output_image.height ? cuda_output_image.height - 1 : y_max;
        // Store data for every pixel within the bounding box that falls within the radius
        for (int x = x_min; x <= x_max; ++x) {
            for (int y = y_min; y <= y_max; ++y) {
                const float x_ab = (float)x + 0.5f - cuda_particles[i].location[0];
                const float y_ab = (float)y + 0.5f - cuda_particles[i].location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= cuda_particles[i].radius) {
                    const unsigned int pixel_offset = y * cuda_output_image.width + x;
                    // Offset into cuda_pixel_contrib buffers is index + histogram
                    // Increment cuda_pixel_contribs, so next contributor stores to correct offset
                    const unsigned int storage_offset = cuda_pixel_index[pixel_offset] + (cuda_pixel_contribs[pixel_offset]++);
                    // Copy data to cuda_pixel_contrib buffers
                    memcpy(cuda_pixel_contrib_colours + (4 * storage_offset), cuda_particles[i].color, 4 * sizeof(unsigned char));
                    memcpy(cuda_pixel_contrib_depth + storage_offset, &cuda_particles[i].location[2], sizeof(float));
                }
            }
        }
    }
    // Pair sort the colours contributing to each pixel based on ascending depth
    for (int i = 0; i < cuda_output_image.width * cuda_output_image.height; ++i) {
        // Pair sort the colours which contribute to a single pigment
        iterative_sort_pairs(
            cuda_pixel_contrib_depth,
            cuda_pixel_contrib_colours,
            cuda_pixel_index[i],
            cuda_pixel_index[i + 1] - 1
        );
    }

    // TODO: Delete if Stage 2 Changes
    // Copies to device memory, as it is used by Stage 3
    CUDA_CALL(cudaMemcpy(d_pixel_index, cuda_pixel_index, (cuda_output_image_width * cuda_output_image_width + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_pixel_contrib_colours, cuda_pixel_contrib_colours, cuda_pixel_contrib_count * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice));
}

void cuda_begin(const Particle* init_particles, const unsigned int init_particles_count,
    const unsigned int out_image_width, const unsigned int out_image_height) {

    cuda_particles_count = init_particles_count;
    cuda_particles = (Particle*)malloc(sizeof(Particle) * cuda_particles_count);
    memcpy(cuda_particles, init_particles, init_particles_count * sizeof(Particle));
    
    cuda_pixel_contribs = (unsigned int*)malloc(out_image_width * out_image_height * sizeof(unsigned int));
    cuda_pixel_index = (unsigned int*)malloc((out_image_width * out_image_height + 1) * sizeof(unsigned int));

    cuda_pixel_contrib_colours = (unsigned char*)malloc(cuda_pixel_contrib_count * 4 * sizeof(unsigned char));
    cuda_pixel_contrib_depth = (float*)malloc(cuda_pixel_contrib_count * sizeof(float));

    cuda_pixel_contrib_count = 0;
    cuda_pixel_contrib_colours = 0;
    cuda_pixel_contrib_depth = 0;

    cuda_output_image_width = (int)out_image_width;
    cuda_output_image_height = (int)out_image_height;

    cuda_output_image.data = (unsigned char*)malloc(cuda_output_image_width * cuda_output_image_height * 3 * sizeof(unsigned char));
    cuda_output_image.height = cuda_output_image_height;
    cuda_output_image.width = cuda_output_image_width;
    cuda_output_image.channels = 3;

    CUDA_CALL(cudaMalloc(&d_particles, init_particles_count * sizeof(Particle)));
    CUDA_CALL(cudaMemcpy(d_particles, init_particles, init_particles_count * sizeof(Particle), cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMalloc(&d_pixel_contribs, ((int)out_image_width) * ((int)out_image_height) * sizeof(unsigned int)));
    CUDA_CALL(cudaMalloc(&d_pixel_index, (((int)out_image_width) * ((int)out_image_height) + 1) * sizeof(unsigned int)));
    CUDA_CALL(cudaMalloc((void**)&d_pixel_contrib_colours, cuda_pixel_contrib_count * 4 * sizeof(unsigned char)));
    d_pixel_contrib_colours = 0;
    d_pixel_contrib_depth = 0;

    CUDA_CALL(cudaMemcpyToSymbol(D_PARTICLES_COUNT, &cuda_particles_count, sizeof(unsigned int)));

    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_WIDTH, &cuda_output_image_width, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_HEIGHT, &cuda_output_image_height, sizeof(int)));
    CUDA_CALL(cudaMalloc(&d_output_image_data, cuda_output_image_width * cuda_output_image_height * 3 * sizeof(unsigned char)));
}

///
/// Default Functions.
///
void cuda_stage1()
{
    // Reset the pixel contributions histogram on the device
    CUDA_CALL(cudaMemset(d_pixel_contribs, 0, cuda_output_image.width * cuda_output_image.height * sizeof(unsigned int)))

    // Calculate the grid and block dimensions
	dim3 threadsPerBlock(32, 1, 1);
    dim3 blocksPerGrid((int)ceil((float)cuda_particles_count/threadsPerBlock.x));

    // Launch the CUDA kernel (using best performing implementation)
    // cuda_stage1_outer_parallel_inner_unrolled <<<blocksPerGrid, threadsPerBlock>>> (d_particles, d_pixel_contribs);

    // Another kernel, which performs worse. See the comment for cuda_stage1_outer_parallel.
	cuda_stage1_outer_parallel <<<blocksPerGrid, threadsPerBlock>>> (d_particles, d_pixel_contribs);

    CUDA_CALL(cudaGetLastError())
    CUDA_CALL(cudaDeviceSynchronize())

#ifdef VALIDATION
	CUDA_CALL(cudaMemcpy(cuda_pixel_contribs, d_pixel_contribs, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost))
    validate_pixel_contribs(cuda_particles, cuda_particles_count, cuda_pixel_contribs, cuda_output_image.width, cuda_output_image.height);
#endif
}

void cuda_stage2()
{
    /*
    // TODO: Delete if Stage 2 Changes.
	// Copying to host memory, as Stage 2 does not use CUDA
    CUDA_CALL(cudaMemcpy(cuda_pixel_contribs, d_pixel_contribs, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost))

	// Exclusive prefix sum across the histogram to create an index
	cuda_pixel_index[0] = 0;
    for (int i = 0; i < cuda_output_image.width * cuda_output_image.height; ++i) {
        cuda_pixel_index[i + 1] = cuda_pixel_index[i] + cuda_pixel_contribs[i];
    }

    // Recover the total from the index
    const unsigned int TOTAL_CONTRIBS = cuda_pixel_index[cuda_output_image.width * cuda_output_image.height];
    if (TOTAL_CONTRIBS > cuda_pixel_contrib_count) {
        // (Re)Allocate colour storage
        if (cuda_pixel_contrib_colours)
        {
            free(cuda_pixel_contrib_colours);
            CUDA_CALL(cudaFree(d_pixel_contrib_colours));
        }
        if (cuda_pixel_contrib_depth)
        {
            free(cuda_pixel_contrib_depth);
            CUDA_CALL(cudaFree(d_pixel_contrib_depth));
        }

        cuda_pixel_contrib_colours = (unsigned char*)malloc(TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
        cuda_pixel_contrib_depth = (float*)malloc(TOTAL_CONTRIBS * sizeof(float));
        cuda_pixel_contrib_count = TOTAL_CONTRIBS;

        CUDA_CALL(cudaMemcpy(d_pixel_index, cuda_pixel_index, ((cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int)), cudaMemcpyHostToDevice))
    	CUDA_CALL(cudaMalloc(&d_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char)))
    	CUDA_CALL(cudaMalloc(&d_pixel_contrib_depth, TOTAL_CONTRIBS * sizeof(float)))
    }

    // Reset the pixel contributions
    memset(cuda_pixel_contribs, 0, cuda_output_image.width * cuda_output_image.height * sizeof(unsigned int));
    CUDA_CALL(cudaMemset(d_pixel_contribs, 0, cuda_output_image.width * cuda_output_image.height * sizeof(unsigned int)))


	// Store colours according to index
    // For each particle, store a copy of the colour/depth in cuda_pixel_contribs for each contributed pixel
    for (unsigned int i = 0; i < cuda_particles_count; ++i) {
        // Compute bounding box [inclusive-inclusive]
        int x_min = (int)roundf(cuda_particles[i].location[0] - cuda_particles[i].radius);
        int y_min = (int)roundf(cuda_particles[i].location[1] - cuda_particles[i].radius);
        int x_max = (int)roundf(cuda_particles[i].location[0] + cuda_particles[i].radius);
        int y_max = (int)roundf(cuda_particles[i].location[1] + cuda_particles[i].radius);
        // Clamp bounding box to image bounds
        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= cuda_output_image.width ? cuda_output_image.width - 1 : x_max;
        y_max = y_max >= cuda_output_image.height ? cuda_output_image.height - 1 : y_max;
        // Store data for every pixel within the bounding box that falls within the radius
        for (int x = x_min; x <= x_max; ++x) {
            for (int y = y_min; y <= y_max; ++y) {
                const float x_ab = (float)x + 0.5f - cuda_particles[i].location[0];
                const float y_ab = (float)y + 0.5f - cuda_particles[i].location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= cuda_particles[i].radius) {
                    const unsigned int pixel_offset = y * cuda_output_image.width + x;
                    // Offset into cuda_pixel_contrib buffers is index + histogram
                    // Increment cuda_pixel_contribs, so next contributor stores to correct offset
                    const unsigned int storage_offset = cuda_pixel_index[pixel_offset] + (cuda_pixel_contribs[pixel_offset]++);
                    // Copy data to cuda_pixel_contrib buffers
                    memcpy(cuda_pixel_contrib_colours + (4 * storage_offset), cuda_particles[i].color, 4 * sizeof(unsigned char));
                    memcpy(cuda_pixel_contrib_depth + storage_offset, &cuda_particles[i].location[2], sizeof(float));
                }
            }
        }
    }

    CUDA_CALL(cudaMemcpy(d_pixel_index, cuda_pixel_index, (cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_pixel_contrib_colours, cuda_pixel_contrib_colours, cuda_pixel_contrib_count * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_pixel_contrib_depth, cuda_pixel_contrib_depth, cuda_pixel_contrib_count * sizeof(float), cudaMemcpyHostToDevice));

    // Calculate the grid and block dimensions
    dim3 threadsPerBlock(32, 1, 1);
    dim3 blocksPerGrid((cuda_output_image_width * cuda_output_image_height + threadsPerBlock.x - 1) / threadsPerBlock.x);

    cuda_stage2_last_parallel<<<blocksPerGrid, threadsPerBlock>>>(d_pixel_contrib_depth, d_pixel_contrib_colours, d_pixel_index);

    CUDA_CALL(cudaGetLastError())
	CUDA_CALL(cudaDeviceSynchronize())

    // TODO: Delete if Stage 2 Changes
    // Copies to device memory, as it is used by Stage 3
    CUDA_CALL(cudaMemcpy(d_pixel_index, cuda_pixel_index, (cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_pixel_contrib_colours, cuda_pixel_contrib_colours, cuda_pixel_contrib_count * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    */

    serial_stage2();


#ifdef VALIDATION
    validate_pixel_index(cuda_pixel_contribs, cuda_pixel_index, cuda_output_image_width, cuda_output_image_height);
    validate_sorted_pairs(cuda_particles, cuda_particles_count, cuda_pixel_index, cuda_output_image_width, cuda_output_image_height, cuda_pixel_contrib_colours, cuda_pixel_contrib_depth);
#endif    
}

void cuda_stage3(){
    // Memset output image data to 255 (white) 
    memset(cuda_output_image.data, 255, cuda_output_image.width * cuda_output_image.height * cuda_output_image.channels * sizeof(unsigned char));
    CUDA_CALL(cudaMemcpy(d_output_image_data, cuda_output_image.data, cuda_output_image.width * cuda_output_image.height * cuda_output_image.channels * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Calculate the grid and block dimensions
    dim3 threadsPerBlock(32, 1, 1);
    dim3 blocksPerGrid((cuda_output_image.width * cuda_output_image.height + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // Launch the CUDA kernel (using best performing implementation)
	cuda_stage3_order_rearranged_outer_parallel <<<blocksPerGrid, threadsPerBlock>>> (d_pixel_index, d_pixel_contrib_colours, d_output_image_data);

	// Another kernel, which performs worse. See the comment for cuda_stage3_outer_parallel.
	// cuda_stage3_outer_parallel << <blocksPerGrid, threadsPerBlock >> > (d_pixel_index, d_pixel_contrib_colours, d_output_image_data);

    CUDA_CALL(cudaGetLastError())
    CUDA_CALL(cudaDeviceSynchronize())

#ifdef VALIDATION
    CUDA_CALL(cudaMemcpy(cuda_pixel_contrib_colours, d_pixel_contrib_colours, cuda_pixel_contrib_count * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(cuda_pixel_index, d_pixel_index, ((cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int)), cudaMemcpyDeviceToHost))
	CUDA_CALL(cudaMemcpy(cuda_output_image.data, d_output_image_data, cuda_output_image_width * cuda_output_image_height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost))
    validate_blend(cuda_pixel_index, cuda_pixel_contrib_colours, &cuda_output_image);
#endif    
}

void cuda_end(CImage *output_image) {
    // TODO: Check if matches cuda_begin BEFORE DOING EXPERIMENTS

    output_image->width = cuda_output_image.width;
    output_image->height = cuda_output_image.height;
    output_image->channels = cuda_output_image.channels;
    CUDA_CALL(cudaMemcpy(output_image->data, d_output_image_data, cuda_output_image_width * cuda_output_image_height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // memcpy(output_image->data, cuda_output_image.data, cuda_output_image.width * cuda_output_image.height * cuda_output_image.channels * sizeof(unsigned char));

    CUDA_CALL(cudaFree(d_particles));
    CUDA_CALL(cudaFree(d_pixel_contribs));
    CUDA_CALL(cudaFree(d_pixel_index));
    CUDA_CALL(cudaFree(d_pixel_contrib_colours));
    CUDA_CALL(cudaFree(d_pixel_contrib_depth));
    CUDA_CALL(cudaFree(d_output_image_data));

    free(cuda_particles);
    free(cuda_pixel_contribs);
    free(cuda_pixel_index);
    free(cuda_output_image.data);
}