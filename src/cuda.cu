#include "cuda.cuh"
#include "helper.h"
#include "cuda_runtime.h"

#include <cstring>
#include <cmath>
#include <device_launch_parameters.h>

unsigned int cuda_pixel_contrib_count;                      // The number of contributors d_pixel_contrib_colours and d_pixel_contrib_depth have been allocated for
unsigned int cuda_particles_count;                          // Number of particles in d_particles

///
/// Host Variables
///
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


void sort_pairs(float* keys_start, unsigned char* colours_start, const int first, const int last) {
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
        sort_pairs(keys_start, colours_start, first, j - 1);
        sort_pairs(keys_start, colours_start, j + 1, last);
    }
}

__global__ void stage1_outer_loop_parallelized(Particle* particles, unsigned int* pixel_contribs) {
    // Compute the index for the current thread
    int index = threadIdx.x + blockIdx.x * blockDim.x;

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
__global__ void stage1_outer_loop_parallelized_inner_loops_collapsed(Particle* particles, unsigned int* pixel_contribs) {
    // Compute the index for the current thread
    int index = threadIdx.x + blockIdx.x * blockDim.x;

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

void fake_stage3() {
    // Memset output image data to 255 (white)
    memset(cuda_output_image.data, 255, cuda_output_image.width * cuda_output_image.height * cuda_output_image.channels * sizeof(unsigned char));

    // Order dependent blending into output image
    for (int i = 0; i < cuda_output_image.width * cuda_output_image.height; ++i) {
        for (unsigned int j = cuda_pixel_index[i]; j < cuda_pixel_index[i + 1]; ++j) {
            // Blend each of the red/green/blue colours according to the below blend formula
            // dest = src * opacity + dest * (1 - opacity);
            const float opacity = (float)cuda_pixel_contrib_colours[j * 4 + 3] / (float)255;
            cuda_output_image.data[(i * 3) + 0] = (unsigned char)((float)cuda_pixel_contrib_colours[j * 4 + 0] * opacity + (float)cuda_output_image.data[(i * 3) + 0] * (1 - opacity));
            cuda_output_image.data[(i * 3) + 1] = (unsigned char)((float)cuda_pixel_contrib_colours[j * 4 + 1] * opacity + (float)cuda_output_image.data[(i * 3) + 1] * (1 - opacity));
            cuda_output_image.data[(i * 3) + 2] = (unsigned char)((float)cuda_pixel_contrib_colours[j * 4 + 2] * opacity + (float)cuda_output_image.data[(i * 3) + 2] * (1 - opacity));
            // cuda_pixel_contrib_colours is RGBA
            // cuda_output_image.data is RGB (final output image does not have an alpha channel!)
        }
    }
}

void cuda_begin(const Particle* init_particles, const unsigned int init_particles_count,
    const unsigned int out_image_width, const unsigned int out_image_height) {

    cuda_particles_count = init_particles_count;
    cuda_pixel_contrib_count = 0;

    cuda_particles = (Particle*)malloc(sizeof(Particle) * cuda_particles_count);
    cuda_pixel_contribs = (unsigned int*)malloc(out_image_width * out_image_height * sizeof(unsigned int));
    cuda_pixel_index = (unsigned int*)malloc((out_image_width * out_image_height + 1) * sizeof(unsigned int));

    cuda_output_image.data = (unsigned char*)malloc(cuda_output_image_width * cuda_output_image_height * 3 * sizeof(unsigned char));
    cuda_output_image.height = cuda_output_image_height;
    cuda_output_image.width = cuda_output_image_width;
    cuda_output_image.channels = 3;

    cuda_output_image_width = (int)out_image_width;
    cuda_output_image_height = (int)out_image_height;

    CUDA_CALL(cudaMalloc(&d_particles, init_particles_count * sizeof(Particle)));
    CUDA_CALL(cudaMemcpy(d_particles, init_particles, init_particles_count * sizeof(Particle), cudaMemcpyHostToDevice));

	CUDA_CALL(cudaMalloc(&d_pixel_contribs, ((int)out_image_width) * ((int)out_image_height) * sizeof(unsigned int)));
    CUDA_CALL(cudaMalloc(&d_pixel_index, (((int)out_image_width) * ((int)out_image_height) + 1) * sizeof(unsigned int)));
    d_pixel_contrib_colours = 0;
    d_pixel_contrib_depth = 0;

    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_WIDTH, &cuda_output_image_width, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_HEIGHT, &cuda_output_image_height, sizeof(int)));
    CUDA_CALL(cudaMalloc(&d_output_image_data, cuda_output_image_width * cuda_output_image_height * 3 * sizeof(unsigned char)));
}

void cuda_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
	// You will need to copy the data back to host before passing to these functions
	// skip_pixel_contribs(cuda_particles, cuda_particles_count, cuda_pixel_contribs, cuda_output_image_width, cuda_output_image_height);

    // Reset the pixel contributions histogram on the device
    cudaMemset(d_pixel_contribs, 0, cuda_output_image.width * cuda_output_image.height * sizeof(unsigned int));

    // Calculate the grid and block dimensions
	dim3 threadsPerBlock(32, 1, 1);
    dim3 blocksPerGrid((int)ceil((float)cuda_particles_count/threadsPerBlock.x));

    // Launch the CUDA kernel
	stage1_outer_loop_parallelized_inner_loops_collapsed <<<blocksPerGrid, threadsPerBlock>>> (d_particles, d_pixel_contribs);
	// stage1_outer_loop_parallelized << <blocksPerGrid, threadsPerBlock >> > (d_particles, d_pixel_contribs);

#ifdef VALIDATION
    CUDA_CALL(cudaMemcpy(cuda_pixel_contribs, d_pixel_contribs, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost))
    validate_pixel_contribs(cuda_particles, cuda_particles_count, cuda_pixel_contribs, cuda_output_image_width, cuda_output_image_height);
#endif
}

void cuda_stage2() {
    // CUDA_CALL(cudaMemcpy(cuda_pixel_contribs, d_pixel_contribs, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost))
	// skip_pixel_index(cuda_pixel_contribs, cuda_pixel_index, cuda_output_image_width, cuda_output_image_height);
	// skip_sorted_pairs(cuda_particles, cuda_particles_count, cuda_pixel_index, cuda_output_image_width, cuda_output_image_height, cuda_pixel_contrib_colours, cuda_pixel_contrib_depth);



	// Exclusive prefix sum across the histogram to create an index
    cuda_pixel_index[0] = 0;
    for (int i = 0; i < cuda_output_image.width * cuda_output_image.height; ++i) {
        cuda_pixel_index[i + 1] = cuda_pixel_index[i] + cuda_pixel_contribs[i];
    }
    // Transfering variables changed on the host to device
    CUDA_CALL(cudaMemcpy(d_pixel_index, cuda_pixel_index, ((cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int)), cudaMemcpyHostToDevice))

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

        CUDA_CALL(cudaMalloc(&d_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char)))
        CUDA_CALL(cudaMalloc(&d_pixel_contrib_depth, TOTAL_CONTRIBS * sizeof(float)))
    }

    // Reset the pixel contributions histogram on the device
    cudaMemset(d_pixel_contribs, 0, cuda_output_image.width * cuda_output_image.height * sizeof(unsigned int));


    // Calculate the grid and block dimensions
    dim3 threadsPerBlock(32, 1, 1);
    dim3 blocksPerGrid((int)ceil((float)cuda_particles_count / threadsPerBlock.x));

    // Launch the CUDA kernel
    ///stage2_outer_loop_paralelized <<<blocksPerGrid, threadsPerBlock>>> (d_pixel_contrib_depth, d_pixel_contrib_colours, d_pixel_index, d_particles);

    cudaDeviceSynchronize();

    // TODO: Copy the variables that changed back into host memory
    // TODO: Create a kernel (stage2_outer_loop_paralelized) from Store colours loop according to index loop and delete the bellow from this function

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
        sort_pairs(
            cuda_pixel_contrib_depth,
            cuda_pixel_contrib_colours,
            cuda_pixel_index[i],
            cuda_pixel_index[i + 1] - 1
        );
    }


    // TODO: Copy the sorted pairs into device memory (?)

    // validate_pixel_index(cuda_pixel_contribs, cuda_pixel_index, cuda_output_image_width, cuda_output_image_height);
    // validate_sorted_pairs(cuda_particles, cuda_particles_count, cuda_pixel_index, cuda_output_image_width, cuda_output_image_height, cuda_pixel_contrib_colours, cuda_pixel_contrib_depth);

#ifdef VALIDATION
    // Note: Only validate_equalised_histogram() MUST be uncommented, the others are optional
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)

	validate_pixel_index(cuda_pixel_contribs, cuda_pixel_index, cuda_output_image_width, cuda_output_image_height);
    validate_sorted_pairs(cuda_particles, cuda_particles_count, cuda_pixel_index, cuda_output_image_width, cuda_output_image_height, cuda_pixel_contrib_colours, cuda_pixel_contrib_depth);
#endif    
}

void cuda_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // You will need to copy the data back to host before passing to these functions
    // skip_blend(cuda_pixel_index, cuda_pixel_contrib_colours, &cuda_output_image);
    
    fake_stage3();
    // validate_blend(cuda_pixel_index, cuda_pixel_contrib_colours, &cuda_output_image);

#ifdef VALIDATION
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    validate_blend(cuda_pixel_index, cuda_pixel_contrib_colours, &cuda_output_image);
#endif    
}

void cuda_end(CImage *output_image) {
    output_image->width = cuda_output_image.width;
    output_image->height = cuda_output_image.height;
    output_image->channels = cuda_output_image.channels;
    // CUDA_CALL(cudaMemcpy(output_image->data, d_output_image_data, cuda_output_image_width * cuda_output_image_height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    memcpy(output_image->data, cuda_output_image.data, cuda_output_image.width * cuda_output_image.height * cuda_output_image.channels * sizeof(unsigned char));

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