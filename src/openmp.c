#include "openmp.h"
#include "helper.h"

#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

///
/// Algorithm storage
///
unsigned int openmp_particles_count;
Particle* openmp_particles;
unsigned int* openmp_pixel_contribs;
unsigned int* openmp_pixel_index;
unsigned char* openmp_pixel_contrib_colours;
float* openmp_pixel_contrib_depth;
unsigned int openmp_pixel_contrib_count;
CImage openmp_output_image;

void openmp_sort_pairs(float* keys_start, unsigned char* colours_start, const int first, const int last) {
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
#pragma omp parallel
        {
#pragma omp sections
            {
#pragma omp section
                openmp_sort_pairs(keys_start, colours_start, first, j - 1);
#pragma omp section
                openmp_sort_pairs(keys_start, colours_start, j + 1, last);
            }
        }
    }
}

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

void openmp_begin(const Particle* init_particles, const unsigned int init_particles_count,
    const unsigned int out_image_width, const unsigned int out_image_height) {

    // Allocate a opy of the initial particles, to be used during computation
    openmp_particles_count = init_particles_count;
    openmp_particles = malloc(init_particles_count * sizeof(Particle));
    memcpy(openmp_particles, init_particles, init_particles_count * sizeof(Particle));

    // Allocate a histogram to track how many particles contribute to each pixel
    openmp_pixel_contribs = (unsigned int*)malloc(out_image_width * out_image_height * sizeof(unsigned int));
    // Allocate an index to track where data for each pixel's contributing colour starts/ends
    openmp_pixel_index = (unsigned int*)malloc((out_image_width * out_image_height + 1) * sizeof(unsigned int));
    // Init a buffer to store colours contributing to each pixel into (allocated in stage 2)
    openmp_pixel_contrib_colours = 0;
    // Init a buffer to store depth of colours contributing to each pixel into (allocated in stage 2)
    openmp_pixel_contrib_depth = 0;
    // This tracks the number of contributes the two above buffers are allocated for, init 0
    openmp_pixel_contrib_count = 0;

    // Allocate output image
    openmp_output_image.width = (int)out_image_width;
    openmp_output_image.height = (int)out_image_height;
    openmp_output_image.channels = 3;  // RGB
    openmp_output_image.data = (unsigned char*)malloc(openmp_output_image.width * openmp_output_image.height * openmp_output_image.channels * sizeof(unsigned char));
}

void openmp_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_pixel_contribs(openmp_particles, openmp_particles_count, openmp_pixel_contribs, openmp_output_image.width, openmp_output_image.height);

    // Reset the pixel contributions histogram
    memset(openmp_pixel_contribs, 0, openmp_output_image.width * openmp_output_image.height * sizeof(unsigned int));

    signed int i;
#pragma omp parallel for private(i)
    for (i = 0; i < openmp_particles_count; ++i) {
        // Compute bounding box [inclusive-inclusive]
        int x_min = (int)roundf(openmp_particles[i].location[0] - openmp_particles[i].radius);
        int y_min = (int)roundf(openmp_particles[i].location[1] - openmp_particles[i].radius);
        int x_max = (int)roundf(openmp_particles[i].location[0] + openmp_particles[i].radius);
        int y_max = (int)roundf(openmp_particles[i].location[1] + openmp_particles[i].radius);
        // Clamp bounding box to image bounds
        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= openmp_output_image.width ? openmp_output_image.width - 1 : x_max;
        y_max = y_max >= openmp_output_image.height ? openmp_output_image.height - 1 : y_max;
        // For each pixel in the bounding box, check that it falls within the radius
        int x, y;
#pragma omp parallel for private (x, y) collapse(2)
        for (x = x_min; x <= x_max; ++x) {
            for (y = y_min; y <= y_max; ++y) {
                const float x_ab = (float)x + 0.5f - openmp_particles[i].location[0];
                const float y_ab = (float)y + 0.5f - openmp_particles[i].location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= openmp_particles[i].radius) {
                    const unsigned int pixel_offset = y * openmp_output_image.width + x;
#pragma omp atomic
                    ++openmp_pixel_contribs[pixel_offset];
                }
            }
        }
    }

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    validate_pixel_contribs(openmp_particles, openmp_particles_count, openmp_pixel_contribs, openmp_output_image.width, openmp_output_image.height);
#endif
}

void openmp_stage2() {
    // Optionally during development call the skip function/s with the correct inputs to skip this stage
    // skip_pixel_index(openmp_pixel_contribs, openmp_pixel_index, openmp_output_image.width, openmp_output_image.height);
    // skip_sorted_pairs(openmp_particles, openmp_particles_count, openmp_pixel_index, openmp_output_image.width, openmp_output_image.height, openmp_pixel_contrib_colours, openmp_pixel_contrib_depth);


    // Exclusive prefix sum across the histogram to create an index
    openmp_pixel_index[0] = 0;
    for (int i = 0; i < openmp_output_image.width * openmp_output_image.height; ++i) {
        openmp_pixel_index[i + 1] = openmp_pixel_index[i] + openmp_pixel_contribs[i];
    }


    // Recover the total from the index
    const unsigned int TOTAL_CONTRIBS = openmp_pixel_index[openmp_output_image.width * openmp_output_image.height];
    if (TOTAL_CONTRIBS > openmp_pixel_contrib_count) {
        // (Re)Allocate colour storage
        if (openmp_pixel_contrib_colours) free(openmp_pixel_contrib_colours);
        if (openmp_pixel_contrib_depth) free(openmp_pixel_contrib_depth);
        openmp_pixel_contrib_colours = (unsigned char*)malloc(TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
        openmp_pixel_contrib_depth = (float*)malloc(TOTAL_CONTRIBS * sizeof(float));
        openmp_pixel_contrib_count = TOTAL_CONTRIBS;
    }

    // Reset the pixel contributions histogram
    memset(openmp_pixel_contribs, 0, openmp_output_image.width * openmp_output_image.height * sizeof(unsigned int));

    // Store colours according to index
    // For each particle, store a copy of the colour/depth in openmp_pixel_contribs for each contributed pixel
    for (int i = 0; i < openmp_particles_count; ++i) {
        // Compute bounding box [inclusive-inclusive]
        int x_min = (int)roundf(openmp_particles[i].location[0] - openmp_particles[i].radius);
        int y_min = (int)roundf(openmp_particles[i].location[1] - openmp_particles[i].radius);
        int x_max = (int)roundf(openmp_particles[i].location[0] + openmp_particles[i].radius);
        int y_max = (int)roundf(openmp_particles[i].location[1] + openmp_particles[i].radius);
        // Clamp bounding box to image bounds
        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= openmp_output_image.width ? openmp_output_image.width - 1 : x_max;
        y_max = y_max >= openmp_output_image.height ? openmp_output_image.height - 1 : y_max;
        // Store data for every pixel within the bounding box that falls within the radius
        int x;
#pragma omp parallel for private(x) collapse(2)
        for (x = x_min; x <= x_max; ++x) {
            for (int y = y_min; y <= y_max; ++y) {
                const float x_ab = (float)x + 0.5f - openmp_particles[i].location[0];
                const float y_ab = (float)y + 0.5f - openmp_particles[i].location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= openmp_particles[i].radius) {
                    const unsigned int pixel_offset = y * openmp_output_image.width + x;
                    // Offset into openmp_pixel_contrib buffers is index + histogram
                    // Increment openmp_pixel_contribs, so next contributor stores to correct offset
                    const unsigned int storage_offset = openmp_pixel_index[pixel_offset] + (openmp_pixel_contribs[pixel_offset]++);
                    // Copy data to openmp_pixel_contrib buffers
                    memcpy(openmp_pixel_contrib_colours + (4 * storage_offset), openmp_particles[i].color, 4 * sizeof(unsigned char));
                    memcpy(openmp_pixel_contrib_depth + storage_offset, &openmp_particles[i].location[2], sizeof(float));
                }
            }
        }
    }

    int i;
#pragma omp parallel for private (i)
    // Pair sort the colours contributing to each pixel based on ascending depth
    for (i = 0; i < openmp_output_image.width * openmp_output_image.height; ++i) {
        // Pair sort the colours which contribute to a single pigment
        regular_sort_pairs(
            openmp_pixel_contrib_depth,
            openmp_pixel_contrib_colours,
            openmp_pixel_index[i],
            openmp_pixel_index[i + 1] - 1
        );
    }

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // Note: Only validate_equalised_histogram() MUST be uncommented, the others are optional
    validate_pixel_index(openmp_pixel_contribs, openmp_pixel_index, openmp_output_image.width, openmp_output_image.height);
    validate_sorted_pairs(openmp_particles, openmp_particles_count, openmp_pixel_index, openmp_output_image.width, openmp_output_image.height, openmp_pixel_contrib_colours, openmp_pixel_contrib_depth);
#endif    
}

void openmp_stage3() {
    // Memset output image data to 255 (white)
    memset(openmp_output_image.data, 255, openmp_output_image.width * openmp_output_image.height * openmp_output_image.channels * sizeof(unsigned char));

    // Order dependent blending into output image
    int i;
#pragma omp parallel for private (i)
    for (i = 0; i < openmp_output_image.width * openmp_output_image.height; ++i) {
    	for (int j = openmp_pixel_index[i]; j < openmp_pixel_index[i + 1]; ++j) {
            // Get the color and opacity values
            const unsigned char* color = &openmp_pixel_contrib_colours[j * 4];
            const float opacity = (float)color[3] / (float)255;

            // Compute the blended color channels
            const float src_r = (float)color[0] * opacity;
            const float src_g = (float)color[1] * opacity;
            const float src_b = (float)color[2] * opacity;
            const float dest_r = openmp_output_image.data[(i * 3) + 0];
            const float dest_g = openmp_output_image.data[(i * 3) + 1];
            const float dest_b = openmp_output_image.data[(i * 3) + 2];
            const float blended_r = src_r + dest_r * (1 - opacity);
            const float blended_g = src_g + dest_g * (1 - opacity);
            const float blended_b = src_b + dest_b * (1 - opacity);

            // Store the blended color channels in the output image
            openmp_output_image.data[(i * 3) + 0] = (unsigned char)blended_r;
            openmp_output_image.data[(i * 3) + 1] = (unsigned char)blended_g;
            openmp_output_image.data[(i * 3) + 2] = (unsigned char)blended_b;
        }
    }

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    validate_blend(openmp_pixel_index, openmp_pixel_contrib_colours, &openmp_output_image);
#endif    
}

void openmp_end(CImage* output_image) {
    // Store return value
    output_image->width = openmp_output_image.width;
    output_image->height = openmp_output_image.height;
    output_image->channels = openmp_output_image.channels;
    memcpy(output_image->data, openmp_output_image.data, openmp_output_image.width * openmp_output_image.height * openmp_output_image.channels * sizeof(unsigned char));
    // Release allocations
    free(openmp_pixel_contrib_depth);
    free(openmp_pixel_contrib_colours);
    free(openmp_output_image.data);
    free(openmp_pixel_index);
    free(openmp_pixel_contribs);
    free(openmp_particles);
    // Return ptrs to nullptr
    openmp_pixel_contrib_depth = 0;
    openmp_pixel_contrib_colours = 0;
    openmp_output_image.data = 0;
    openmp_pixel_index = 0;
    openmp_pixel_contribs = 0;
    openmp_particles = 0;
}