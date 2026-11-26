#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_backend.h"
#include <cstdio>
#include <cstring>

struct AsmlMetal {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
    id<MTLBuffer> weights;
    id<MTLBuffer> input_q;
    id<MTLBuffer> input_scales;
    id<MTLBuffer> output;
    size_t input_capacity;
    size_t output_capacity;
};

static void set_error(char* out, size_t cap, NSString* message) {
    if (!out || cap == 0) return;
    const char* utf8 = message ? message.UTF8String : "Metal operation failed";
    std::snprintf(out, cap, "%s", utf8 ? utf8 : "Metal operation failed");
}

extern "C" AsmlMetal* asml_metal_create(const void* model, size_t model_size,
                                        const char* source_path, char* error,
                                        size_t error_capacity) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            set_error(error, error_capacity, @"No Metal device is available");
            return nullptr;
        }
        NSError* ns_error = nil;
        NSString* path = [NSString stringWithUTF8String:source_path];
        NSString* source = [NSString stringWithContentsOfFile:path
                                                      encoding:NSUTF8StringEncoding
                                                         error:&ns_error];
        if (!source) {
            set_error(error, error_capacity, ns_error.localizedDescription);
            return nullptr;
        }
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&ns_error];
        if (!library) {
            set_error(error, error_capacity, ns_error.localizedDescription);
            return nullptr;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"q8_matvec"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&ns_error];
        if (!pipeline) {
            set_error(error, error_capacity, ns_error.localizedDescription);
            return nullptr;
        }
        id<MTLBuffer> weights = [device newBufferWithBytes:model length:model_size options:MTLResourceStorageModeShared];
        if (!weights) {
            set_error(error, error_capacity, @"Could not allocate the Metal model buffer");
            return nullptr;
        }
        AsmlMetal* metal = new AsmlMetal;
        metal->device = device;
        metal->queue = [device newCommandQueue];
        metal->pipeline = pipeline;
        metal->weights = weights;
        metal->input_capacity = 0;
        metal->output_capacity = 0;
        if (!metal->queue) {
            set_error(error, error_capacity, @"Could not create a Metal command queue");
            delete metal;
            return nullptr;
        }
        return metal;
    }
}

extern "C" void asml_metal_destroy(AsmlMetal* metal) { delete metal; }

extern "C" bool asml_metal_matvec(AsmlMetal* metal, size_t matrix_offset, uint32_t rows,
                                   uint32_t cols, size_t row_stride, uint32_t scale_bytes,
                                   const int8_t* input_q, const float* input_scales, float* output,
                                   char* error, size_t error_capacity) {
    AsmlMetalMatrix matrix{matrix_offset, row_stride, rows, 0};
    return asml_metal_matvec_batch(metal, &matrix, 1, cols, scale_bytes, input_q,
                                   input_scales, output, error, error_capacity);
}

extern "C" bool asml_metal_matvec_batch(AsmlMetal* metal, const AsmlMetalMatrix* matrices,
                                        uint32_t matrix_count, uint32_t cols,
                                        uint32_t scale_bytes, const int8_t* input_q,
                                        const float* input_scales, float* output,
                                        char* error, size_t error_capacity) {
    @autoreleasepool {
        if (!metal || !matrices || matrix_count == 0 || cols == 0 || cols % 32 != 0 ||
            (scale_bytes != 2 && scale_bytes != 4)) {
            set_error(error, error_capacity, @"Invalid Metal matvec arguments");
            return false;
        }
        uint32_t total_rows = 0;
        for (uint32_t i = 0; i < matrix_count; ++i) {
            const auto& matrix = matrices[i];
            if (matrix.rows == 0 || matrix.output_offset != total_rows ||
                matrix.matrix_offset > metal->weights.length ||
                matrix.row_stride > (metal->weights.length - matrix.matrix_offset) / matrix.rows) {
                set_error(error, error_capacity, @"Invalid Metal matrix descriptor");
                return false;
            }
            total_rows += matrix.rows;
        }
        size_t blocks = cols / 32;
        size_t scale_size = blocks * sizeof(float);
        if (metal->input_capacity < cols) {
            metal->input_q = [metal->device newBufferWithLength:cols options:MTLResourceStorageModeShared];
            metal->input_scales = [metal->device newBufferWithLength:scale_size options:MTLResourceStorageModeShared];
            metal->input_capacity = cols;
        }
        if (metal->output_capacity < total_rows) {
            metal->output = [metal->device newBufferWithLength:size_t(total_rows) * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
            metal->output_capacity = total_rows;
        }
        if (!metal->input_q || !metal->input_scales || !metal->output) {
            set_error(error, error_capacity, @"Could not allocate Metal matvec buffers");
            return false;
        }
        std::memcpy(metal->input_q.contents, input_q, cols);
        std::memcpy(metal->input_scales.contents, input_scales, scale_size);

        id<MTLCommandBuffer> command = [metal->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:metal->pipeline];
        [encoder setBuffer:metal->weights offset:0 atIndex:0];
        [encoder setBuffer:metal->input_q offset:0 atIndex:1];
        [encoder setBuffer:metal->input_scales offset:0 atIndex:2];
        [encoder setBuffer:metal->output offset:0 atIndex:3];
        [encoder setBytes:matrices length:sizeof(*matrices) * matrix_count atIndex:4];
        [encoder setBytes:&matrix_count length:sizeof(matrix_count) atIndex:5];
        [encoder setBytes:&cols length:sizeof(cols) atIndex:6];
        [encoder setBytes:&scale_bytes length:sizeof(scale_bytes) atIndex:7];
        [encoder dispatchThreadgroups:MTLSizeMake(total_rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            set_error(error, error_capacity, command.error.localizedDescription);
            return false;
        }
        std::memcpy(output, metal->output.contents, size_t(total_rows) * sizeof(float));
        return true;
    }
}
