#pragma once

#include <cstddef>
#include <cstdint>

struct AsmlMetal;
struct AsmlMetalMatrix {
    size_t matrix_offset;
    size_t row_stride;
    uint32_t rows;
    uint32_t output_offset;
};

#ifdef __cplusplus
extern "C" {
#endif
AsmlMetal* asml_metal_create(const void* model, size_t model_size, const char* library_path,
                             char* error, size_t error_capacity);
void asml_metal_destroy(AsmlMetal* metal);
bool asml_metal_matvec(AsmlMetal* metal, size_t matrix_offset, uint32_t rows, uint32_t cols,
                       size_t row_stride, uint32_t scale_bytes, const int8_t* input_q,
                       const float* input_scales, float* output, char* error,
                       size_t error_capacity);
bool asml_metal_matvec_batch(AsmlMetal* metal, const AsmlMetalMatrix* matrices,
                             uint32_t matrix_count, uint32_t cols, uint32_t scale_bytes,
                             const int8_t* input_q, const float* input_scales, float* output,
                             char* error, size_t error_capacity);
#ifdef __cplusplus
}
#endif
