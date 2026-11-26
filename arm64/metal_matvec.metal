#include <metal_stdlib>
using namespace metal;

struct MatrixDesc {
    ulong matrix_offset;
    ulong row_stride;
    uint rows;
    uint output_offset;
};

kernel void q8_matvec(
    device const char* weights [[buffer(0)]],
    device const char* input_q [[buffer(1)]],
    device const float* input_scales [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant MatrixDesc* matrices [[buffer(4)]],
    constant uint& matrix_count [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& scale_bytes [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    MatrixDesc matrix;
    bool found = false;
    for (uint i = 0; i < matrix_count; ++i) {
        if (row >= matrices[i].output_offset && row < matrices[i].output_offset + matrices[i].rows) {
            matrix = matrices[i];
            found = true;
            break;
        }
    }
    if (!found) return;

    uint local_row = row - matrix.output_offset;
    device const char* weight_row = weights + matrix.matrix_offset + ulong(local_row) * matrix.row_stride;
    device const uchar* weight_scales = (device const uchar*)(weight_row + cols);
    float partial = 0.0f;
    for (uint col = lane; col < cols; col += 32) {
        uint block = col / 32;
        float ws;
        if (scale_bytes == 2) {
            ushort bits = *(device const ushort*)(weight_scales + block * 2);
            ws = float(as_type<half>(bits));
        } else {
            ws = *(device const float*)(weight_scales + block * 4);
        }
        partial += float(int(weight_row[col])) * float(int(input_q[col])) * ws * input_scales[block];
    }
    float sum = simd_sum(partial);
    if (lane == 0) output[row] = sum;
}
