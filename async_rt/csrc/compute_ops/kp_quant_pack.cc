#include <cstdint>
#include <torch/extension.h>
#include <kutacc.h>

void quant_back_out(
    const torch::Tensor &input,
    torch::Tensor &output,
    torch::Tensor &scale)
{
    // 类型检查保持不变...
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be bfloat16");
    TORCH_CHECK(output.scalar_type() == torch::kInt8,    "output must be int8");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32,  "scale must be float32");
    TORCH_CHECK(input.size(0) == output.size(0), "height mismatch");
    TORCH_CHECK(input.size(1) == output.size(1), "width mismatch");
    TORCH_CHECK(scale.size(0) == input.size(0), "scale height mismatch");

    auto input_c = input.contiguous();
    auto output_c = output.contiguous();

    const auto num_tokens = input_c.size(0);   // 行数
    const auto hidden_size = input_c.size(1);  // 列数

    // 假设 kutacc::quant_pack 签名: (bf16* src, int8* dst, float* scale, int rows, int cols)
    kutacc::quant_pack(
        reinterpret_cast<__bf16*>(input_c.data_ptr<at::BFloat16>()),
        output_c.data_ptr<int8_t>(),
        scale.data_ptr<float>(),
        hidden_size,    // 行数
        num_tokens);  // 列数
}

// void quant_pack_impl(const utils::Tensor &input, const utils::Tensor &output, const utils::Tensor &scale)
// {
//     bfloat16_t *input_data = input.data_ptr<bfloat16_t>();
//     int8_t *output_data = output.data_ptr<int8_t>();
//     float *scale_data = scale.data_ptr<float>();
//     int hidden_size = input.size(-1);
//     int num_tokens = input.numel() / hidden_size;
//     kutacc::quant_pack(input_data, output_data, scale_data, hidden_size, num_tokens);
// }
