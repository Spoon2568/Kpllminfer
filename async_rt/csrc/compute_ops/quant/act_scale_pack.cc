#include <cstdint>
#include <cstring>

#include <torch/extension.h>
#include <parallel.h>

void act_scale_pack_out(
    const torch::Tensor& act,
    const torch::Tensor& scale,
    torch::Tensor& output)
{
    TORCH_CHECK(act.dim() == 2 && scale.dim() == 2, "act_scale_pack: act and scale dim must be two.");
    TORCH_CHECK(scale.size(1) == 1, "act_scale_pack: scale.size(1) != 1");
    TORCH_CHECK(output.dim() == 2, "act_scale_pack: output must be 2D");
    TORCH_CHECK(act.scalar_type() == torch::kInt8, "act_scale_pack: act must be int8");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "act_scale_pack: scale must be float32");
    TORCH_CHECK(output.scalar_type() == torch::kUInt8, "act_scale_pack: output must be uint8");

    int64_t h = act.size(0);
    int64_t w1 = act.size(1);
    int64_t w2 = output.size(1);

    TORCH_CHECK(h == scale.size(0), "act_scale_pack: act and scale height mismatch");
    TORCH_CHECK(w1 + 4 == w2, "act_scale_pack: output width must be act width + 4");

    const int8_t* act_data = act.data_ptr<int8_t>();
    const float* scale_data = scale.data_ptr<float>();
    uint8_t* output_data = output.data_ptr<uint8_t>();

    utils::parallel_for(0, h, 1, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            uint8_t* dst = output_data + i * w2;
            std::memcpy(dst, act_data + i * w1, w1);
            *reinterpret_cast<float*>(dst + w1) = scale_data[i];
        }
    });
}
