  #include <cstdint>

  #include <torch/extension.h>
  #include <kutacc.h>

  void quant_out(
      const torch::Tensor& input,
      torch::Tensor& output,
      torch::Tensor& scale
  ) {
      TORCH_CHECK(input.dim() == 2, "quant_out: input must be 2D");
      TORCH_CHECK(output.dim() == 2, "quant_out: output must be 2D");
      TORCH_CHECK(scale.dim() == 1, "quant_out: scale must be 1D");

      TORCH_CHECK(input.scalar_type() == torch::kBFloat16,
                  "quant_out: input must be bfloat16");
      TORCH_CHECK(output.scalar_type() == torch::kInt8,
                  "quant_out: output must be int8");
      TORCH_CHECK(scale.scalar_type() == torch::kFloat32,
                  "quant_out: scale must be float32");

      TORCH_CHECK(input.size(0) == output.size(0), "height mismatch");
      TORCH_CHECK(input.size(1) == output.size(1), "width mismatch");
      TORCH_CHECK(scale.size(0) == input.size(0), "scale height mismatch");

      auto input_c = input.contiguous();
      auto output_c = output.contiguous();

      const auto height = input_c.size(0);
      const auto width = input_c.size(1);

      kutacc::quant(
          height,
          width,
          reinterpret_cast<const __bf16*>(input_c.data_ptr<at::BFloat16>()),
          input_c.stride(0),
          output_c.data_ptr<int8_t>(),
          output_c.stride(0),
          scale.data_ptr<float>()
      );

      if (!output.is_contiguous()) {
          output.copy_(output_c);
      }
  }