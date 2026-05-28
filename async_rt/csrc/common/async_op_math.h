#include <arm_sve.h>
#include <limits>

static const uint32_t svexp_f32_coeff[] = {
    0x3f7ffff6, // x^1: 0x1.ffffecp-1f
    0x3efffedb, // x^2: 0x1.fffdb6p-2f
    0x3e2aaf33, // x^3: 0x1.555e66p-3f
    0x3d2b9f17, // x^4: 0x1.573e2ep-5f
    0x3c072010, // x^5: 0x1.0e4020p-7f
};

static inline void bf16_load_to_2xfp32(
    svbool_t pred16, svbool_t pred_lo, svbool_t pred_hi,
    const uint16_t* ptr, std::size_t offset,
    svfloat32_t& out_lo, svfloat32_t& out_hi
) {
    svuint16_t raw = svld1_u16(pred16, ptr + offset);
    svuint32_t lo_u32 = svlsl_n_u32_x(pred_lo, svunpklo_u32(raw), 16);
    svuint32_t hi_u32 = svlsl_n_u32_x(pred_hi, svunpkhi_u32(raw), 16);
    out_lo = svreinterpret_f32_u32(lo_u32);
    out_hi = svreinterpret_f32_u32(hi_u32);
}

static inline void fp32x2_store_as_bf16(
    svbool_t pred16, svbool_t pred_lo, svbool_t pred_hi,
    uint16_t* ptr, std::size_t offset,
    svfloat32_t f32_lo, svfloat32_t f32_hi
) {
    svuint32_t u32_lo = svlsr_n_u32_x(pred_lo, svreinterpret_u32_f32(f32_lo), 16);
    svuint32_t u32_hi = svlsr_n_u32_x(pred_hi, svreinterpret_u32_f32(f32_hi), 16);
    svuint16_t lo_u16 = svreinterpret_u16_u32(u32_lo);
    svuint16_t hi_u16 = svreinterpret_u16_u32(u32_hi);
    svuint16_t packed = svuzp1_u16(lo_u16, hi_u16);
    svst1_u16(pred16, ptr + offset, packed);
}

static inline svfloat32_t svexp_f32_z(svbool_t pg, svfloat32_t x) {
    const auto c1 = svreinterpret_f32_u32(svdup_n_u32(svexp_f32_coeff[0]));
    const auto c2 = svreinterpret_f32_u32(svdup_n_u32(svexp_f32_coeff[1]));
    const auto c3 = svreinterpret_f32_u32(svdup_n_u32(svexp_f32_coeff[2]));
    const auto c4 = svreinterpret_f32_u32(svdup_n_u32(svexp_f32_coeff[3]));
    const auto c5 = svreinterpret_f32_u32(svdup_n_u32(svexp_f32_coeff[4]));

    const auto shift   = svreinterpret_f32_u32(svdup_n_u32(0x4b00007f)); // 2^23 + 127 = 0x1.0000fep23f
    const auto inv_ln2 = svreinterpret_f32_u32(svdup_n_u32(0x3fb8aa3b)); // 1 / ln(2) = 0x1.715476p+0f
    const auto neg_ln2_hi =
        svreinterpret_f32_u32(svdup_n_u32(0xbf317200)); // -ln(2) from bits  -1 to -19: -0x1.62e400p-1f
    const auto neg_ln2_lo =
        svreinterpret_f32_u32(svdup_n_u32(0xb5bfbe8e)); // -ln(2) from bits -20 to -42: -0x1.7f7d1cp-20f

    const auto inf       = svdup_n_f32(std::numeric_limits<float>::infinity());
    const auto max_input = svdup_n_f32(88.37f); // Approximately ln(2^127.5)
    const auto zero      = svdup_n_f32(0.f);
    const auto min_input = svdup_n_f32(-86.64f); // Approximately ln(2^-125)

    // Range reduction:
    //   e^x = 2^n * e^r
    // where:
    //   n = floor(x / ln(2))
    //   r = x - n * ln(2)
    //
    // By adding x / ln(2) with 2^23 + 127 (shift):
    //   * As FP32 fraction part only has 23-bits, the addition of 2^23 + 127 forces decimal part
    //     of x / ln(2) out of the result. The integer part of x / ln(2) (i.e. n) + 127 will occupy
    //     the whole fraction part of z in FP32 format.
    //     Subtracting 2^23 + 127 (shift) from z will result in the integer part of x / ln(2)
    //     (i.e. n) because the decimal part has been pushed out and lost.
    //   * The addition of 127 makes the FP32 fraction part of z ready to be used as the exponent
    //     in FP32 format. Left shifting z by 23 bits will result in 2^n.
    const auto z     = svmla_f32_x(pg, shift, x, inv_ln2);
    const auto n     = svsub_f32_x(pg, z, shift);
    const auto scale = svreinterpret_f32_u32(svlsl_n_u32_z(pg, svreinterpret_u32_f32(z), 23)); // 2^n

    // The calculation of n * ln(2) is done using 2 steps to achieve accuracy beyond FP32.
    // This outperforms longer Taylor series (3-4 tabs) both in term of accuracy and performance.
    const auto r_hi = svmla_f32_x(pg, x, n, neg_ln2_hi);
    const auto r    = svmla_f32_x(pg, r_hi, n, neg_ln2_lo);

    // Compute the truncated Taylor series of e^r.
    //   poly = scale * (1 + c1 * r + c2 * r^2 + c3 * r^3 + c4 * r^4 + c5 * r^5)
    const auto r2 = svmul_f32_x(pg, r, r);

    const auto p1     = svmul_f32_x(pg, c1, r);
    const auto p23    = svmla_f32_x(pg, c2, c3, r);
    const auto p45    = svmla_f32_x(pg, c4, c5, r);
    const auto p2345  = svmla_f32_x(pg, p23, p45, r2);
    const auto p12345 = svmla_f32_x(pg, p1, p2345, r2);

    auto poly = svmla_f32_x(pg, scale, p12345, scale);

    // Handle underflow and overflow.
    poly = svsel_f32(svcmplt_f32(pg, x, min_input), zero, poly);
    poly = svsel_f32(svcmpgt_f32(pg, x, max_input), inf, poly);

    return poly;
}

inline svfloat32_t svtanh_f32_z(svbool_t pg, svfloat32_t val)
{
    const svfloat32_t CONST_1        = svdup_n_f32(1.f);
    const svfloat32_t CONST_2        = svdup_n_f32(2.f);
    const svfloat32_t CONST_MIN_TANH = svdup_n_f32(-10.f);
    const svfloat32_t CONST_MAX_TANH = svdup_n_f32(10.f);

    svfloat32_t x     = svmin_f32_x(pg, svmax_f32_z(pg, val, CONST_MIN_TANH), CONST_MAX_TANH);
    svfloat32_t exp2x = svexp_f32_z(pg, svmul_f32_z(pg, CONST_2, x));
    svfloat32_t num   = svsub_f32_x(pg, exp2x, CONST_1);
    svfloat32_t den   = svadd_f32_x(pg, exp2x, CONST_1);
    svfloat32_t tanh  = svdiv_f32_x(pg, num, den);
    return tanh;
}