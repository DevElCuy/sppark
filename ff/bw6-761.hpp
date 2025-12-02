// BW6-761 field parameters for sppark
// Base field: 761 bits (12 x 64-bit limbs)
// Scalar field: 377 bits (6 x 64-bit limbs, same as BLS12-377 base field)

#ifndef __SPPARK_FF_BW6_761_HPP__
#define __SPPARK_FF_BW6_761_HPP__

#include <cstdint>
#if defined(__CUDACC__) || defined(__HIPCC__)

namespace device {
#define TO_CUDA_T(limb64) (uint32_t)(limb64), (uint32_t)(limb64>>32)

    // Base field Fq (761 bits)
    static __device__ __constant__ __align__(16) const uint32_t BW6_761_P[24] = {
        TO_CUDA_T(0xf49d00000000008b), TO_CUDA_T(0xe6913e6870000082),
        TO_CUDA_T(0x160cf8aeeaf0a437), TO_CUDA_T(0x98a116c25667a8f8),
        TO_CUDA_T(0x71dcd3dc73ebff2e), TO_CUDA_T(0x8689c8ed12f9fd90),
        TO_CUDA_T(0x03cebaff25b42304), TO_CUDA_T(0x707ba638e584e919),
        TO_CUDA_T(0x528275ef8087be41), TO_CUDA_T(0xb926186a81d14688),
        TO_CUDA_T(0xd187c94004faff3e), TO_CUDA_T(0x0122e824fb83ce0a)
    };

    static __device__ __constant__ __align__(16) const uint32_t BW6_761_RR[24] = { /* (1<<1536)%P */
        TO_CUDA_T(0xc686392d2d1fa659), TO_CUDA_T(0x7b14c9b2f79484ab),
        TO_CUDA_T(0x7fa1e825c1d2b459), TO_CUDA_T(0xd6ec28f848329d88),
        TO_CUDA_T(0x4afb427b73a1ed40), TO_CUDA_T(0x972c69400d5930ae),
        TO_CUDA_T(0x2c7a26bf8c995976), TO_CUDA_T(0xac52e458c6e57af9),
        TO_CUDA_T(0xac731bfa0c536dfe), TO_CUDA_T(0x121e5c630b103f50),
        TO_CUDA_T(0x8f1b0953b886cda4), TO_CUDA_T(0x00ad253c2da8d807)
    };

    static __device__ __constant__ __align__(16) const uint32_t BW6_761_one[24] = { /* (1<<768)%P */
        TO_CUDA_T(0x0202ffffffff85d5), TO_CUDA_T(0x5a5826358fff8ce7),
        TO_CUDA_T(0x9e996e43827faade), TO_CUDA_T(0xda6aff320ee47df4),
        TO_CUDA_T(0xece9cb3e1d94b80b), TO_CUDA_T(0xc0e667a25248240b),
        TO_CUDA_T(0xa74da5bfdcad3905), TO_CUDA_T(0x2352e7fe462f2103),
        TO_CUDA_T(0x7b56588008b1c87c), TO_CUDA_T(0x45848a63e711022f),
        TO_CUDA_T(0xd7a81ebb9f65a9df), TO_CUDA_T(0x0051f77ef127e87d)
    };

    static __device__ __constant__ __align__(16) const uint32_t BW6_761_Px7[24] = { /* P << 7 (left-aligned) */
        TO_CUDA_T(0x4e80000000004580), TO_CUDA_T(0x489f34380000417a),
        TO_CUDA_T(0x067c577578521bf3), TO_CUDA_T(0x508b612b33d47c0b),
        TO_CUDA_T(0xee69ee39f5ff974c), TO_CUDA_T(0x44e476897cfec838),
        TO_CUDA_T(0xe75d7f92da118243), TO_CUDA_T(0x3dd31c72c2748c81),
        TO_CUDA_T(0x413af7c043df20b8), TO_CUDA_T(0x930c3540e8a34429),
        TO_CUDA_T(0xc3e4a0027d7f9f5c), TO_CUDA_T(0x9174127dc1e70568)
    };

    static __device__ __constant__ /*const*/ uint32_t BW6_761_M0 = 0x8fa798dd;

    // Scalar field Fr (377 bits - same as BLS12-377 base field)
    static __device__ __constant__ __align__(16) const uint32_t BW6_761_r[12] = {
        TO_CUDA_T(0x8508c00000000001), TO_CUDA_T(0x170b5d4430000000),
        TO_CUDA_T(0x1ef3622fba094800), TO_CUDA_T(0x1a22d9f300f5138f),
        TO_CUDA_T(0xc63b05c06ca1493b), TO_CUDA_T(0x01ae3a4617c510ea)
    };

    static __device__ __constant__ __align__(16) const uint32_t BW6_761_rRR[12] = { /* (1<<768)%r */
        TO_CUDA_T(0xb786686c9400cd22), TO_CUDA_T(0x0329fcaab00431b1),
        TO_CUDA_T(0x22a5f11162d6b46d), TO_CUDA_T(0xbfdf7d03827dc3ac),
        TO_CUDA_T(0x837e92f041790bf9), TO_CUDA_T(0x006dfccb1e914b88)
    };

    static __device__ __constant__ __align__(16) const uint32_t BW6_761_rone[12] = { /* (1<<384)%r */
        TO_CUDA_T(0x02cdffffffffff68), TO_CUDA_T(0x51409f837fffffb1),
        TO_CUDA_T(0x9f7db3a98a7d3ff2), TO_CUDA_T(0x7b4e97b76e7c6305),
        TO_CUDA_T(0x4cf495bf803c84e8), TO_CUDA_T(0x008d6661e2fdf49a)
    };

    static __device__ __constant__ __align__(16) const uint32_t BW6_761_rx7[12] = { /* r << 7 (left-aligned) */
        TO_CUDA_T(0x8460000000000080), TO_CUDA_T(0x85aea21800000042),
        TO_CUDA_T(0x79b117dd04a4000b), TO_CUDA_T(0x116cf9807a89c78f),
        TO_CUDA_T(0x1d82e03650a49d8d), TO_CUDA_T(0xd71d230be2887563)
    };

    static __device__ __constant__ /*const*/ uint32_t BW6_761_m0 = 0xffffffff;
}

# if defined(__CUDA_ARCH__) || defined(__HIPCC__)   // CUDA compiler
#  if defined(__CUDACC__)
#   include "mont_t.cuh"
#   include "field_large.cuh"  // Simplified field for large (761-bit) fields
#  elif defined(__HIPCC__)
#   include "mont_t.hip"
typedef uint64_t vec256[4];
#  endif

namespace bw6_761 {

// Base field (761 bits) - use field_large_t to avoid template instantiation issues
typedef field_large_t<761, device::BW6_761_P, 0x8fa798dd,  // M0 value directly
                      device::BW6_761_RR, device::BW6_761_one,
                      device::BW6_761_Px7> fp_large;

struct fp_t : public fp_large {
    using mem_t = fp_t;
    using fp_large::bit_length;
    using fp_large::zero;
    using fp_large::degree;
    using fp_large::csel;

    __device__ __forceinline__ fp_t() : fp_large() {}
    __device__ __forceinline__ fp_t(const fp_large& a) : fp_large(a) {}
    __device__ __forceinline__ fp_t(uint32_t v) : fp_large(v) {}

    // Expose overloaded methods
    static __host__ __device__ __forceinline__ fp_t zero() {
        return fp_large::zero();
    }

    static __host__ __device__ __forceinline__ fp_t one() {
        return fp_large::one();
    }

    static __host__ __device__ __forceinline__ fp_t one(bool set_z) {
        return fp_large::one(set_z);
    }

    // Ensure inf/infinity support for point arithmetic
    __device__ __forceinline__ void inf() {
        for (unsigned i = 0; i < fp_large::NLIMBS; i++)
            this->val[i] = 0;
    }
};

// Scalar field (377 bits) - use mont_t (works fine for smaller fields)
typedef mont_t<377, device::BW6_761_r, device::BW6_761_m0,
                    device::BW6_761_rRR, device::BW6_761_rone,
                    device::BW6_761_rx7> fr_mont;
struct fr_t : public fr_mont {
    using mem_t = fr_t;
    using fr_mont::bit_length;
    using fr_mont::degree;
    using fr_mont::csel;
    __device__ __forceinline__ fr_t() {}
    __device__ __forceinline__ fr_t(const fr_mont& a) : fr_mont(a) {}
    template<typename... Ts> constexpr fr_t(Ts... a)  : fr_mont{a...} {}
#  ifdef __HIPCC__
    __host__   __forceinline__ fr_t(vec256 a)         : fr_mont(a) {}
#  endif
};

} // namespace bw6_761

# endif
#endif

#if !defined(__CUDA_ARCH__) && !defined(__HIPCC__)  // host-side field types
# include "field_large.cuh"

# ifndef TO_CUDA_T
#  define TO_CUDA_T(limb64) (uint32_t)(limb64), (uint32_t)((limb64)>>32)
# endif

namespace bw6_761 {
// Host copies of field parameters for field_large_t
inline constexpr uint32_t BW6_761_P_host[24] = {
    TO_CUDA_T(0xf49d00000000008b), TO_CUDA_T(0xe6913e6870000082),
    TO_CUDA_T(0x160cf8aeeaf0a437), TO_CUDA_T(0x98a116c25667a8f8),
    TO_CUDA_T(0x71dcd3dc73ebff2e), TO_CUDA_T(0x8689c8ed12f9fd90),
    TO_CUDA_T(0x03cebaff25b42304), TO_CUDA_T(0x707ba638e584e919),
    TO_CUDA_T(0x528275ef8087be41), TO_CUDA_T(0xb926186a81d14688),
    TO_CUDA_T(0xd187c94004faff3e), TO_CUDA_T(0x0122e824fb83ce0a)
};

inline constexpr uint32_t BW6_761_RR_host[24] = { /* (1<<1536)%P */
    TO_CUDA_T(0xc686392d2d1fa659), TO_CUDA_T(0x7b14c9b2f79484ab),
    TO_CUDA_T(0x7fa1e825c1d2b459), TO_CUDA_T(0xd6ec28f848329d88),
    TO_CUDA_T(0x4afb427b73a1ed40), TO_CUDA_T(0x972c69400d5930ae),
    TO_CUDA_T(0x2c7a26bf8c995976), TO_CUDA_T(0xac52e458c6e57af9),
    TO_CUDA_T(0xac731bfa0c536dfe), TO_CUDA_T(0x121e5c630b103f50),
    TO_CUDA_T(0x8f1b0953b886cda4), TO_CUDA_T(0x00ad253c2da8d807)
};

inline constexpr uint32_t BW6_761_one_host[24] = { /* (1<<768)%P */
    TO_CUDA_T(0x0202ffffffff85d5), TO_CUDA_T(0x5a5826358fff8ce7),
    TO_CUDA_T(0x9e996e43827faade), TO_CUDA_T(0xda6aff320ee47df4),
    TO_CUDA_T(0xece9cb3e1d94b80b), TO_CUDA_T(0xc0e667a25248240b),
    TO_CUDA_T(0xa74da5bfdcad3905), TO_CUDA_T(0x2352e7fe462f2103),
    TO_CUDA_T(0x7b56588008b1c87c), TO_CUDA_T(0x45848a63e711022f),
    TO_CUDA_T(0xd7a81ebb9f65a9df), TO_CUDA_T(0x0051f77ef127e87d)
};

inline constexpr uint32_t BW6_761_Px7_host[24] = { /* P << 7 (left-aligned) */
    TO_CUDA_T(0x4e80000000004580), TO_CUDA_T(0x489f34380000417a),
    TO_CUDA_T(0x067c577578521bf3), TO_CUDA_T(0x508b612b33d47c0b),
    TO_CUDA_T(0xee69ee39f5ff974c), TO_CUDA_T(0x44e476897cfec838),
    TO_CUDA_T(0xe75d7f92da118243), TO_CUDA_T(0x3dd31c72c2748c81),
    TO_CUDA_T(0x413af7c043df20b8), TO_CUDA_T(0x930c3540e8a34429),
    TO_CUDA_T(0xc3e4a0027d7f9f5c), TO_CUDA_T(0x9174127dc1e70568)
};

inline constexpr uint32_t BW6_761_r_host[12] = {
    TO_CUDA_T(0x8508c00000000001), TO_CUDA_T(0x170b5d4430000000),
    TO_CUDA_T(0x1ef3622fba094800), TO_CUDA_T(0x1a22d9f300f5138f),
    TO_CUDA_T(0xc63b05c06ca1493b), TO_CUDA_T(0x01ae3a4617c510ea)
};

inline constexpr uint32_t BW6_761_rRR_host[12] = { /* (1<<768)%r */
    TO_CUDA_T(0xb786686c9400cd22), TO_CUDA_T(0x0329fcaab00431b1),
    TO_CUDA_T(0x22a5f11162d6b46d), TO_CUDA_T(0xbfdf7d03827dc3ac),
    TO_CUDA_T(0x837e92f041790bf9), TO_CUDA_T(0x006dfccb1e914b88)
};

inline constexpr uint32_t BW6_761_rone_host[12] = { /* (1<<384)%r */
    TO_CUDA_T(0x02cdffffffffff68), TO_CUDA_T(0x51409f837fffffb1),
    TO_CUDA_T(0x9f7db3a98a7d3ff2), TO_CUDA_T(0x7b4e97b76e7c6305),
    TO_CUDA_T(0x4cf495bf803c84e8), TO_CUDA_T(0x008d6661e2fdf49a)
};

inline constexpr uint32_t BW6_761_rx7_host[12] = { /* r << 7 (left-aligned) */
    TO_CUDA_T(0x8460000000000080), TO_CUDA_T(0x85aea21800000042),
    TO_CUDA_T(0x79b117dd04a4000b), TO_CUDA_T(0x116cf9807a89c78f),
    TO_CUDA_T(0x1d82e03650a49d8d), TO_CUDA_T(0xd71d230be2887563)
};

// Reuse field_large_t on host for basic type-checking; not optimized for CPU use
typedef field_large_t<761, BW6_761_P_host, 0x8fa798dd,
                      BW6_761_RR_host, BW6_761_one_host,
                      BW6_761_Px7_host> fp_large;

struct fp_t : public fp_large {
    using mem_t = fp_t;
    using fp_large::bit_length;
    using fp_large::zero;
    using fp_large::degree;
    using fp_large::csel;

    inline fp_t() : fp_large() {}
    inline fp_t(const fp_large& a) : fp_large(a) {}
    inline fp_t(uint32_t v) : fp_large(v) {}

    inline static fp_t zero() { return fp_large::zero(); }
    inline static fp_t one() { return fp_large::one(); }
    inline static fp_t one(bool set_z) { return fp_large::one(set_z); }

    inline void inf() {
        for (unsigned i = 0; i < fp_large::NLIMBS; i++)
            this->val[i] = 0;
    }
};

typedef field_large_t<377, BW6_761_r_host, 0xffffffff,
                      BW6_761_rRR_host, BW6_761_rone_host,
                      BW6_761_rx7_host> fr_large;

struct fr_t : public fr_large {
    using mem_t = fr_t;
    using fr_large::bit_length;
    using fr_large::degree;
    using fr_large::csel;

    inline fr_t() {}
    inline fr_t(const fr_large& a) : fr_large(a) {}
    template<typename... Ts>
    constexpr fr_t(Ts... a) : fr_large{a...} {}

    inline static fr_t zero() { return fr_large::zero(); }
    inline static fr_t one() { return fr_large::one(); }
    inline static fr_t one(bool set_z) { return fr_large::one(set_z); }
};
} // namespace bw6_761

#endif

#ifdef FEATURE_BW6_761
using namespace bw6_761;
#endif

#endif
