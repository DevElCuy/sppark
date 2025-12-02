// Large field arithmetic for BW6-761 (761-bit fields)
// Inspired by Icicle's simpler template approach
// Compatible with sppark's jacobian_t, xyzz_t, and pippenger.cuh
//
// This implementation uses a simpler template structure than mont_t
// to avoid template instantiation depth issues with large (24-limb) fields.

#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define HOST_INLINE __host__ __forceinline__
#define DEVICE_INLINE __device__ __forceinline__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define HOST_INLINE inline
#define DEVICE_INLINE inline
#define HOST_DEVICE_INLINE inline
#endif

// Simple storage template for limbs
template <unsigned LIMBS_COUNT>
struct storage_large {
    static constexpr unsigned LC = LIMBS_COUNT;
    uint32_t limbs[LIMBS_COUNT];
};

// PTX intrinsics for efficient arithmetic on device
namespace ptx_large {
#ifdef __CUDA_ARCH__
    __device__ __forceinline__ uint32_t add_cc(uint32_t a, uint32_t b) {
        uint32_t r;
        asm("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
        return r;
    }

    __device__ __forceinline__ uint32_t addc_cc(uint32_t a, uint32_t b) {
        uint32_t r;
        asm("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
        return r;
    }

    __device__ __forceinline__ uint32_t addc(uint32_t a, uint32_t b) {
        uint32_t r;
        asm("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
        return r;
    }

    __device__ __forceinline__ uint32_t sub_cc(uint32_t a, uint32_t b) {
        uint32_t r;
        asm("sub.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
        return r;
    }

    __device__ __forceinline__ uint32_t subc_cc(uint32_t a, uint32_t b) {
        uint32_t r;
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
        return r;
    }

    __device__ __forceinline__ uint32_t subc(uint32_t a, uint32_t b) {
        uint32_t r;
        asm("subc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
        return r;
    }

    __device__ __forceinline__ uint32_t mul_lo(uint32_t a, uint32_t b) {
        uint32_t r;
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
        return r;
    }

    __device__ __forceinline__ uint32_t mul_hi(uint32_t a, uint32_t b) {
        uint32_t r;
        asm("mul.hi.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
        return r;
    }

    __device__ __forceinline__ uint32_t mad_lo_cc(uint32_t a, uint32_t b, uint32_t c) {
        uint32_t r;
        asm("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
        return r;
    }

    __device__ __forceinline__ uint32_t madc_hi_cc(uint32_t a, uint32_t b, uint32_t c) {
        uint32_t r;
        asm("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
        return r;
    }

    __device__ __forceinline__ uint32_t mad_lo(uint32_t a, uint32_t b, uint32_t c) {
        uint32_t r;
        asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
        return r;
    }

    __device__ __forceinline__ uint32_t madc_hi(uint32_t a, uint32_t b, uint32_t c) {
        uint32_t r;
        asm("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
        return r;
    }
#endif
}

// Large field template - simpler than mont_t to avoid template instantiation issues
template<unsigned NBITS,
         const uint32_t* MOD,
         uint32_t M0,  // M0 value directly as template parameter
         const uint32_t* RR,
         const uint32_t* ONE,
         const uint32_t* MOD_SHIFTED>
class __align__(8) field_large_t {
public:
    static constexpr unsigned NLIMBS = (NBITS + 31) / 32;
    static constexpr HOST_DEVICE_INLINE unsigned bit_length() { return NBITS; }
    static constexpr unsigned degree = 1;

    uint32_t val[NLIMBS];

    // Default constructor
    DEVICE_INLINE field_large_t() {
#pragma unroll
        for (unsigned i = 0; i < NLIMBS; i++)
            val[i] = 0;
    }

    // Copy constructor
    DEVICE_INLINE field_large_t(const field_large_t& other) {
#pragma unroll
        for (unsigned i = 0; i < NLIMBS; i++)
            val[i] = other.val[i];
    }

    // Constructor from single value
    DEVICE_INLINE field_large_t(uint32_t v) {
        val[0] = v;
#pragma unroll
        for (unsigned i = 1; i < NLIMBS; i++)
            val[i] = 0;
    }

    // Variadic constructor for initializer list (only for integral types)
    template<typename... Ts, typename = typename std::enable_if<
        (sizeof...(Ts) > 1 && std::is_integral<typename std::common_type<Ts...>::type>::value)>::type>
    DEVICE_INLINE field_large_t(Ts... a) : val{static_cast<uint32_t>(a)...} {}

    // Copy assignment
    DEVICE_INLINE field_large_t& operator=(const field_large_t& other) {
#pragma unroll
        for (unsigned i = 0; i < NLIMBS; i++)
            val[i] = other.val[i];
        return *this;
    }

    // Static constructors
    static HOST_DEVICE_INLINE field_large_t zero() {
        field_large_t r;
#pragma unroll
        for (unsigned i = 0; i < NLIMBS; i++)
            r.val[i] = 0;
        return r;
    }

    static HOST_DEVICE_INLINE field_large_t one() {
        field_large_t r;
#pragma unroll
        for (unsigned i = 0; i < NLIMBS; i++)
            r.val[i] = ONE[i];
        return r;
    }

    // one() with parameter: return zero if set_z is true, otherwise return one
    static HOST_DEVICE_INLINE field_large_t one(bool set_z) {
        return set_z ? zero() : one();
    }

    // Comparison
    HOST_DEVICE_INLINE bool is_zero() const {
#ifdef __CUDA_ARCH__
        uint32_t result = val[0];
#pragma unroll
        for (unsigned i = 1; i < NLIMBS; i++)
            result |= val[i];
        return result == 0;
#else
        for (unsigned i = 0; i < NLIMBS; i++)
            if (val[i] != 0) return false;
        return true;
#endif
    }

    HOST_DEVICE_INLINE bool operator==(const field_large_t& other) const {
#ifdef __CUDA_ARCH__
        uint32_t diff = 0;
#pragma unroll
        for (unsigned i = 0; i < NLIMBS; i++)
            diff |= (val[i] ^ other.val[i]);
        return diff == 0;
#else
        for (unsigned i = 0; i < NLIMBS; i++)
            if (val[i] != other.val[i]) return false;
        return true;
#endif
    }

    HOST_DEVICE_INLINE bool operator!=(const field_large_t& other) const {
        return !(*this == other);
    }

    // Portable addition path shared by host and large-field device builds
    HOST_DEVICE_INLINE field_large_t add_portable(const field_large_t& other) const {
        field_large_t r;
        uint64_t carry = 0;
        for (unsigned i = 0; i < NLIMBS; i++) {
            uint64_t sum = (uint64_t)val[i] + other.val[i] + carry;
            r.val[i] = (uint32_t)sum;
            carry = sum >> 32;
        }

        bool gte = (carry != 0);
        if (!gte) {
            for (int i = NLIMBS - 1; i >= 0; i--) {
                if (r.val[i] > MOD[i]) {
                    gte = true;
                    break;
                } else if (r.val[i] < MOD[i]) {
                    break;
                }
            }
        }

        if (gte) {
            uint64_t borrow = 0;
            for (unsigned i = 0; i < NLIMBS; i++) {
                uint64_t diff = (uint64_t)r.val[i] - MOD[i] - borrow;
                r.val[i] = (uint32_t)diff;
                borrow = (diff >> 32) & 1;
            }
        }

        return r;
    }

    // Addition with reduction
    HOST_DEVICE_INLINE field_large_t operator+(const field_large_t& other) const {
#ifdef __CUDA_ARCH__
        if constexpr (NLIMBS <= 12) {
            field_large_t r;
            // Add
            r.val[0] = ptx_large::add_cc(val[0], other.val[0]);
#pragma unroll
            for (unsigned i = 1; i < NLIMBS; i++)
                r.val[i] = ptx_large::addc_cc(val[i], other.val[i]);
            uint32_t carry = ptx_large::addc(0, 0);

            // Reduce if needed: subtract modulus
            field_large_t reduced;
            uint32_t borrow = ptx_large::sub_cc(r.val[0], MOD[0]);
            reduced.val[0] = r.val[0] - MOD[0];
#pragma unroll
            for (unsigned i = 1; i < NLIMBS; i++) {
                borrow = ptx_large::subc_cc(r.val[i], MOD[i]);
                reduced.val[i] = borrow;
            }
            borrow = ptx_large::subc(0, 0);

            // Select: if no carry from add and borrow from sub, keep r; else use reduced
            bool should_reduce = (carry != 0) || (borrow == 0);
#pragma unroll
            for (unsigned i = 0; i < NLIMBS; i++)
                r.val[i] = should_reduce ? reduced.val[i] : r.val[i];

            return r;
        }
#endif
        return add_portable(other);
    }

    // Portable subtraction used for host and large-field device builds
    HOST_DEVICE_INLINE field_large_t sub_portable(const field_large_t& other) const {
        field_large_t r;
        uint64_t borrow = 0;
        for (unsigned i = 0; i < NLIMBS; i++) {
            uint64_t diff = (uint64_t)val[i] - other.val[i] - borrow;
            r.val[i] = (uint32_t)diff;
            borrow = (diff >> 32) & 1;
        }

        if (borrow) {
            uint64_t carry = 0;
            for (unsigned i = 0; i < NLIMBS; i++) {
                uint64_t sum = (uint64_t)r.val[i] + MOD[i] + carry;
                r.val[i] = (uint32_t)sum;
                carry = sum >> 32;
            }
        }

        return r;
    }

    // Subtraction with reduction
    HOST_DEVICE_INLINE field_large_t operator-(const field_large_t& other) const {
#ifdef __CUDA_ARCH__
        if constexpr (NLIMBS <= 12) {
            field_large_t r;
            r.val[0] = ptx_large::sub_cc(val[0], other.val[0]);
#pragma unroll
            for (unsigned i = 1; i < NLIMBS; i++)
                r.val[i] = ptx_large::subc_cc(val[i], other.val[i]);
            uint32_t borrow = ptx_large::subc(0, 0);

            // If borrow, add modulus
            if (borrow) {
                r.val[0] = ptx_large::add_cc(r.val[0], MOD[0]);
#pragma unroll
                for (unsigned i = 1; i < NLIMBS; i++)
                    r.val[i] = ptx_large::addc_cc(r.val[i], MOD[i]);
                ptx_large::addc(0, 0);
            }

            return r;
        }
#endif
        return sub_portable(other);
    }

    HOST_DEVICE_INLINE field_large_t neg_portable() const {
        if (is_zero()) return *this;

        field_large_t r;
        uint64_t borrow = 0;
        for (unsigned i = 0; i < NLIMBS; i++) {
            uint64_t diff = (uint64_t)MOD[i] - val[i] - borrow;
            r.val[i] = (uint32_t)diff;
            borrow = (diff >> 32) & 1;
        }
        return r;
    }

    // Negation
    HOST_DEVICE_INLINE field_large_t operator-() const {
        if (is_zero()) return *this;

#ifdef __CUDA_ARCH__
        if constexpr (NLIMBS <= 12) {
            field_large_t r;
            r.val[0] = ptx_large::sub_cc(MOD[0], val[0]);
#pragma unroll
            for (unsigned i = 1; i < NLIMBS; i++)
                r.val[i] = ptx_large::subc_cc(MOD[i], val[i]);
            ptx_large::subc(0, 0);
            return r;
        }
#endif
        return neg_portable();
    }

    // Portable multiplication used for host and large-field device builds
    HOST_DEVICE_INLINE field_large_t mul_portable(const field_large_t& other) const {
        uint64_t t[2 * NLIMBS + 1];
        for (unsigned i = 0; i < 2 * NLIMBS + 1; i++)
            t[i] = 0;

        // Multiply
        for (unsigned i = 0; i < NLIMBS; i++) {
            uint64_t carry = 0;
            for (unsigned j = 0; j < NLIMBS; j++) {
                uint64_t prod = (uint64_t)val[i] * other.val[j];
                uint64_t sum = t[i + j] + prod + carry;
                t[i + j] = (uint32_t)sum;
                carry = sum >> 32;
            }
            t[i + NLIMBS] += carry;
        }

        // Montgomery reduction
        for (unsigned i = 0; i < NLIMBS; i++) {
            uint64_t m = (uint64_t)t[i] * M0;
            uint64_t carry = 0;
            for (unsigned j = 0; j < NLIMBS; j++) {
                uint64_t prod = m * MOD[j];
                uint64_t sum = t[i + j] + (uint32_t)prod + carry;
                t[i + j] = (uint32_t)sum;
                carry = (sum >> 32) + (prod >> 32);
            }
            uint64_t sum = t[i + NLIMBS] + carry;
            t[i + NLIMBS] = (uint32_t)sum;
            t[i + NLIMBS + 1] += (sum >> 32);
        }

        // Result extraction
        field_large_t r;
        for (unsigned i = 0; i < NLIMBS; i++)
            r.val[i] = (uint32_t)t[i + NLIMBS];

        // Final reduction
        bool gte = false;
        for (int i = NLIMBS - 1; i >= 0; i--) {
            if (r.val[i] > MOD[i]) {
                gte = true;
                break;
            } else if (r.val[i] < MOD[i]) {
                break;
            }
        }

        if (gte) {
            uint64_t borrow = 0;
            for (unsigned i = 0; i < NLIMBS; i++) {
                uint64_t diff = (uint64_t)r.val[i] - MOD[i] - borrow;
                r.val[i] = (uint32_t)diff;
                borrow = (diff >> 32) & 1;
            }
        }

        return r;
    }

    // Montgomery multiplication: (a * b * R^-1) mod p
    DEVICE_INLINE field_large_t operator*(const field_large_t& other) const {
#ifdef __CUDA_ARCH__
        if constexpr (NLIMBS <= 12) {
            // CIOS Montgomery multiplication
            uint32_t t[NLIMBS + 1];

#pragma unroll 1  // Don't unroll outer loop to save registers
            for (unsigned i = 0; i < NLIMBS; i++)
                t[i] = 0;
            t[NLIMBS] = 0;

#pragma unroll 1
            for (unsigned i = 0; i < NLIMBS; i++) {
                uint32_t carry = 0;

                // t = t + val[i] * other
#pragma unroll
                for (unsigned j = 0; j < NLIMBS; j++) {
                    uint32_t lo = ptx_large::mul_lo(val[i], other.val[j]);
                    uint32_t hi = ptx_large::mul_hi(val[i], other.val[j]);

                    uint32_t sum_lo = ptx_large::add_cc(t[j], lo);
                    uint32_t sum_hi = ptx_large::addc_cc(carry, hi);
                    uint32_t sum_carry = ptx_large::addc(0, 0);

                    t[j] = sum_lo;
                    carry = ptx_large::add_cc(sum_hi, sum_carry);
                    uint32_t c2 = ptx_large::addc(0, 0);
                    carry = ptx_large::add_cc(carry, c2);
                    ptx_large::addc(0, 0);
                }

                t[NLIMBS] = ptx_large::add_cc(t[NLIMBS], carry);
                uint32_t overflow = ptx_large::addc(0, 0);

                // Montgomery reduction step
                uint32_t m = ptx_large::mul_lo(t[0], M0);
                carry = 0;

                // t = t + m * MOD
                uint32_t k_lo = ptx_large::mul_lo(m, MOD[0]);
                uint32_t k_hi = ptx_large::mul_hi(m, MOD[0]);
                uint32_t tmp = ptx_large::add_cc(t[0], k_lo);
                carry = ptx_large::addc(k_hi, 0);

#pragma unroll
                for (unsigned j = 1; j < NLIMBS; j++) {
                    uint32_t lo = ptx_large::mul_lo(m, MOD[j]);
                    uint32_t hi = ptx_large::mul_hi(m, MOD[j]);

                    t[j - 1] = ptx_large::add_cc(t[j], lo);
                    uint32_t c1 = ptx_large::addc_cc(carry, hi);
                    uint32_t c2 = ptx_large::addc(0, 0);

                    carry = ptx_large::add_cc(c1, c2);
                    ptx_large::addc(0, 0);
                }

                t[NLIMBS - 1] = ptx_large::add_cc(t[NLIMBS], carry);
                t[NLIMBS] = ptx_large::addc(overflow, 0);
            }

            // Final reduction
            field_large_t r;
            uint32_t borrow = ptx_large::sub_cc(t[0], MOD[0]);
            r.val[0] = borrow;
#pragma unroll
            for (unsigned i = 1; i < NLIMBS; i++) {
                borrow = ptx_large::subc_cc(t[i], MOD[i]);
                r.val[i] = borrow;
            }
            borrow = ptx_large::subc(t[NLIMBS], 0);

            // Select result
            bool use_t = (borrow != 0);
#pragma unroll
            for (unsigned i = 0; i < NLIMBS; i++)
                r.val[i] = use_t ? t[i] : r.val[i];

            return r;
        }
#endif
        return mul_portable(other);
    }

    // Squaring (just use multiplication for simplicity)
    DEVICE_INLINE field_large_t sqr() const {
        return (*this) * (*this);
    }

    // Convert to Montgomery form
    DEVICE_INLINE field_large_t to_montgomery() const {
        field_large_t rr;
#pragma unroll
        for (unsigned i = 0; i < NLIMBS; i++)
            rr.val[i] = RR[i];
        return (*this) * rr;
    }

    // Convert from Montgomery form
    DEVICE_INLINE field_large_t from_montgomery() const {
        field_large_t one_val;
        one_val.val[0] = 1;
#pragma unroll
        for (unsigned i = 1; i < NLIMBS; i++)
            one_val.val[i] = 0;
        return (*this) * one_val;
    }

    // Check if even (for conditional logic)
    DEVICE_INLINE bool is_even() const {
        return (val[0] & 1) == 0;
    }

    // Compound assignment operators
    DEVICE_INLINE field_large_t& operator+=(const field_large_t& other) {
        *this = *this + other;
        return *this;
    }

    DEVICE_INLINE field_large_t& operator-=(const field_large_t& other) {
        *this = *this - other;
        return *this;
    }

    DEVICE_INLINE field_large_t& operator*=(const field_large_t& other) {
        *this = *this * other;
        return *this;
    }

    // Conditional negation: if flag is true and value is non-zero, negate it
    DEVICE_INLINE field_large_t& cneg(bool flag) {
        if (!flag || is_zero()) return *this;
        *this = -*this;
        return *this;
    }

    static HOST_DEVICE_INLINE field_large_t cneg(const field_large_t& a, bool flag) {
        field_large_t r = a;
        return r.cneg(flag);
    }

    // Conditional zero: if set_z is non-zero, return zero, otherwise return a
    friend HOST_DEVICE_INLINE field_large_t czero(const field_large_t& a, int set_z) {
        if (set_z) return field_large_t::zero();
        return a;
    }

    // Left shift with modular reduction
    DEVICE_INLINE field_large_t& operator<<=(unsigned l) {
        while (l--) {
            *this = *this + *this;  // Double the value with reduction
        }
        return *this;
    }

    friend DEVICE_INLINE field_large_t operator<<(field_large_t a, unsigned l) {
        return a <<= l;
    }

    // Two-argument is_zero for compatibility with affine point checks
    // In mont_t, this seems to check if either argument is zero
    HOST_DEVICE_INLINE bool is_zero(const field_large_t& other) const {
        return is_zero() || other.is_zero();
    }

    // Conditional select: return a if sel_a is non-zero, otherwise return b
    static HOST_DEVICE_INLINE field_large_t csel(const field_large_t& a, const field_large_t& b, int sel_a) {
#ifdef __CUDA_ARCH__
        field_large_t r;
        bool select_a = (sel_a != 0);
#pragma unroll
        for (unsigned i = 0; i < NLIMBS; i++)
            r.val[i] = select_a ? a.val[i] : b.val[i];
        return r;
#else
        return (sel_a != 0) ? a : b;
#endif
    }

    // Power operator (exponentiation)
    DEVICE_INLINE field_large_t& operator^=(uint32_t p) {
        if (p == 0) {
            *this = one();
            return *this;
        }
        if (p == 1) return *this;
        if (p == 2) {
            *this = sqr();
            return *this;
        }

        field_large_t base = *this;
        *this = one();
        while (p > 0) {
            if (p & 1) *this = *this * base;
            base = base.sqr();
            p >>= 1;
        }
        return *this;
    }

    friend DEVICE_INLINE field_large_t operator^(field_large_t a, uint32_t p) {
        return a ^= p;
    }

    // Power with int (for compile-time constants)
    DEVICE_INLINE field_large_t& operator^=(int p) {
        return *this ^= (uint32_t)p;
    }

    friend DEVICE_INLINE field_large_t operator^(field_large_t a, int p) {
        if (p == 2) return a.sqr();  // Optimize squaring
        return a ^= p;
    }

    // Squaring helper
    friend DEVICE_INLINE field_large_t sqr(const field_large_t& a) {
        return a.sqr();
    }
};
