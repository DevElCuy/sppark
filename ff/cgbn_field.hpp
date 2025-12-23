#ifndef __SPPARK_FF_CGBN_FIELD_HPP__
#define __SPPARK_FF_CGBN_FIELD_HPP__

/**
 * CGBN-based Field Arithmetic Wrapper for Large Prime Fields
 *
 * Purpose: Provide mont_t-compatible interface using CGBN cooperative groups
 *          to avoid stack overflow for large fields (e.g., BW6-761's 761-bit field)
 *
 * Architecture:
 * - TPI threads cooperate on each field element (e.g., TPI=8 for 761-bit field)
 * - Field operations use cgbn_add, cgbn_mul, cgbn_modular_inverse, etc.
 * - Memory layout: cgbn_mem_t<BITS> for host/device transfers
 * - Computation: cgbn_t (register-based) for GPU operations
 *
 * Usage:
 *   typedef cgbn_field_t<768, 8, BW6_761_P_DEVICE> bw6_fp_t;
 *
 * Threading Model:
 *   WARNING: All methods must be called by TPI threads together!
 *   Instance ID: (blockIdx.x * blockDim.x + threadIdx.x) / TPI
 */

#include <cstdint>
#include <cgbn/cgbn.h>

namespace device {

template<unsigned BITS,           // Bit width (round up to 32-bit boundary, e.g., 768 for 761-bit field)
         unsigned TPI,            // Threads Per Instance (4, 8, 16, or 32)
         const uint32_t* MODULUS> // Device constant memory pointer to modulus
class cgbn_field_t {
public:
    // CGBN types
    typedef cgbn_context_t<TPI, cgbn_default_parameters_t> context_t;
    typedef cgbn_env_t<context_t, BITS> env_t;
    typedef typename env_t::cgbn_t cgbn_t;
    typedef cgbn_mem_t<BITS> mem_t;

    // Storage for field element (used in device/host memory)
    mem_t val;

    // Number of 32-bit limbs
    static constexpr unsigned NLIMBS = BITS / 32;

    /**
     * Default constructor (uninitialized)
     */
    __device__ __host__ cgbn_field_t() {}

    /**
     * Constructor from memory representation
     */
    __device__ __host__ cgbn_field_t(const mem_t& v) : val(v) {}

    /**
     * Load from device memory to register-based cgbn_t
     * Must be called by all TPI threads
     */
    __device__ static inline void load(env_t& env, cgbn_t& r, const cgbn_field_t* src) {
        cgbn_load(env, r, &(src->val));
    }

    /**
     * Store from register-based cgbn_t to device memory
     * Must be called by all TPI threads
     */
    __device__ static inline void store(env_t& env, cgbn_field_t* dst, const cgbn_t& a) {
        cgbn_store(env, &(dst->val), a);
    }

    /**
     * Field Addition: r = a + b (mod MODULUS)
     * Must be called by all TPI threads cooperatively
     */
    __device__ static inline void add(env_t& env, cgbn_t& r, const cgbn_t& a, const cgbn_t& b, const cgbn_t& modulus) {
        cgbn_add(env, r, a, b);
        cgbn_rem(env, r, r, modulus);
    }

    /**
     * Field Subtraction: r = a - b (mod MODULUS)
     */
    __device__ static inline void sub(env_t& env, cgbn_t& r, const cgbn_t& a, const cgbn_t& b, const cgbn_t& modulus) {
        cgbn_sub(env, r, a, b);
        // Handle negative result: if r < 0, add modulus
        if (cgbn_compare(env, r, cgbn_t::zero()) < 0) {
            cgbn_add(env, r, r, modulus);
        }
    }

    /**
     * Field Multiplication: r = a * b (mod MODULUS)
     */
    __device__ static inline void mul(env_t& env, cgbn_t& r, const cgbn_t& a, const cgbn_t& b, const cgbn_t& modulus) {
        cgbn_mul(env, r, a, b);
        cgbn_rem(env, r, r, modulus);
    }

    /**
     * Field Squaring: r = a^2 (mod MODULUS)
     */
    __device__ static inline void sqr(env_t& env, cgbn_t& r, const cgbn_t& a, const cgbn_t& modulus) {
        cgbn_mul(env, r, a, a);
        cgbn_rem(env, r, r, modulus);
    }

    /**
     * Field Inversion: r = a^-1 (mod MODULUS)
     * Uses Fermat's Little Theorem: a^-1 = a^(p-2) mod p
     */
    __device__ static inline void inv(env_t& env, cgbn_t& r, const cgbn_t& a, const cgbn_t& modulus) {
        cgbn_modular_inverse(env, r, a, modulus);
    }

    /**
     * Field Negation: r = -a (mod MODULUS)
     */
    __device__ static inline void neg(env_t& env, cgbn_t& r, const cgbn_t& a, const cgbn_t& modulus) {
        cgbn_sub(env, r, modulus, a);
    }

    /**
     * Check if zero
     */
    __device__ static inline bool is_zero(env_t& env, const cgbn_t& a) {
        return cgbn_equals_ui32(env, a, 0);
    }

    /**
     * Check equality
     */
    __device__ static inline bool equals(env_t& env, const cgbn_t& a, const cgbn_t& b) {
        return cgbn_equals(env, a, b);
    }

    /**
     * Set to zero
     */
    __device__ static inline void set_zero(env_t& env, cgbn_t& r) {
        cgbn_set_ui32(env, r, 0);
    }

    /**
     * Set to one (Montgomery form: R mod P, where R = 2^BITS)
     * NOTE: Caller must provide Montgomery one constant
     */
    __device__ static inline void set_one(env_t& env, cgbn_t& r, const cgbn_t& mont_one) {
        cgbn_set(env, r, mont_one);
    }

    /**
     * Copy
     */
    __device__ static inline void set(env_t& env, cgbn_t& r, const cgbn_t& a) {
        cgbn_set(env, r, a);
    }

    // Size information
    static constexpr size_t sizeof_val() { return sizeof(mem_t); }
    static constexpr unsigned nbits() { return BITS; }
    static constexpr unsigned nlimbs() { return NLIMBS; }
};

} // namespace device

#endif // __SPPARK_FF_CGBN_FIELD_HPP__
