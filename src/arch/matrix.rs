use core::arch::x86_64::*;
// template <bool Translate, bool Normalized>

pub unsafe fn mat4x4_12_true_true(b: __m128, c: &__m128, out: &mut [__m128]) {
    let mut buf = [0.0f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), _mm_mul_ps(b, b));
    let [b0_2, b1_2, b2_2, b3_2] = buf;

    {
        let c0 = &mut out[0];
        *c0 = _mm_mul_ps(b, swizzle!(b, 0, 0, 2, 0));
        let tmp = _mm_mul_ps(swizzle!(b, 0, 1, 3, 1), swizzle!(b, 0, 3, 0, 1));
        let tmp = _mm_xor_ps(_mm_set_ps(0.0, 0.0, -0.0, 0.0), tmp);
        *c0 = _mm_mul_ps(_mm_set_ps(0.0, 2.0, 2.0, 1.0), _mm_add_ps(*c0, tmp));
        *c0 = _mm_sub_ps(*c0, _mm_set_ss(b3_2 + b2_2));
    }

    {
        let c1 = &mut out[1];
        *c1 = _mm_mul_ps(b, swizzle!(b, 0, 3, 1, 3));
        let tmp = _mm_mul_ps(swizzle!(b, 0, 0, 3, 2), swizzle!(b, 0, 1, 3, 1));
        let tmp = _mm_xor_ps(_mm_set_ps(0.0, -0.0, 0.0, 0.0), tmp);
        *c1 = _mm_mul_ps(_mm_set_ps(0.0, 2.0, -1.0, 2.0), _mm_add_ps(*c1, tmp));
        *c1 = _mm_add_ps(*c1, _mm_set_ps(0.0, 0.0, b0_2 + b2_2, 0.0));
    }

    {
        let c2 = &mut out[2];
        *c2 = _mm_xor_ps(
            _mm_set_ps(0.0, -0.0, 0.0, -0.0),
            _mm_mul_ps(b, swizzle!(b, 0, 2, 0, 2)),
        );
        *c2 = _mm_add_ps(
            *c2,
            _mm_mul_ps(swizzle!(b, 0, 0, 2, 1), swizzle!(b, 0, 0, 3, 3)),
        );
        *c2 = _mm_mul_ps(*c2, _mm_set_ps(0.0, 1.0, 2.0, 2.0));
        *c2 = _mm_add_ps(*c2, _mm_set_ps(0.0, b3_2 - b1_2, 0.0, 0.0));
    }

    {
        let c3 = &mut out[3];
        *c3 = _mm_mul_ps(b, swizzle!(*c, 0, 1, 3, 1));
        *c3 = _mm_add_ps(
            *c3,
            _mm_mul_ps(swizzle!(b, 0, 0, 0, 3), swizzle!(*c, 0, 3, 2, 2)),
        );
        *c3 = _mm_add_ps(
            *c3,
            _mm_mul_ps(swizzle!(b, 0, 3, 2, 1), swizzle!(*c, 0, 0, 0, 0)),
        );
        let tmp = _mm_mul_ps(swizzle!(b, 0, 1, 3, 2), swizzle!(*c, 0, 2, 1, 3));
        *c3 = _mm_mul_ps(_mm_set_ps(0.0, 2.0, 2.0, 2.0), _mm_sub_ps(tmp, *c3));

        *c3 = if cfg!(target_feature = "sse4.1") {
            _mm_blend_ps(*c3, _mm_set_ps(1.0, 0.0, 0.0, 0.0), 0b1000)
        } else {
            _mm_add_ps(*c3, _mm_set_ps(1.0, 0.0, 0.0, 0.0))
        }
    }
}

/*
template <>
KLN_INLINE void KLN_VEC_CALL
mat4x4_12<true, false>(__m128 b, [[maybe_unused]] __m128 const* c, __m128* out) noexcept
{
    float buf[4];
    _mm_storeu_ps(buf, _mm_mul_ps(b, b));
    float b0_2 = buf[0];
    float b1_2 = buf[1];
    float b2_2 = buf[2];
    float b3_2 = buf[3];

    __m128& c0 = out[0];
    c0         = _mm_mul_ps(b, swizzle!(b, 0, 0, 2, 0));
    __m128 tmp
        = _mm_mul_ps(swizzle!(b, 0, 1, 3, 1), swizzle!(b, 0, 3, 0, 1));
    tmp = _mm_xor_ps(_mm_set_ps(0.0, 0.0, -0.0, 0.0), tmp);
    c0  = _mm_mul_ps(_mm_set_ps(0.0, 2.0, 2.0, 1.0), _mm_add_ps(c0, tmp));
    c0  = _mm_sub_ps(c0, _mm_set_ss(b3_2 + b2_2));

    __m128& c1 = out[1];
    c1         = _mm_mul_ps(b, swizzle!(b, 0, 3, 1, 3));
    tmp = _mm_mul_ps(swizzle!(b, 0, 0, 3, 2), swizzle!(b, 0, 1, 3, 1));
    tmp = _mm_xor_ps(_mm_set_ps(0.0, -0.0, 0.0, 0.0), tmp);
    c1  = _mm_mul_ps(_mm_set_ps(0.0, 2.0, -1.0, 2.0), _mm_add_ps(c1, tmp));
    c1  = _mm_add_ps(c1, _mm_set_ps(0.0, 0.0, b0_2 + b2_2, 0.0));

    __m128& c2 = out[2];
    c2         = _mm_xor_ps(_mm_set_ps(0.0, -0.0, 0.0, -0.0),
                    _mm_mul_ps(b, swizzle!(b, 0, 2, 0, 2)));
    c2         = _mm_add_ps(
        c2, _mm_mul_ps(swizzle!(b, 0, 0, 2, 1), swizzle!(b, 0, 0, 3, 3)));
    c2 = _mm_mul_ps(c2, _mm_set_ps(0.0, 1.0, 2.0, 2.0));
    c2 = _mm_add_ps(c2, _mm_set_ps(0.0, b3_2 - b1_2, 0.0, 0.0));

    __m128& c3 = out[3];
    c3         = _mm_mul_ps(b, swizzle!(*c, 0, 1, 3, 1));
    c3         = _mm_add_ps(
        c3, _mm_mul_ps(swizzle!(b, 0, 0, 0, 3), swizzle!(*c, 0, 3, 2, 2)));
    c3 = _mm_add_ps(
        c3, _mm_mul_ps(swizzle!(b, 0, 3, 2, 1), swizzle!(*c, 0, 0, 0, 0)));
    tmp = _mm_mul_ps(swizzle!(b, 0, 1, 3, 2), swizzle!(*c, 0, 2, 1, 3));
    c3  = _mm_mul_ps(_mm_set_ps(0.0, 2.0, 2.0, 2.0), _mm_sub_ps(tmp, c3));

#ifdef KLEIN_SSE_4_1
    c3 = _mm_blend_ps(
        c3, _mm_set_ps(b0_2 + b1_2 + b2_2 + b3_2, 0.0, 0.0, 0.0), 0b1000);
#else
    c3 = _mm_add_ps(c3, _mm_set_ps(b0_2 + b1_2 + b2_2 + b3_2, 0.0, 0.0, 0.0));
#endif
}

template <>
KLN_INLINE void KLN_VEC_CALL
mat4x4_12<false, true>(__m128 b, [[maybe_unused]] __m128 const* c, __m128* out) noexcept
{
    float buf[4];
    _mm_storeu_ps(buf, _mm_mul_ps(b, b));
    float b0_2 = buf[0];
    float b1_2 = buf[1];
    float b2_2 = buf[2];
    float b3_2 = buf[3];

    __m128& c0 = out[0];
    c0         = _mm_mul_ps(b, swizzle!(b, 0, 0, 2, 0));
    __m128 tmp
        = _mm_mul_ps(swizzle!(b, 0, 1, 3, 1), swizzle!(b, 0, 3, 0, 1));
    tmp = _mm_xor_ps(_mm_set_ps(0.0, 0.0, -0.0, 0.0), tmp);
    c0  = _mm_mul_ps(_mm_set_ps(0.0, 2.0, 2.0, 1.0), _mm_add_ps(c0, tmp));
    c0  = _mm_sub_ps(c0, _mm_set_ss(b3_2 + b2_2));

    __m128& c1 = out[1];
    c1         = _mm_mul_ps(b, swizzle!(b, 0, 3, 1, 3));
    tmp = _mm_mul_ps(swizzle!(b, 0, 0, 3, 2), swizzle!(b, 0, 1, 3, 1));
    tmp = _mm_xor_ps(_mm_set_ps(0.0, -0.0, 0.0, 0.0), tmp);
    c1  = _mm_mul_ps(_mm_set_ps(0.0, 2.0, -1.0, 2.0), _mm_add_ps(c1, tmp));
    c1  = _mm_add_ps(c1, _mm_set_ps(0.0, 0.0, b0_2 + b2_2, 0.0));

    __m128& c2 = out[2];
    c2         = _mm_xor_ps(_mm_set_ps(0.0, -0.0, 0.0, -0.0),
                    _mm_mul_ps(b, swizzle!(b, 0, 2, 0, 2)));
    c2         = _mm_add_ps(
        c2, _mm_mul_ps(swizzle!(b, 0, 0, 2, 1), swizzle!(b, 0, 0, 3, 3)));
    c2 = _mm_mul_ps(c2, _mm_set_ps(0.0, 1.0, 2.0, 2.0));
    c2 = _mm_add_ps(c2, _mm_set_ps(0.0, b3_2 - b1_2, 0.0, 0.0));

    __m128& c3 = out[3];
#ifdef KLEIN_SSE_4_1
    c3 = _mm_blend_ps(c3, _mm_set_ps(1.0, 0.0, 0.0, 0.0), 0b1000);
#else
    c3 = _mm_add_ps(c3, _mm_set_ps(1.0, 0.0, 0.0, 0.0));
#endif
}

template <>
KLN_INLINE void KLN_VEC_CALL
mat4x4_12<false, false>(__m128 b, [[maybe_unused]] __m128 const* c, __m128* out) noexcept
{
    float buf[4];
    _mm_storeu_ps(buf, _mm_mul_ps(b, b));
    float b0_2 = buf[0];
    float b1_2 = buf[1];
    float b2_2 = buf[2];
    float b3_2 = buf[3];

    __m128& c0 = out[0];
    c0         = _mm_mul_ps(b, swizzle!(b, 0, 0, 2, 0));
    __m128 tmp
        = _mm_mul_ps(swizzle!(b, 0, 1, 3, 1), swizzle!(b, 0, 3, 0, 1));
    tmp = _mm_xor_ps(_mm_set_ps(0.0, 0.0, -0.0, 0.0), tmp);
    c0  = _mm_mul_ps(_mm_set_ps(0.0, 2.0, 2.0, 1.0), _mm_add_ps(c0, tmp));
    c0  = _mm_sub_ps(c0, _mm_set_ss(b3_2 + b2_2));

    __m128& c1 = out[1];
    c1         = _mm_mul_ps(b, swizzle!(b, 0, 3, 1, 3));
    tmp = _mm_mul_ps(swizzle!(b, 0, 0, 3, 2), swizzle!(b, 0, 1, 3, 1));
    tmp = _mm_xor_ps(_mm_set_ps(0.0, -0.0, 0.0, 0.0), tmp);
    c1  = _mm_mul_ps(_mm_set_ps(0.0, 2.0, -1.0, 2.0), _mm_add_ps(c1, tmp));
    c1  = _mm_add_ps(c1, _mm_set_ps(0.0, 0.0, b0_2 + b2_2, 0.0));

    __m128& c2 = out[2];
    c2         = _mm_xor_ps(_mm_set_ps(0.0, -0.0, 0.0, -0.0),
                    _mm_mul_ps(b, swizzle!(b, 0, 2, 0, 2)));
    c2         = _mm_add_ps(
        c2, _mm_mul_ps(swizzle!(b, 0, 0, 2, 1), swizzle!(b, 0, 0, 3, 3)));
    c2 = _mm_mul_ps(c2, _mm_set_ps(0.0, 1.0, 2.0, 2.0));
    c2 = _mm_add_ps(c2, _mm_set_ps(0.0, b3_2 - b1_2, 0.0, 0.0));

    __m128& c3 = out[3];
#ifdef KLEIN_SSE_4_1
    c3 = _mm_blend_ps(
        c3, _mm_set_ps(b0_2 + b1_2 + b2_2 + b3_2, 0.0, 0.0, 0.0), 0b1000);
#else
    c3 = _mm_add_ps(c3, _mm_set_ps(b0_2 + b1_2 + b2_2 + b3_2, 0.0, 0.0, 0.0));
#endif
}
*/
