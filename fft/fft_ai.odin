package fft

import "base:intrinsics"
import "base:runtime"
import "core:math"
import "core:math/bits"
import "core:simd"


/*
    ============================================================================================
    ADDED BY ARTIFICIAL INTELLIGENCE: this whole file was written by Claude, an AI model made by
    Anthropic, on 2026-10-16. It is an addition to fft-fun, nothing in fft.odin, misc.odin,
    naive.odin or test.odin was changed. Every name here starts with `ai_` so it can sit next to
    the radix-2 plan in the same package.

    A real input FFT built from the radix-2 plan in fft.odin, with these changes:

      1. Real input as a half size complex FFT. The N real samples are read as N/2 complex ones,
         the even samples the real parts and the odd ones the imaginary parts, and one pass after
         the FFT splits that into the spectrum of the real input. Half the butterflies.
      2. Radix-4 stages, and one radix-2 stage last when log2(N/2) is odd. Half as many passes
         over the data as radix-2 and a quarter fewer multiplies.
      3. Stockham autosort. Each stage reads one buffer and writes the other in the order the
         next stage reads it, so there is no bit reversal and no index table.
      4. The twiddles of each stage in their own table, read in order instead of striding
         through one table of N.
      5. Real and imaginary parts in separate arrays and #simd[4]f32 arithmetic, four
         butterflies at once: NEON on Apple Silicon, SSE on x86, SIMD128 on wasm.
      6. The first stage reads the samples directly, splitting even and odd samples as it loads
         them, so nothing is copied into the work buffers first.

    The output is laid out like pffft_transform_ordered: N/2 complex bins, bin 0 packs the two
    real ones as (DC, Nyquist). Forward only and not scaled, like the radix-2 plan.
    ============================================================================================
*/


// AI: four floats in one SIMD register
AI_F32x4 :: #simd[4]f32

// AI: one Stockham stage, n * stride is always N/2
AI_Stage :: struct {
    n:        int, // length of the sub-transforms at this stage
    stride:   int, // distance between neighbouring elements of a sub-transform
    twiddles: []f32, // radix-4: w¹, w², w³ of each p as six arrays of n/4 (re, im, re, im, re, im)
}

// AI: a plan for one size, its tables and work buffers
AI_Real_FFT_Plan :: struct {
    size:           int, // N, real samples in
    half:           int, // N/2, complex points of the inner FFT and bins out
    stages:         []AI_Stage,
    buffers:        [2][2][]f32, // ping and pong, each the real and imaginary parts of N/2 points
    split_twiddles: []f32, // ½·W_N^k for k < N/2, the real parts then the imaginary ones
}


// AI: N a power of two, at least 64 so every stage has four butterflies or more to vectorize
ai_create_real_fft_plan :: proc(size: int) -> (plan: AI_Real_FFT_Plan) {
    assert(bits.is_power_of_two(size) && size >= 64)

    plan.size = size
    plan.half = size / 2
    for &buffer in plan.buffers {
        for &part in buffer do part = ai_make_aligned(plan.half)
    }

    stages: [dynamic]AI_Stage
    n, stride := plan.half, 1
    for n >= 4 {
        m := n / 4
        twiddles := ai_make_aligned(6 * m)
        for p in 0 ..< m {
            for k in 1 ..= 3 {
                angle := -math.TAU * f64(k * p) / f64(n)
                twiddles[(2 * k - 2) * m + p] = f32(math.cos(angle))
                twiddles[(2 * k - 1) * m + p] = f32(math.sin(angle))
            }
        }
        append(&stages, AI_Stage{n, stride, twiddles})
        n /= 4
        stride *= 4
    }
    if n == 2 do append(&stages, AI_Stage{n = 2, stride = stride})
    plan.stages = stages[:]

    plan.split_twiddles = ai_make_aligned(2 * plan.half)
    for k in 0 ..< plan.half {
        angle := -math.TAU * f64(k) / f64(size)
        plan.split_twiddles[k] = f32(0.5 * math.cos(angle))
        plan.split_twiddles[plan.half + k] = f32(0.5 * math.sin(angle))
    }
    return
}

ai_destroy_real_fft_plan :: proc(plan: ^AI_Real_FFT_Plan) {
    for stage in plan.stages do delete(stage.twiddles)
    delete(plan.stages)
    for buffer in plan.buffers {
        for part in buffer do delete(part)
    }
    delete(plan.split_twiddles)
}

// AI: N real samples to N/2 bins, bin 0 is (DC, Nyquist)
ai_run_real_fft :: proc(plan: ^AI_Real_FFT_Plan, samples: []f32, spectrum: []complex64) #no_bounds_check {
    assert(len(samples) == plan.size && len(spectrum) == plan.half)

    ai_first_stage(plan.stages[0], raw_data(samples), plan.buffers[0])
    current := 0
    for stage in plan.stages[1:] {
        if stage.n == 2 {
            ai_radix2_stage(stage, plan.buffers[current], plan.buffers[1 - current])
        } else {
            ai_radix4_stage(stage, plan.buffers[current], plan.buffers[1 - current])
        }
        current = 1 - current
    }
    ai_split(plan, plan.buffers[current], cast([^]f32)raw_data(spectrum))
}


// AI: the first stage, stride 1, four values of p at once. The input is the samples, each pair
// a complex point, and the four outputs of each butterfly land next to each other, so the
// four butterflies' outputs are transposed on the way out. Public for benchmark_stages.
ai_first_stage :: proc(stage: AI_Stage, samples: [^]f32, dst: [2][]f32) #no_bounds_check {
    m := stage.n / 4
    tw := raw_data(stage.twiddles)
    yr, yi := raw_data(dst[0]), raw_data(dst[1])

    for p := 0; p < m; p += 4 {
        ar, ai := ai_load_pairs(samples[2 * p:])
        br, bi := ai_load_pairs(samples[2 * (p + m):])
        cr, ci := ai_load_pairs(samples[2 * (p + 2 * m):])
        dr, di := ai_load_pairs(samples[2 * (p + 3 * m):])

        y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i := ai_butterfly4(ar, ai, br, bi, cr, ci, dr, di)
        y1r, y1i = ai_cmul(y1r, y1i, ai_load(tw[p:]), ai_load(tw[m + p:]))
        y2r, y2i = ai_cmul(y2r, y2i, ai_load(tw[2 * m + p:]), ai_load(tw[3 * m + p:]))
        y3r, y3i = ai_cmul(y3r, y3i, ai_load(tw[4 * m + p:]), ai_load(tw[5 * m + p:]))

        ai_store_transposed(yr[4 * p:], y0r, y1r, y2r, y3r)
        ai_store_transposed(yi[4 * p:], y0i, y1i, y2i, y3i)
    }
}

// AI: a radix-4 stage with a stride of 4 or more, four values of q at once under one twiddle
ai_radix4_stage :: proc(stage: AI_Stage, src, dst: [2][]f32) #no_bounds_check {
    m := stage.n / 4
    s := stage.stride
    quarter := s * m // the distance between a butterfly's inputs, N/8
    tw := raw_data(stage.twiddles)
    xr, xi := raw_data(src[0]), raw_data(src[1])
    yr, yi := raw_data(dst[0]), raw_data(dst[1])

    for p in 0 ..< m {
        w1r, w1i := ai_splat(tw[p]), ai_splat(tw[m + p])
        w2r, w2i := ai_splat(tw[2 * m + p]), ai_splat(tw[3 * m + p])
        w3r, w3i := ai_splat(tw[4 * m + p]), ai_splat(tw[5 * m + p])

        for q := 0; q < s; q += 4 {
            i := s * p + q
            ar, ai := ai_load(xr[i:]), ai_load(xi[i:])
            br, bi := ai_load(xr[i + quarter:]), ai_load(xi[i + quarter:])
            cr, ci := ai_load(xr[i + 2 * quarter:]), ai_load(xi[i + 2 * quarter:])
            dr, di := ai_load(xr[i + 3 * quarter:]), ai_load(xi[i + 3 * quarter:])

            y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i := ai_butterfly4(ar, ai, br, bi, cr, ci, dr, di)
            y1r, y1i = ai_cmul(y1r, y1i, w1r, w1i)
            y2r, y2i = ai_cmul(y2r, y2i, w2r, w2i)
            y3r, y3i = ai_cmul(y3r, y3i, w3r, w3i)

            o := 4 * s * p + q
            ai_store(yr[o:], y0r); ai_store(yi[o:], y0i)
            ai_store(yr[o + s:], y1r); ai_store(yi[o + s:], y1i)
            ai_store(yr[o + 2 * s:], y2r); ai_store(yi[o + 2 * s:], y2i)
            ai_store(yr[o + 3 * s:], y3r); ai_store(yi[o + 3 * s:], y3i)
        }
    }
}

// AI: the last stage when log2(N/2) is odd, one radix-2 butterfly over the two halves, its
// twiddle is 1
ai_radix2_stage :: proc(stage: AI_Stage, src, dst: [2][]f32) #no_bounds_check {
    s := stage.stride
    xr, xi := raw_data(src[0]), raw_data(src[1])
    yr, yi := raw_data(dst[0]), raw_data(dst[1])

    for q := 0; q < s; q += 4 {
        ar, ai := ai_load(xr[q:]), ai_load(xi[q:])
        br, bi := ai_load(xr[q + s:]), ai_load(xi[q + s:])
        ai_store(yr[q:], ar + br); ai_store(yi[q:], ai + bi)
        ai_store(yr[q + s:], ar - br); ai_store(yi[q + s:], ai - bi)
    }
}

/*
    AI: Z is the FFT of the N/2 complex points z[k] = x[2k] + j·x[2k+1]. With A = Z[k] and
    B = conj(Z[N/2 - k]) the even samples' DFT is (A + B)/2, the odd samples' (A - B)/2j, and

        X[k] = (A + B)/2 - j·W_N^k·(A - B)/2

    The ½ is in the twiddles. Bin 0 and N/2, both real, are Z[0]'s real part plus and minus its
    imaginary part. Each k reads Z[N/2 - k] too, four bins at once read it backwards.
*/
ai_split :: proc(plan: ^AI_Real_FFT_Plan, z: [2][]f32, out: [^]f32) #no_bounds_check {
    half := plan.half
    zr, zi := raw_data(z[0]), raw_data(z[1])
    wr, wi := raw_data(plan.split_twiddles), raw_data(plan.split_twiddles[half:])

    out[0] = zr[0] + zi[0]
    out[1] = zr[0] - zi[0]

    for k in 1 ..< 4 {
        ar, ai := zr[k], zi[k]
        br, bi := zr[half - k], -zi[half - k]
        er, ei := 0.5 * (ar + br), 0.5 * (ai + bi)
        dr, di := ar - br, ai - bi
        or, oi := wr[k] * dr - wi[k] * di, wr[k] * di + wi[k] * dr
        out[2 * k] = er + oi
        out[2 * k + 1] = ei - or
    }

    HALF :: AI_F32x4{0.5, 0.5, 0.5, 0.5}
    for k := 4; k < half; k += 4 {
        ar, ai := ai_load(zr[k:]), ai_load(zi[k:])
        br := simd.swizzle(ai_load(zr[half - k - 3:]), 3, 2, 1, 0)
        bi := -simd.swizzle(ai_load(zi[half - k - 3:]), 3, 2, 1, 0)

        er, ei := HALF * (ar + br), HALF * (ai + bi)
        or, oi := ai_cmul(ar - br, ai - bi, ai_load(wr[k:]), ai_load(wi[k:]))
        xr, xi := er + oi, ei - or

        ai_store(out[2 * k:], simd.shuffle(xr, xi, 0, 4, 1, 5))
        ai_store(out[2 * k + 4:], simd.shuffle(xr, xi, 2, 6, 3, 7))
    }
}


// AI: the radix-4 butterfly, a 4-point DFT without twiddles
//     X0 = (a + c) + (b + d)      X1 = (a - c) - j(b - d)
//     X2 = (a + c) - (b + d)      X3 = (a - c) + j(b - d)
@(private = "file")
ai_butterfly4 :: #force_inline proc "contextless" (
    ar, ai, br, bi, cr, ci, dr, di: AI_F32x4,
) -> (
    y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i: AI_F32x4,
) {
    apcr, apci := ar + cr, ai + ci
    amcr, amci := ar - cr, ai - ci
    bpdr, bpdi := br + dr, bi + di
    bmdr, bmdi := br - dr, bi - di

    y0r, y0i = apcr + bpdr, apci + bpdi
    y1r, y1i = amcr + bmdi, amci - bmdr
    y2r, y2i = apcr - bpdr, apci - bpdi
    y3r, y3i = amcr - bmdi, amci + bmdr
    return
}

// AI: (xr + j·xi)(wr + j·wi), four at once. Odin doesn't fuse a multiply and an add on its own
// like clang does in pffft, simd.fma asks for the fused multiply-add instruction.
@(private = "file")
ai_cmul :: #force_inline proc "contextless" (xr, xi, wr, wi: AI_F32x4) -> (AI_F32x4, AI_F32x4) {
    return simd.fma(xr, wr, -(xi * wi)), simd.fma(xr, wi, xi * wr)
}

// AI: four complex points stored as pairs, split into their real and imaginary parts
@(private = "file")
ai_load_pairs :: #force_inline proc "contextless" (p: [^]f32) -> (re, im: AI_F32x4) {
    lo, hi := ai_load(p), ai_load(p[4:])
    return simd.shuffle(lo, hi, 0, 2, 4, 6), simd.shuffle(lo, hi, 1, 3, 5, 7)
}

// AI: rows r0..r3 stored as columns, 16 floats from dst
@(private = "file")
ai_store_transposed :: #force_inline proc "contextless" (dst: [^]f32, r0, r1, r2, r3: AI_F32x4) {
    t0 := simd.shuffle(r0, r1, 0, 4, 1, 5)
    t1 := simd.shuffle(r0, r1, 2, 6, 3, 7)
    t2 := simd.shuffle(r2, r3, 0, 4, 1, 5)
    t3 := simd.shuffle(r2, r3, 2, 6, 3, 7)
    ai_store(dst, simd.shuffle(t0, t2, 0, 1, 4, 5))
    ai_store(dst[4:], simd.shuffle(t0, t2, 2, 3, 6, 7))
    ai_store(dst[8:], simd.shuffle(t1, t3, 0, 1, 4, 5))
    ai_store(dst[12:], simd.shuffle(t1, t3, 2, 3, 6, 7))
}

@(private = "file")
ai_load :: #force_inline proc "contextless" (p: [^]f32) -> AI_F32x4 {
    return intrinsics.unaligned_load(cast(^AI_F32x4)p)
}

@(private = "file")
ai_store :: #force_inline proc "contextless" (p: [^]f32, v: AI_F32x4) {
    intrinsics.unaligned_store(cast(^AI_F32x4)p, v)
}

@(private = "file")
ai_splat :: #force_inline proc "contextless" (x: f32) -> AI_F32x4 {
    return AI_F32x4{x, x, x, x}
}

@(private = "file")
ai_make_aligned :: proc(count: int) -> []f32 {
    slice, _ := runtime.make_aligned([]f32, count, 16)
    return slice
}
