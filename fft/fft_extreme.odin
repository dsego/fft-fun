package fft

import "base:intrinsics"
import "base:runtime"
import "core:math"
import "core:math/bits"
import "core:simd"


/*
    ============================================================================================
    ADDED BY ARTIFICIAL INTELLIGENCE: this whole file was written by Claude, an AI model made by
    Anthropic, on 2026-10-16. A new file next to fft_ai.odin, nothing else in fft-fun was
    changed for it. Every name here starts with `extreme_`.

    The AI FFT (fft_ai.odin), pushed to pffft's speed, guided by timing each pass on its own
    (benchmark_stages, the steps and their numbers in benchmark_stages/LOG.md):

      1. Bin k and bin N/2 - k come out of the split together, from one load of each, half the
         twiddles and multiplies of the AI FFT's split.
      2. The last FFT stage (radix-2 or radix-4, whichever log2(N/2) leaves) runs inside the
         split, its outputs go straight into the split's arithmetic instead of through a buffer.
         One pass over the data fewer.
      3. The stride 4 stage takes the twiddles of four p in six vector loads and each p's out of
         its lane, instead of six broadcast loads for every p. It had cost as much as 1.6 other
         stages.
      4. The butterflies at p = 0, whose twiddles are 1, skip the multiplies, in their own loop.
      5. The complex multiply's minus on a multiplicand, so it folds into a multiply-subtract.
      6. The first stage's loads and transposed stores in the shape LLVM makes ld2 and st4 of.
      7. The split's loops keep the plan's fields in locals, turn the mirrored bins around in
         the shuffle that pairs re with im, and need no negate. Its twiddles only up to N/4.
      8. Everything else as in fft_ai.odin: a real input as a half size complex FFT, radix-4
         Stockham stages, a twiddle table per stage, real and imaginary parts apart,
         #simd[4]f32 and fused multiply-adds.

    Tried and dropped, see LOG.md: radix-8 stages, twiddles stored already in all four lanes,
    unrolling the inner loop by two, and the four-p twiddles of 3. at stride 16 too.

    The output is laid out like pffft_transform_ordered: N/2 complex bins, bin 0 packs the two
    real ones as (DC, Nyquist). Forward only and not scaled.
    ============================================================================================
*/


// AI: four floats in one SIMD register
Extreme_F32x4 :: #simd[4]f32

// AI: one radix-4 Stockham stage, n * stride is always N/2
Extreme_Stage :: struct {
    n:        int, // length of the sub-transforms at this stage
    stride:   int, // distance between neighbouring elements of a sub-transform
    twiddles: []f32, // w¹, w², w³ of each p as six arrays of n/4: re, im, re, im, re, im
}

// AI: a plan for one size, its tables and work buffers
Extreme_FFT_Plan :: struct {
    size:           int, // N, real samples in
    half:           int, // N/2, complex points of the inner FFT and bins out
    stages:         []Extreme_Stage, // all but the last stage
    last_radix:     int, // 2 or 4, the last stage, run inside the split
    last_stride:    int, // N/2 / last_radix
    last_shift:     uint, // log2(last_stride)
    buffers:        [2][2][]f32, // ping and pong, each the real and imaginary parts of N/2 points
    split_re:       []f32, // ½·W_N^k for k ≤ N/4, real parts. The split pairs k with N/2 - k,
    split_im:       []f32, // k never goes past N/4
}


// AI: N a power of two, at least 128 so the split's vector loop has whole blocks of four
extreme_create_plan :: proc(size: int) -> (plan: Extreme_FFT_Plan) {
    assert(bits.is_power_of_two(size) && size >= 128)

    plan.size = size
    plan.half = size / 2
    for &buffer in plan.buffers {
        for &part in buffer do part = extreme_make_aligned(plan.half)
    }

    stages: [dynamic]Extreme_Stage
    n, stride := plan.half, 1
    for n > 4 {
        m := n / 4
        twiddles := extreme_make_aligned(6 * m)
        for p in 0 ..< m {
            for k in 1 ..= 3 {
                angle := -math.TAU * f64(k * p) / f64(n)
                twiddles[(2 * k - 2) * m + p] = f32(math.cos(angle))
                twiddles[(2 * k - 1) * m + p] = f32(math.sin(angle))
            }
        }
        append(&stages, Extreme_Stage{n, stride, twiddles})
        n /= 4
        stride *= 4
    }
    plan.stages = stages[:]
    plan.last_radix = n
    plan.last_stride = stride
    plan.last_shift = uint(bits.log2(uint(stride)))

    plan.split_re = extreme_make_aligned(plan.half / 2 + 1)
    plan.split_im = extreme_make_aligned(plan.half / 2 + 1)
    for k in 0 ..= plan.half / 2 {
        angle := -math.TAU * f64(k) / f64(size)
        plan.split_re[k] = f32(0.5 * math.cos(angle))
        plan.split_im[k] = f32(0.5 * math.sin(angle))
    }
    return
}

extreme_destroy_plan :: proc(plan: ^Extreme_FFT_Plan) {
    for stage in plan.stages do delete(stage.twiddles)
    delete(plan.stages)
    for buffer in plan.buffers {
        for part in buffer do delete(part)
    }
    delete(plan.split_re)
    delete(plan.split_im)
}

// AI: N real samples to N/2 bins, bin 0 is (DC, Nyquist)
extreme_run :: proc(plan: ^Extreme_FFT_Plan, samples: []f32, spectrum: []complex64) #no_bounds_check {
    assert(len(samples) == plan.size && len(spectrum) == plan.half)

    extreme_first_stage(plan.stages[0], raw_data(samples), plan.buffers[0])
    current := 0
    for stage in plan.stages[1:] {
        extreme_radix4_stage(stage, plan.buffers[current], plan.buffers[1 - current])
        current = 1 - current
    }

    out := cast([^]f32)raw_data(spectrum)
    if plan.last_radix == 2 {
        extreme_radix2_split(plan, plan.buffers[current], out)
    } else {
        extreme_radix4_split(plan, plan.buffers[current], out)
    }
}

// AI: N real samples to N/2 bins in two arrays, like vDSP's: re[k] and im[k], DC in re[0] and
// Nyquist in im[0]. Saves the split pairing re with im.
extreme_run_arrays :: proc(plan: ^Extreme_FFT_Plan, samples: []f32, re, im: []f32) #no_bounds_check {
    assert(len(samples) == plan.size && len(re) == plan.half && len(im) == plan.half)

    extreme_first_stage(plan.stages[0], raw_data(samples), plan.buffers[0])
    current := 0
    for stage in plan.stages[1:] {
        extreme_radix4_stage(stage, plan.buffers[current], plan.buffers[1 - current])
        current = 1 - current
    }

    if plan.last_radix == 2 {
        extreme_radix2_split_arrays(plan, plan.buffers[current], raw_data(re), raw_data(im))
    } else {
        extreme_radix4_split_arrays(plan, plan.buffers[current], raw_data(re), raw_data(im))
    }
}


// AI: the first stage, stride 1, four values of p at once. The input is the samples, each pair
// a complex point, and the four outputs of each butterfly land next to each other, so the four
// butterflies' outputs are transposed on the way out. Public for benchmark_stages.
extreme_first_stage :: proc(stage: Extreme_Stage, samples: [^]f32, dst: [2][]f32) #no_bounds_check {
    m := stage.n / 4
    tw := raw_data(stage.twiddles)
    out_re, out_im := raw_data(dst[0]), raw_data(dst[1])

    for p := 0; p < m; p += 4 {
        a_re, a_im := extreme_load_pairs(samples[2 * p:])
        b_re, b_im := extreme_load_pairs(samples[2 * (p + m):])
        c_re, c_im := extreme_load_pairs(samples[2 * (p + 2 * m):])
        d_re, d_im := extreme_load_pairs(samples[2 * (p + 3 * m):])

        y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im := extreme_butterfly4(a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im)
        y1_re, y1_im = extreme_cmul(y1_re, y1_im, extreme_load(tw[p:]), extreme_load(tw[m + p:]))
        y2_re, y2_im = extreme_cmul(y2_re, y2_im, extreme_load(tw[2 * m + p:]), extreme_load(tw[3 * m + p:]))
        y3_re, y3_im = extreme_cmul(y3_re, y3_im, extreme_load(tw[4 * m + p:]), extreme_load(tw[5 * m + p:]))

        extreme_store_transposed(out_re[4 * p:], y0_re, y1_re, y2_re, y3_re)
        extreme_store_transposed(out_im[4 * p:], y0_im, y1_im, y2_im, y3_im)
    }
}

// AI: a radix-4 stage with a stride of 4 or more, four values of q at once under one twiddle. The
// twiddles of p = 0 are all 1, its butterflies skip the multiplies.
extreme_radix4_stage :: proc(stage: Extreme_Stage, src, dst: [2][]f32) #no_bounds_check {
    m := stage.n / 4
    s := stage.stride
    quarter := s * m // the distance between a butterfly's inputs, N/8
    tw := raw_data(stage.twiddles)
    in_re, in_im := raw_data(src[0]), raw_data(src[1])
    out_re, out_im := raw_data(dst[0]), raw_data(dst[1])

    if s == 4 {
        // One block of q for each p: four p at a time, their twiddles six vector loads from the
        // table and each p's taken out of its lane, instead of six broadcast loads for every p.
        // p = 0's twiddles in the table are 1.
        for p := 0; p < m; p += 4 {
            w: [6]Extreme_F32x4
            #unroll for i in 0 ..< 6 do w[i] = extreme_load(tw[i * m + p:])
            extreme_radix4_lane(in_re, in_im, out_re, out_im, p, quarter, w, 0)
            extreme_radix4_lane(in_re, in_im, out_re, out_im, p, quarter, w, 1)
            extreme_radix4_lane(in_re, in_im, out_re, out_im, p, quarter, w, 2)
            extreme_radix4_lane(in_re, in_im, out_re, out_im, p, quarter, w, 3)
        }
        return
    }

    // p = 0 without multiplies, its own loop so the inner loop has no branch
    for q := 0; q < s; q += 4 {
        extreme_radix4_block(in_re, in_im, out_re, out_im, q, q, s, quarter, false, {}, {}, {}, {}, {}, {})
    }
    for p in 1 ..< m {
        w1_re, w1_im := extreme_splat(tw[p]), extreme_splat(tw[m + p])
        w2_re, w2_im := extreme_splat(tw[2 * m + p]), extreme_splat(tw[3 * m + p])
        w3_re, w3_im := extreme_splat(tw[4 * m + p]), extreme_splat(tw[5 * m + p])

        for q := 0; q < s; q += 4 {
            extreme_radix4_block(
                in_re, in_im, out_re, out_im, s * p + q, 4 * s * p + q, s, quarter,
                true, w1_re, w1_im, w2_re, w2_im, w3_re, w3_im,
            )
        }
    }
}

// AI: four butterflies of a radix-4 stage, inputs from first_in, outputs from first_out. TWIDDLED
// is a compile time constant, the copy for p = 0 has no multiplies.
@(private = "file")
extreme_radix4_block :: #force_inline proc "contextless" (
    in_re, in_im, out_re, out_im: [^]f32,
    first_in, first_out, s, quarter: int,
    $TWIDDLED: bool,
    w1_re, w1_im, w2_re, w2_im, w3_re, w3_im: Extreme_F32x4,
) #no_bounds_check {
    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im := extreme_butterfly4(
        extreme_load(in_re[first_in:]), extreme_load(in_im[first_in:]),
        extreme_load(in_re[first_in + quarter:]), extreme_load(in_im[first_in + quarter:]),
        extreme_load(in_re[first_in + 2 * quarter:]), extreme_load(in_im[first_in + 2 * quarter:]),
        extreme_load(in_re[first_in + 3 * quarter:]), extreme_load(in_im[first_in + 3 * quarter:]),
    )
    when TWIDDLED {
        y1_re, y1_im = extreme_cmul(y1_re, y1_im, w1_re, w1_im)
        y2_re, y2_im = extreme_cmul(y2_re, y2_im, w2_re, w2_im)
        y3_re, y3_im = extreme_cmul(y3_re, y3_im, w3_re, w3_im)
    }

    extreme_store(out_re[first_out:], y0_re); extreme_store(out_im[first_out:], y0_im)
    extreme_store(out_re[first_out + s:], y1_re); extreme_store(out_im[first_out + s:], y1_im)
    extreme_store(out_re[first_out + 2 * s:], y2_re); extreme_store(out_im[first_out + 2 * s:], y2_im)
    extreme_store(out_re[first_out + 3 * s:], y3_re); extreme_store(out_im[first_out + 3 * s:], y3_im)
}


/*
    AI: the split, the real input's spectrum out of Z, the half size FFT of the samples as complex
    points. With E the even samples' DFT and T the odd samples' DFT times W_N^k, both at k:

        E = (Z[k] + conj(Z[N/2 - k])) / 2
        T = -j·W_N^k·(Z[k] - conj(Z[N/2 - k])) / 2

        X[k]       = E + T
        X[N/2 - k] = conj(E - T)

    The second because the even and odd samples are real, so their DFTs at N/2 - k are the
    conjugates of those at k, and W_N^(N/2 - k) = -conj(W_N^k). The ½ is in the twiddles.

    Z itself isn't stored, the last FFT stage runs here: Z[q + j·s] is output j of the last
    stage's butterfly at q, s its stride.
*/

// AI: the split with a radix-2 last stage, into interleaved bins and into two arrays. Public for
// benchmark_stages.
extreme_radix2_split :: proc(plan: ^Extreme_FFT_Plan, x: [2][]f32, out: [^]f32) {
    extreme_radix2_split_into(plan, x, out, nil, false)
}

extreme_radix2_split_arrays :: proc(plan: ^Extreme_FFT_Plan, x: [2][]f32, re, im: [^]f32) {
    extreme_radix2_split_into(plan, x, re, im, true)
}

// AI: the split with a radix-4 last stage, into interleaved bins and into two arrays
extreme_radix4_split :: proc(plan: ^Extreme_FFT_Plan, x: [2][]f32, out: [^]f32) {
    extreme_radix4_split_into(plan, x, out, nil, false)
}

extreme_radix4_split_arrays :: proc(plan: ^Extreme_FFT_Plan, x: [2][]f32, re, im: [^]f32) {
    extreme_radix4_split_into(plan, x, re, im, true)
}

// AI: the split with a radix-2 last stage, s = N/4: Z[q] = x[q] + x[q+s], Z[q+s] = x[q] - x[q+s].
// With ARRAYS the bins go to out (real parts) and out_im, without into out interleaved.
@(private = "file")
extreme_radix2_split_into :: proc(plan: ^Extreme_FFT_Plan, x: [2][]f32, out, out_im: [^]f32, $ARRAYS: bool) #no_bounds_check {
    s := plan.last_stride
    half := plan.half
    x_re, x_im := raw_data(x[0]), raw_data(x[1])
    // In locals: the loop's stores through out could, as far as the compiler knows, change the
    // plan, and it reloaded these from the plan on every iteration
    twiddle_re, twiddle_im := raw_data(plan.split_re), raw_data(plan.split_im)

    for k in ([]int{0, 1, 2, 3, s}) do extreme_split_one(plan, x, out, out_im, ARRAYS, k)

    // Bins k to k+3 and N/2 - k down to N/2 - k - 3, which are Z[s + (s-k-l)] for lane l
    for k := 4; k < s; k += 4 {
        mirror := s - k - 3
        low_re := extreme_load(x_re[k:]) + extreme_load(x_re[k + s:])
        low_im := extreme_load(x_im[k:]) + extreme_load(x_im[k + s:])
        high_re := extreme_reverse(extreme_load(x_re[mirror:]) - extreme_load(x_re[mirror + s:]))
        high_im := extreme_reverse(extreme_load(x_im[mirror:]) - extreme_load(x_im[mirror + s:]))
        extreme_split_four(out, out_im, twiddle_re, twiddle_im, half, k, low_re, low_im, high_re, high_im, ARRAYS)
    }
}

// AI: the split with a radix-4 last stage, s = N/8. Z[q + j·s] is output j of the butterfly at q.
// The butterfly at q and the one at its mirror r = s - q - 3 together hold both bins of four
// pairs: k = q + j·s pairs with N/2 - k = (s - q) + (3 - j)·s, in the mirror's lanes backwards.
@(private = "file")
extreme_radix4_split_into :: proc(plan: ^Extreme_FFT_Plan, x: [2][]f32, out, out_im: [^]f32, $ARRAYS: bool) #no_bounds_check {
    s := plan.last_stride
    half := plan.half
    x_re, x_im := raw_data(x[0]), raw_data(x[1])
    // In locals, see extreme_radix2_split_into
    twiddle_re, twiddle_im := raw_data(plan.split_re), raw_data(plan.split_im)

    for q in ([]int{0, 1, 2, 3, s / 2, s - 3, s - 2, s - 1}) {
        extreme_split_one(plan, x, out, out_im, ARRAYS, q)
        extreme_split_one(plan, x, out, out_im, ARRAYS, q + s)
    }
    extreme_split_one(plan, x, out, out_im, ARRAYS, 2 * s)

    for q := 4; q <= s / 2 - 4; q += 4 {
        r := s - q - 3
        q0_re, q0_im, q1_re, q1_im, q2_re, q2_im, q3_re, q3_im := extreme_butterfly4(
            extreme_load(x_re[q:]), extreme_load(x_im[q:]),
            extreme_load(x_re[q + s:]), extreme_load(x_im[q + s:]),
            extreme_load(x_re[q + 2 * s:]), extreme_load(x_im[q + 2 * s:]),
            extreme_load(x_re[q + 3 * s:]), extreme_load(x_im[q + 3 * s:]),
        )
        r0_re, r0_im, r1_re, r1_im, r2_re, r2_im, r3_re, r3_im := extreme_butterfly4(
            extreme_load(x_re[r:]), extreme_load(x_im[r:]),
            extreme_load(x_re[r + s:]), extreme_load(x_im[r + s:]),
            extreme_load(x_re[r + 2 * s:]), extreme_load(x_im[r + 2 * s:]),
            extreme_load(x_re[r + 3 * s:]), extreme_load(x_im[r + 3 * s:]),
        )

        extreme_split_four(out, out_im, twiddle_re, twiddle_im, half, q, q0_re, q0_im, extreme_reverse(r3_re), extreme_reverse(r3_im), ARRAYS)
        extreme_split_four(out, out_im, twiddle_re, twiddle_im, half, q + s, q1_re, q1_im, extreme_reverse(r2_re), extreme_reverse(r2_im), ARRAYS)
        extreme_split_four(out, out_im, twiddle_re, twiddle_im, half, r, r0_re, r0_im, extreme_reverse(q3_re), extreme_reverse(q3_im), ARRAYS)
        extreme_split_four(out, out_im, twiddle_re, twiddle_im, half, r + s, r1_re, r1_im, extreme_reverse(q2_re), extreme_reverse(q2_im), ARRAYS)
    }
}

// AI: bins k to k+3 out of low, Z at k to k+3, and bins N/2 - k down to N/2 - k - 3 out of high,
// Z at N/2 - k down to N/2 - k - 3. With ARRAYS into out and out_im, without into out interleaved.
@(private = "file")
extreme_split_four :: #force_inline proc "contextless" (
    out, out_im, twiddle_re, twiddle_im: [^]f32,
    half, k: int,
    low_re, low_im, high_re, high_im: Extreme_F32x4,
    $ARRAYS: bool,
) {
    HALF :: Extreme_F32x4{0.5, 0.5, 0.5, 0.5}
    w_re, w_im := extreme_load(twiddle_re[k:]), extreme_load(twiddle_im[k:])

    // E = (sum_re + j·diff_im) / 2, and -j·T = odd = the twiddle times (diff_re + j·sum_im)
    sum_re, diff_im := low_re + high_re, low_im - high_im
    diff_re, sum_im := low_re - high_re, low_im + high_im
    odd_im := simd.fma(diff_re, w_im, sum_im * w_re)
    // -odd_re directly, the minus on a multiplicand like in extreme_cmul, both bins' imaginary
    // parts are ±½·diff_im minus odd_re, two multiply-adds and no negate
    minus_odd_re := simd.fma(-diff_re, w_re, sum_im * w_im)
    even_re := HALF * sum_re

    // X[k] = E + T, ascending from k
    x_re := even_re + odd_im
    x_im := simd.fma(HALF, diff_im, minus_odd_re)

    // X[N/2 - k] = conj(E - T), descending from N/2 - k in the lanes
    m_re := even_re - odd_im
    m_im := simd.fma(-HALF, diff_im, minus_odd_re)
    mirror := half - k - 3

    when ARRAYS {
        extreme_store(out[k:], x_re)
        extreme_store(out_im[k:], x_im)
        extreme_store(out[mirror:], extreme_reverse(m_re))
        extreme_store(out_im[mirror:], extreme_reverse(m_im))
    } else {
        extreme_store(out[2 * k:], simd.shuffle(x_re, x_im, 0, 4, 1, 5))
        extreme_store(out[2 * k + 4:], simd.shuffle(x_re, x_im, 2, 6, 3, 7))
        // Stored ascending from N/2 - k - 3, lane 3 first: the turning around in the same
        // shuffle that pairs re with im
        extreme_store(out[2 * mirror:], simd.shuffle(m_re, m_im, 3, 7, 2, 6))
        extreme_store(out[2 * mirror + 4:], simd.shuffle(m_re, m_im, 1, 5, 0, 4))
    }
}

// AI: bin k and bin N/2 - k one at a time, for the bins the vector loops don't reach. Bin 0 is
// (DC, Nyquist), both real. With arrays into out and out_im, without into out interleaved.
@(private = "file")
extreme_split_one :: proc(plan: ^Extreme_FFT_Plan, x: [2][]f32, out, out_im: [^]f32, arrays: bool, k: int) #no_bounds_check {
    // Bin b's real and imaginary parts
    bin_at := proc(out, out_im: [^]f32, arrays: bool, b: int) -> (^f32, ^f32) {
        if arrays do return &out[b], &out_im[b]
        return &out[2 * b], &out[2 * b + 1]
    }

    if k == 0 {
        z := extreme_z_at(plan, x, 0)
        dc, nyquist := bin_at(out, out_im, arrays, 0)
        dc^ = real(z) + imag(z)
        nyquist^ = real(z) - imag(z)
        return
    }

    low := extreme_z_at(plan, x, k)
    high := extreme_z_at(plan, x, plan.half - k)
    twiddle := complex(plan.split_re[k], plan.split_im[k])

    even := complex(0.5 * (real(low) + real(high)), 0.5 * (imag(low) - imag(high)))
    odd := twiddle * complex(real(low) - real(high), imag(low) + imag(high))

    low_re, low_im := bin_at(out, out_im, arrays, k)
    low_re^ = real(even) + imag(odd)
    low_im^ = imag(even) - real(odd)
    high_re, high_im := bin_at(out, out_im, arrays, plan.half - k)
    high_re^ = real(even) - imag(odd)
    high_im^ = -imag(even) - real(odd)
}

// AI: one point of Z, output j of the last stage's butterfly at q, for bin = q + j·s
@(private = "file")
extreme_z_at :: proc(plan: ^Extreme_FFT_Plan, x: [2][]f32, bin: int) -> complex64 #no_bounds_check {
    s := plan.last_stride
    q, j := bin & (s - 1), bin >> plan.last_shift // s a power of two, no division
    a := complex(x[0][q], x[1][q])
    b := complex(x[0][q + s], x[1][q + s])
    if plan.last_radix == 2 do return a + b if j == 0 else a - b

    c := complex(x[0][q + 2 * s], x[1][q + 2 * s])
    d := complex(x[0][q + 3 * s], x[1][q + 3 * s])
    J :: complex64(1i)
    switch j {
    case 0: return (a + c) + (b + d)
    case 1: return (a - c) - J * (b - d)
    case 2: return (a + c) - (b + d)
    case:   return (a - c) + J * (b - d)
    }
}


// AI: the radix-4 butterfly, a 4-point DFT without twiddles
//     X0 = (a + c) + (b + d)      X1 = (a - c) - j(b - d)
//     X2 = (a + c) - (b + d)      X3 = (a - c) + j(b - d)
@(private = "file")
extreme_butterfly4 :: #force_inline proc "contextless" (
    a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im: Extreme_F32x4,
) -> (
    y0_re, y0_im, y1_re, y1_im, y2_re, y2_im, y3_re, y3_im: Extreme_F32x4,
) {
    a_plus_c_re, a_plus_c_im := a_re + c_re, a_im + c_im
    a_minus_c_re, a_minus_c_im := a_re - c_re, a_im - c_im
    b_plus_d_re, b_plus_d_im := b_re + d_re, b_im + d_im
    b_minus_d_re, b_minus_d_im := b_re - d_re, b_im - d_im

    y0_re, y0_im = a_plus_c_re + b_plus_d_re, a_plus_c_im + b_plus_d_im
    y1_re, y1_im = a_minus_c_re + b_minus_d_im, a_minus_c_im - b_minus_d_re
    y2_re, y2_im = a_plus_c_re - b_plus_d_re, a_plus_c_im - b_plus_d_im
    y3_re, y3_im = a_minus_c_re - b_minus_d_im, a_minus_c_im + b_minus_d_re
    return
}

// AI: (x_re + j·x_im)(w_re + j·w_im), four at once, with fused multiply-adds. The minus on a
// multiplicand, not on the product, so it folds into a multiply-subtract (fmls) instead of
// costing a negate of its own.
@(private = "file")
extreme_cmul :: #force_inline proc "contextless" (x_re, x_im, w_re, w_im: Extreme_F32x4) -> (Extreme_F32x4, Extreme_F32x4) {
    return simd.fma(-x_im, w_im, x_re * w_re), simd.fma(x_re, w_im, x_im * w_re)
}

// AI: four complex points stored as pairs, split into their real and imaginary parts. One 8 wide
// load and two strided shuffles of it, the shape LLVM turns into a single ld2 on NEON, which
// splits the pairs in the load unit instead of with shuffles.
@(private = "file")
extreme_load_pairs :: #force_inline proc "contextless" (p: [^]f32) -> (re, im: Extreme_F32x4) {
    pairs := intrinsics.unaligned_load(cast(^#simd[8]f32)p)
    return simd.shuffle(pairs, pairs, 0, 2, 4, 6), simd.shuffle(pairs, pairs, 1, 3, 5, 7)
}

// AI: rows r0..r3 stored as columns, 16 floats from dst. The four rows joined and interleaved in
// one 16 wide shuffle and one store, the shape LLVM turns into a single st4 on NEON.
@(private = "file")
extreme_store_transposed :: #force_inline proc "contextless" (dst: [^]f32, r0, r1, r2, r3: Extreme_F32x4) {
    r01 := simd.shuffle(r0, r1, 0, 1, 2, 3, 4, 5, 6, 7)
    r23 := simd.shuffle(r2, r3, 0, 1, 2, 3, 4, 5, 6, 7)
    columns := simd.shuffle(r01, r23, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15)
    intrinsics.unaligned_store(cast(^#simd[16]f32)dst, columns)
}

// AI: the stride 4 stage's block at p + LANE, its twiddles in lane LANE of the four p's. The same
// with a loop over q for stride 16 made both stages slower, 0.33 to 1.13 µs at stride 4 and
// 0.33 to 0.52 µs at stride 16 for N = 4096, so it's stride 4 only.
@(private = "file")
extreme_radix4_lane :: #force_inline proc "contextless" (
    in_re, in_im, out_re, out_im: [^]f32,
    p, quarter: int,
    w: [6]Extreme_F32x4,
    $LANE: int,
) {
    extreme_radix4_block(
        in_re, in_im, out_re, out_im, 4 * (p + LANE), 16 * (p + LANE), 4, quarter, true,
        extreme_lane(w[0], LANE), extreme_lane(w[1], LANE),
        extreme_lane(w[2], LANE), extreme_lane(w[3], LANE),
        extreme_lane(w[4], LANE), extreme_lane(w[5], LANE),
    )
}

// AI: one lane in all four, multiplies by it become NEON's multiply by element
@(private = "file")
extreme_lane :: #force_inline proc "contextless" (v: Extreme_F32x4, $LANE: int) -> Extreme_F32x4 {
    return simd.swizzle(v, LANE, LANE, LANE, LANE)
}

// AI: the four lanes in reverse order
@(private = "file")
extreme_reverse :: #force_inline proc "contextless" (v: Extreme_F32x4) -> Extreme_F32x4 {
    return simd.swizzle(v, 3, 2, 1, 0)
}

@(private = "file")
extreme_load :: #force_inline proc "contextless" (p: [^]f32) -> Extreme_F32x4 {
    return intrinsics.unaligned_load(cast(^Extreme_F32x4)p)
}

@(private = "file")
extreme_store :: #force_inline proc "contextless" (p: [^]f32, v: Extreme_F32x4) {
    intrinsics.unaligned_store(cast(^Extreme_F32x4)p, v)
}

@(private = "file")
extreme_splat :: #force_inline proc "contextless" (x: f32) -> Extreme_F32x4 {
    return Extreme_F32x4{x, x, x, x}
}

@(private = "file")
extreme_make_aligned :: proc(count: int) -> []f32 {
    slice, _ := runtime.make_aligned([]f32, count, 16)
    return slice
}
