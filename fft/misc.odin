package fft

import "core:math/bits"

/*
    optimized butterfly structure

    x ------------\   /-----------> x'
                   \ / +
                    /
                   / \ -
    y --- w^k/N --/   \-----------> y'

    x + w * y
    x - w * y

    Radix R:
        r inputs and r outputs, N/R butterflies per stage, log_r(N) stages
        span = (radix - 1) * stride

    Run decimation in stages - eg. for 8-point DFT we get 3 stages:

        STAGE 1 butterflies | twiddles
        -------------------------------------
         x[0] >< x[4]       |  w0/2 = w0/8
         x[2] >< x[6]       |  w0/2 = w0/8
         x[1] >< x[5]       |  w0/2 = w0/8
         x[3] >< x[7]       |  w0/2 = w0/8

        STAGE 2 butterflies
        -------------------------------------
         x[0] >< x[2]       |  w0/4 = w0/8
         x[4] >< x[6]       |  w1/4 = w2/8
        -------------------------------------
         x[1] >< x[3]       |  w0/4 = w0/8
         x[5] >< x[7]       |  w1/4 = w2/8

        STAGE 3 butterflies
        -------------------------------------
         x[0] >< x[1]       |  w0/8
        -------------------------------------
         x[2] >< x[3]       |  w2/8
        -------------------------------------
         x[4] >< x[5]       |  w1/8
        -------------------------------------
         x[6] >< x[7]       |  w3/8


    Twiddle factors for 8-point FFT
    -------------------------------
        w[0] ( 1.000, -0.000)
        w[1] ( 0.707, -0.707)
        w[2] (-0.000, -1.000)
        w[3] (-0.707, -0.707)
        w[4] (-1.000,  0.000)
        w[5] (-0.707,  0.707)
        w[6] ( 0.000,  1.000)
        w[7] ( 0.707,  0.707)
*/


// NOTE: This is just a learning/debugging artifact
_run_8_butterflies :: proc(plan: FFT_Plan, samples:[]f32) {
    RADIX :: uint(2)
    stride := uint(1)
    group_increment := RADIX
    twiddle_increment := plan.fft_size / RADIX

    // copy samples over to internal complex buffer with scrambled indexes
    for sample, i in samples {
        j := reverse_bits(uint(i), bits.log2(plan.fft_size))
        plan.buffer[j] = complex(sample, 0)
    }

    /* This is what it would look like with unscrambled input
        stage 0
        butterfly(&plan.buffer[0], &plan.buffer[4], plan.twiddle_lookup[0])
        butterfly(&plan.buffer[1], &plan.buffer[5], plan.twiddle_lookup[0])
        butterfly(&plan.buffer[2], &plan.buffer[6], plan.twiddle_lookup[0])
        butterfly(&plan.buffer[3], &plan.buffer[7], plan.twiddle_lookup[0])


        stage 1
        butterfly(&plan.buffer[0], &plan.buffer[2], plan.twiddle_lookup[0])
        butterfly(&plan.buffer[4], &plan.buffer[6], plan.twiddle_lookup[2])
        butterfly(&plan.buffer[1], &plan.buffer[3], plan.twiddle_lookup[0])
        butterfly(&plan.buffer[5], &plan.buffer[7], plan.twiddle_lookup[2])

        stage 2
        butterfly(&plan.buffer[0], &plan.buffer[1], plan.twiddle_lookup[0])
        butterfly(&plan.buffer[4], &plan.buffer[5], plan.twiddle_lookup[1])
        butterfly(&plan.buffer[2], &plan.buffer[3], plan.twiddle_lookup[2])
        butterfly(&plan.buffer[6], &plan.buffer[7], plan.twiddle_lookup[3])
    */


    // stage 0, stride 1
    radix2_butterfly(&plan.buffer[0], &plan.buffer[1], plan.twiddle_lookup[0])
    // --
    radix2_butterfly(&plan.buffer[2], &plan.buffer[3], plan.twiddle_lookup[0])
    // --
    radix2_butterfly(&plan.buffer[4], &plan.buffer[5], plan.twiddle_lookup[0])
    // --
    radix2_butterfly(&plan.buffer[6], &plan.buffer[7], plan.twiddle_lookup[0])


    // stage 1, stride 2
    radix2_butterfly(&plan.buffer[0], &plan.buffer[2], plan.twiddle_lookup[0])
    radix2_butterfly(&plan.buffer[1], &plan.buffer[3], plan.twiddle_lookup[2])
    // --
    radix2_butterfly(&plan.buffer[4], &plan.buffer[6], plan.twiddle_lookup[0])
    radix2_butterfly(&plan.buffer[5], &plan.buffer[7], plan.twiddle_lookup[2])

    // stage 2, stride 4
    radix2_butterfly(&plan.buffer[0], &plan.buffer[4], plan.twiddle_lookup[0])
    radix2_butterfly(&plan.buffer[1], &plan.buffer[5], plan.twiddle_lookup[1])
    radix2_butterfly(&plan.buffer[2], &plan.buffer[6], plan.twiddle_lookup[2])
    radix2_butterfly(&plan.buffer[3], &plan.buffer[7], plan.twiddle_lookup[3])
}


/*
    Fast complex multiplication with intermediate values:
        R + jI = (a + jb)(c + jd) = (ac - bd) + j(bc + ad)
        k1 = a(c + d)
        k2 = d(a + b)
        k3 = c(b - a)
        R = k1 - k2
        I = k1 + k3
*/
fast_cpx_mul :: proc(x: complex64, y: complex64) -> complex64 {
    a, b := real(x), imag(x)
    c, d := real(y), imag(y)
    k1 := a * (c + d)
    k2 := d * (a + b)
    k3 := c * (b - a)
    r := k1 - k1
    j := k1 + k3
    return complex(r, j)
}

