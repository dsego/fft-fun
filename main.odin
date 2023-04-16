package main

import "core:testing"
import "core:math"
import "core:math/bits"
import "core:fmt"
import "core:time"

main :: proc() {
    // fmt.println("Hello World!")

    // b : uint =
    // fmt.println(b >> 1)
    // reversed := bits.reverse_bits(b)
    // fmt.println(size_of(uint))
    // fmt.println(4 >> 1)

    // start := time.Stopwatch{}
    // time.stopwatch_start(&start)
    // time.stopwatch_stop(&start)
    // duration := time.stopwatch_duration(start)
    // ms := time.duration_milliseconds(duration)
    // fmt.printf("{}ms\n", ms)

    samples: [8]f32
    generate_samples(samples[:])

    plan := create_fft_plan(8)
    defer destroy_fft_plan(plan)

    run_fft_plan(plan, samples[:])

    fmt.println(plan.buffer)
    // fmt.println(scrambled)
    fmt.println(expected)
}


// ¯\_(ツ)_/¯
// https://stackoverflow.com/a/34236981/156372
reverse_bits :: proc(value: uint, k: uint) -> (result: uint = 0) {
    for i in 0..<k {
        result |= ((value >> i) & 1) << (k - i - 1)
    }
    return result
}


// Decimation-in-time radix-2 FFT
FFT_Plan :: struct {
    fft_size: uint,
    buffer: []complex64,
    twiddle_lookup: []complex64,
}


create_fft_plan :: proc(fft_size: uint) -> (plan: FFT_Plan) {
    assert(bits.is_power_of_two(fft_size))

    plan.buffer = make([]complex64, fft_size)
    plan.twiddle_lookup = make([]complex64, fft_size)
    plan.fft_size = fft_size
    phase_delta := 2.0 * math.PI / f32(fft_size)

    for i in 0..<fft_size {
        phase := phase_delta * f32(i)
        plan.twiddle_lookup[i] = complex(math.cos(phase), -math.sin(phase))
    }
    return
}

destroy_fft_plan :: proc(plan: FFT_Plan) {
    delete(plan.buffer)
    delete(plan.twiddle_lookup)
}

/*
    Radix R:
        Run decimation in stages, r inputs and r outputs, N/R butterflies per stage, log_r(N) stages.
*/
run_fft_plan :: proc(plan: FFT_Plan, samples:[]f32) {
    // copy samples over to internal buffer with scrambled (bit reversed) indexes
    for sample, i in samples {
        j := reverse_bits(uint(i), bits.log2(plan.fft_size))
        plan.buffer[j] = complex(sample, 0)
    }

    RADIX :: uint(2)
    stride := uint(1)
    group_increment := RADIX
    twiddle_increment := plan.fft_size / RADIX
    butterfly_count := uint(1)

    // log_r(N) stages
    for stride < plan.fft_size {
        group := uint(0)
        for group < plan.fft_size {
            k := uint(0)
            for b in 0..<butterfly_count {
                butterfly(
                    &plan.buffer[group+b],
                    &plan.buffer[group+b+stride],
                    plan.twiddle_lookup[k],
                )
                k += twiddle_increment
            }
            group += group_increment
        }
        butterfly_count *= RADIX
        twiddle_increment /= RADIX
        group_increment *= RADIX
        stride *= RADIX
    }
}


/*
    optimized radix-2 butterfly structure
    x -----------\   /----------> x'
                  \ / +
                   o
                  / \ -
    y -- w^k/N --/   \----------> y'
*/
butterfly :: proc(x: ^complex64, y: ^complex64, w: complex64) {
    a := x^ + w * y^
    b := x^ - w * y^
    x^ = a
    y^ = b
}

naive_dft :: proc(
    dft: []complex64,
    samples: []f32,
    dft_size: uint
) {
    phase_angle:f32 = 2 * math.PI / f32(dft_size)

    for freq in 0..<dft_size {
        for sample, i in samples {
            time := phase_angle * f32(i)

            // Fourier formula: cos(2πft) - i×sin(2πft)
            ft := f32(freq) * time
            re := sample * math.cos(ft)
            im := -sample * math.sin(ft)
            dft[freq] = dft[freq] + complex(re, im)
        }
    }
}

// for real input the DFT output is symmetrical
naive_dft_real :: proc(
    dft: []complex64,
    samples: []f32,
    dft_size: uint
) {
    phase_angle:f32 = 2 * math.PI / f32(dft_size)
    real_size := dft_size % 2 == 0 ? dft_size / 2 : (dft_size+1) / 2

    for freq in 0..<real_size {
        for sample, i in samples {
            time := phase_angle * f32(i)

            // Fourier formula: cos(2πft) - i×sin(2πft)
            ft := f32(freq) * time
            re := sample * math.cos(ft)
            im := -sample * math.sin(ft)
            dft[freq] = dft[freq] + complex(re, im)
        }
        // exploit DFT symmetry
        if freq > 0 {
            dft[dft_size - freq] = conj(dft[freq])
        }
    }
}


// 1kHz + 2kHz + π/2 phase
generate_samples :: proc(data: []f32) {
    SAMPLERATE :: 8000.00
    for _, i in data {
        k := 2.0 * math.PI * f32(i) / SAMPLERATE
        data[i] = math.sin(k * 1000) + math.sin(k * 2000 + 0.5 * math.PI)
    }
}

approx_eq :: proc(x: f32, y: f32) -> bool {
    tolerance :: 0.00001
    return abs(x - y) <= tolerance
}


expected : []complex64 = {
    complex(0, 0),
    complex(0,-4),
    complex(4, 0),
    complex(0, 0),
    complex(0, 0),
    complex(0, 0),
    complex(4, 0),
    complex(0, 4),
}

@(test)
test_naive_dft :: proc(^testing.T) {
    dft: [8]complex64
    samples: [8]f32
    generate_samples(samples[:])
    naive_dft(dft[:], samples[:], 8)
    for _, i in dft {
        assert(approx_eq(real(dft[i]), real(expected[i])))
        assert(approx_eq(imag(dft[i]), imag(expected[i])))
    }
}

@(test)
test_naive_dft_real :: proc(^testing.T) {
    dft: [8]complex64
    samples: [8]f32
    generate_samples(samples[:])
    naive_dft_real(dft[:], samples[:], 8)
    for _, i in dft {
        assert(approx_eq(real(dft[i]), real(expected[i])))
        assert(approx_eq(imag(dft[i]), imag(expected[i])))
    }
}

@(test)
test_fft :: proc(^testing.T) {
    samples: [8]f32
    generate_samples(samples[:])

    plan := create_fft_plan(8)
    defer destroy_fft_plan(plan)

    run_fft_plan(plan, samples[:])

    for _, i in plan.buffer {
        assert(approx_eq(real(plan.buffer[i]), real(expected[i])))
        assert(approx_eq(imag(plan.buffer[i]), imag(expected[i])))
    }
}


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


// NOTE: This is just a learning artifact
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
    butterfly(&plan.buffer[0], &plan.buffer[1], plan.twiddle_lookup[0])
    // --
    butterfly(&plan.buffer[2], &plan.buffer[3], plan.twiddle_lookup[0])
    // --
    butterfly(&plan.buffer[4], &plan.buffer[5], plan.twiddle_lookup[0])
    // --
    butterfly(&plan.buffer[6], &plan.buffer[7], plan.twiddle_lookup[0])


    // stage 1, stride 2
    butterfly(&plan.buffer[0], &plan.buffer[2], plan.twiddle_lookup[0])
    butterfly(&plan.buffer[1], &plan.buffer[3], plan.twiddle_lookup[2])
    // --
    butterfly(&plan.buffer[4], &plan.buffer[6], plan.twiddle_lookup[0])
    butterfly(&plan.buffer[5], &plan.buffer[7], plan.twiddle_lookup[2])

    // stage 2, stride 4
    butterfly(&plan.buffer[0], &plan.buffer[4], plan.twiddle_lookup[0])
    butterfly(&plan.buffer[1], &plan.buffer[5], plan.twiddle_lookup[1])
    butterfly(&plan.buffer[2], &plan.buffer[6], plan.twiddle_lookup[2])
    butterfly(&plan.buffer[3], &plan.buffer[7], plan.twiddle_lookup[3])
}
