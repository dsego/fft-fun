package fft

import "core:math"
import "core:testing"


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
