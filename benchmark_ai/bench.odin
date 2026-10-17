package bench_ai

/*
    ADDED BY ARTIFICIAL INTELLIGENCE: written by Claude, an AI model made by Anthropic, on
    2026-10-16. A separate benchmark for fft/fft_ai.odin, benchmark/bench.odin is unchanged.

    Checks the AI FFT against pffft, Apple's vDSP and the radix-2 plan, then times all of them on
    the same signal, best of 5 runs. Run it from the repository root, next to pffft.dylib:

        odin run benchmark_ai -o:speed -microarch:native
*/

import "base:runtime"
import "core:fmt"
import "core:math"
import "core:time"
import "../external/pffft"
import "../fft"

ITERATIONS :: 20_000
RUNS :: 5

main :: proc() {
    fmt.printf("pffft SIMD width: %v\n\n", pffft.pffft_simd_size())

    for size in ([]int{1024, 4096, 8192}) {
        samples := make_aligned(size)
        defer delete(samples)
        for &sample, i in samples {
            t := f32(i)
            sample = 0.6 * math.sin(0.031 * t) + 0.3 * math.sin(0.47 * t + 1) + 0.1 * math.cos(2.9 * t)
        }

        // pffft ordered, the reference
        setup := pffft.pffft_new_setup(size, .PFFFT_REAL)
        defer pffft.pffft_destroy_setup(setup)
        reference := make_aligned(size)
        defer delete(reference)
        unordered := make_aligned(size)
        defer delete(unordered)
        work := make_aligned(size)
        defer delete(work)
        pffft.pffft_transform_ordered(setup, raw_data(samples), raw_data(reference), raw_data(work), .PFFFT_FORWARD)

        // AI FFT
        plan := fft.ai_create_real_fft_plan(size)
        defer fft.ai_destroy_real_fft_plan(&plan)
        spectrum := make([]complex64, size / 2)
        defer delete(spectrum)
        fft.ai_run_real_fft(&plan, samples, spectrum)

        // Radix-2 plan, N complex bins out of the real samples
        radix2 := fft.create_fft_plan(uint(size))
        defer fft.destroy_fft_plan(radix2)
        fft.run_fft_plan(radix2, samples)

        // vDSP, the classic FFT and the DFT interface, into split arrays
        log2n := uint(0)
        for 1 << log2n < size do log2n += 1
        vdsp_fft := vDSP_create_fftsetup(log2n, FFT_RADIX_2)
        defer vDSP_destroy_fftsetup(vdsp_fft)
        vdsp_dft := vDSP_DFT_zrop_CreateSetup(nil, uint(size), DFT_FORWARD)
        defer vDSP_DFT_DestroySetup(vdsp_dft)
        even, odd := make_aligned(size / 2), make_aligned(size / 2)
        defer delete(even)
        defer delete(odd)
        vdsp_re, vdsp_im := make_aligned(size / 2), make_aligned(size / 2)
        defer delete(vdsp_re)
        defer delete(vdsp_im)

        vdsp_fft_run(samples, even, odd, vdsp_fft, log2n)
        fft_error := vdsp_error(reference, even, odd)
        vdsp_dft_run(samples, even, odd, vdsp_re, vdsp_im, vdsp_dft)
        dft_error := vdsp_error(reference, vdsp_re, vdsp_im)

        ai := ([^]f32)(raw_data(spectrum))[:size]
        peak, ai_error, radix2_error: f32
        for value in reference do peak = max(peak, abs(value))
        for value, i in reference do ai_error = max(ai_error, abs(ai[i] - value))
        radix2_error = abs(real(radix2.buffer[0]) - reference[0])
        radix2_error = max(radix2_error, abs(real(radix2.buffer[size / 2]) - reference[1]))
        for k in 1 ..< size / 2 {
            radix2_error = max(radix2_error, abs(real(radix2.buffer[k]) - reference[2 * k]))
            radix2_error = max(radix2_error, abs(imag(radix2.buffer[k]) - reference[2 * k + 1]))
        }

        fmt.printf("N = %v, largest bin %.1f\n", size, peak)
        fmt.printf("  error vs pffft:  AI %.2e, radix-2 %.2e\n", ai_error / peak, radix2_error / peak)
        fmt.printf("                   vDSP FFT %.2e, vDSP DFT %.2e\n", fft_error / peak, dft_error / peak)

        fmt.printf("  radix-2:         % 8.3f µs\n", best_of(proc(data: rawptr) {
            p := cast(^Args)data
            fft.run_fft_plan(p.radix2^, p.samples)
        }, &Args{samples = samples, radix2 = &radix2}))
        fmt.printf("  AI FFT:          % 8.3f µs\n", best_of(proc(data: rawptr) {
            p := cast(^Args)data
            fft.ai_run_real_fft(p.plan, p.samples, p.spectrum)
        }, &Args{samples = samples, plan = &plan, spectrum = spectrum}))
        fmt.printf("  pffft ordered:   % 8.3f µs\n", best_of(proc(data: rawptr) {
            p := cast(^Args)data
            pffft.pffft_transform_ordered(p.setup, raw_data(p.samples), raw_data(p.out), raw_data(p.work), .PFFFT_FORWARD)
        }, &Args{samples = samples, setup = setup, out = reference, work = work}))
        fmt.printf("  pffft unordered: % 8.3f µs\n", best_of(proc(data: rawptr) {
            p := cast(^Args)data
            pffft.pffft_transform(p.setup, raw_data(p.samples), raw_data(p.out), raw_data(p.work), .PFFFT_FORWARD)
        }, &Args{samples = samples, setup = setup, out = unordered, work = work}))
        fmt.printf("  vDSP FFT:        % 8.3f µs\n", best_of(proc(data: rawptr) {
            p := cast(^Args)data
            vdsp_fft_run(p.samples, p.out, p.work, p.vdsp_setup, p.log2n)
        }, &Args{samples = samples, out = even, work = odd, vdsp_setup = vdsp_fft, log2n = log2n}))
        fmt.printf("  vDSP DFT:        % 8.3f µs\n\n", best_of(proc(data: rawptr) {
            p := cast(^Args)data
            vdsp_dft_run(p.samples, p.out, p.work, p.vdsp_re, p.vdsp_im, p.vdsp_setup)
        }, &Args{samples = samples, out = even, work = odd, vdsp_re = vdsp_re, vdsp_im = vdsp_im, vdsp_setup = vdsp_dft}))
    }
}

// The samples split into even and odd ones, then vDSP's FFT in place on them
vdsp_fft_run :: proc(samples, even, odd: []f32, setup: rawptr, log2n: uint) {
    split := DSPSplitComplex{raw_data(even), raw_data(odd)}
    vDSP_ctoz(raw_data(samples), 2, &split, 1, uint(len(even)))
    vDSP_fft_zrip(setup, &split, 1, log2n, FFT_FORWARD)
}

// The samples split into even and odd ones, then vDSP's DFT out of place
vdsp_dft_run :: proc(samples, even, odd, re, im: []f32, setup: rawptr) {
    split := DSPSplitComplex{raw_data(even), raw_data(odd)}
    vDSP_ctoz(raw_data(samples), 2, &split, 1, uint(len(even)))
    vDSP_DFT_Execute(setup, raw_data(even), raw_data(odd), raw_data(re), raw_data(im))
}

// vDSP's result, twice the DFT in two arrays, against pffft's interleaved one
vdsp_error :: proc(reference, re, im: []f32) -> (error: f32) {
    for k in 0 ..< len(re) {
        error = max(error, abs(0.5 * re[k] - reference[2 * k]), abs(0.5 * im[k] - reference[2 * k + 1]))
    }
    return
}

Args :: struct {
    samples:  []f32,
    radix2:   ^fft.FFT_Plan,
    plan:     ^fft.AI_Real_FFT_Plan,
    spectrum: []complex64,
    setup:    rawptr,
    out:      []f32,
    work:     []f32,
    vdsp_setup: rawptr,
    log2n:      uint,
    vdsp_re:    []f32,
    vdsp_im:    []f32,
}

// Microseconds per call, the fastest of RUNS runs of ITERATIONS calls after a warm-up
best_of :: proc(run: proc(data: rawptr), data: rawptr) -> f64 {
    for _ in 0 ..< 1000 do run(data)
    best := max(f64)
    for _ in 0 ..< RUNS {
        stopwatch: time.Stopwatch
        time.stopwatch_start(&stopwatch)
        for _ in 0 ..< ITERATIONS do run(data)
        time.stopwatch_stop(&stopwatch)
        best = min(best, time.duration_microseconds(time.stopwatch_duration(stopwatch)) / ITERATIONS)
    }
    return best
}

// pffft wants 16 byte aligned buffers
make_aligned :: proc(count: int) -> []f32 {
    slice, _ := runtime.make_aligned([]f32, count, 16)
    return slice
}
