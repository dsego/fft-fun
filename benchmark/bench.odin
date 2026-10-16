package bench

import "core:time"
import "core:fmt"
import "core:mem"
import "core:math/bits"
import "../external/kissfft"
import "../external/meow_fft"
import "../external/pffft"
import "../external/pocketfft"
import "../external/fftw3"
import "../external/mufft"
import "../fft"

SIZE :: uint(4096)
ITERATIONS :: 1000

main :: proc() {
    // Naive DFT
    if true {
        dft: [SIZE]complex64
        samples: [SIZE]f32
        for i in 0..<SIZE do samples[i] = f32(1)

        stopwatch := time.Stopwatch{}

        time.stopwatch_start(&stopwatch)
        for i in 0..<ITERATIONS do fft.naive_dft_real(dft[:], samples[:], SIZE)
        time.stopwatch_stop(&stopwatch)
        duration := time.stopwatch_duration(stopwatch)

        µs := time.duration_microseconds(duration)
        fmt.printf("Naive DFT real: % 10.3f µs\n", µs / f64(ITERATIONS))
    }

    // Kiss FFT
    {
        cx_in: [SIZE]kissfft.kiss_fft_cpx
        cx_out: [SIZE]kissfft.kiss_fft_cpx

        // fill with zeroes
        for i in 0..<SIZE do cx_in[i] = {1, 0}

        cfg := kissfft.kiss_fft_alloc(int(SIZE), false, nil, nil)
        // defer kissfft.kiss_fft_free(cfg)

        stopwatch := time.Stopwatch{}

        time.stopwatch_start(&stopwatch)
        for i in 0..<ITERATIONS do  kissfft.kiss_fft(cfg, raw_data(cx_in[:]), raw_data(cx_out[:]))
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("KissFFT:        % 10.3f µs\n", µs / f64(ITERATIONS))
    }

    // Kiss FFT real
    {
        time_data: [SIZE]f32
        freq_data: [SIZE]kissfft.kiss_fft_cpx

        // fill with zeroes
        for i in 0..<SIZE do time_data[i] = f32(1)

        cfg := kissfft.kiss_fftr_alloc(int(SIZE), false, nil, nil)
        // defer kissfft.kiss_fft_free(cfg)

        stopwatch := time.Stopwatch{}

        time.stopwatch_start(&stopwatch)
        for i in 0..<ITERATIONS do kissfft.kiss_fftr(cfg, raw_data(time_data[:]), raw_data(freq_data[:]))
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("KissFFT real:   % 10.3f µs\n", µs / f64(ITERATIONS))
    }

    // Meow FFT
    {
        freq_data: [SIZE]meow_fft.Meow_FFT_Complex
        samples: [SIZE]f32
        for i in 0..<SIZE do samples[i] = f32(1)

        stopwatch := time.Stopwatch{}
        workset_bytes := meow_fft.meow_fft_generate_workset_real(int(SIZE), nil)
        workset, err := mem.alloc(int(workset_bytes))
        defer free(workset)
        meow_fft.meow_fft_generate_workset_real(int(SIZE), workset)

        time.stopwatch_start(&stopwatch)
        for i in 0..<ITERATIONS do meow_fft.meow_fft_real(workset, raw_data(samples[:]), raw_data(freq_data[:]))
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("Meow FFT:       % 10.3f µs\n", µs / f64(ITERATIONS))
    }

    // Pffft
    {
        out : [SIZE*2]f32
        samples: [SIZE]f32
        for i in 0..<SIZE do samples[i] = f32(1)

        stopwatch := time.Stopwatch{}

        setup := pffft.pffft_new_setup(int(SIZE), pffft.pffft_transform_t.PFFFT_REAL)
        defer pffft.pffft_destroy_setup(setup)

        time.stopwatch_start(&stopwatch)
        for i in 0..<ITERATIONS {
            pffft.pffft_transform_ordered(setup, raw_data(samples[:]),raw_data(out[:]), nil, pffft.pffft_direction_t.PFFFT_FORWARD)
        }
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("Pffft:          % 10.3f µs\n", µs / f64(ITERATIONS))
    }

    // Pocket FFT
    {
        stopwatch := time.Stopwatch{}

        out : [SIZE*2]f64
        for i in 0..<SIZE do out[i] = f64(1)

        plan := pocketfft.make_rfft_plan(SIZE)
        defer pocketfft.destroy_rfft_plan(plan)

        time.stopwatch_start(&stopwatch)
        for i in 0..<ITERATIONS do pocketfft.rfft_forward(plan, raw_data(out[:]), 1.0)
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("Pocket FFT:     % 10.3f µs\n", µs / f64(ITERATIONS))
    }

    // FFTW3
    {
        input : [SIZE]f64
        out : [SIZE]fftw3.fftw_complex

        for i in 0..<SIZE do input[i] = f64(1)

        plan := fftw3.fftw_plan_dft_r2c_1d(int(SIZE), raw_data(input[:]), raw_data(out[:]), 0)
        defer fftw3.fftw_destroy_plan(plan)

        stopwatch := time.Stopwatch{}

        time.stopwatch_start(&stopwatch)
        for i in 0..<ITERATIONS do fftw3.fftw_execute(plan)
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("FFTW3:          % 10.3f µs\n", µs / f64(ITERATIONS))
    }

    // MUFFT
    {
        dft: [SIZE]complex64
        samples: [SIZE]f32
        for i in 0..<SIZE do samples[i] = f32(1)

        plan := mufft.mufft_create_plan_1d_r2c(SIZE, 0)
        defer mufft.mufft_free_plan_1d(plan)

        stopwatch := time.Stopwatch{}

        time.stopwatch_start(&stopwatch)
        for i in 0..<ITERATIONS do mufft.mufft_execute_plan_1d(plan, raw_data(dft[:]), raw_data(samples[:]))
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("MUFFT:          % 10.3f µs\n", µs / f64(ITERATIONS))
    }


    // My radix-2 FFT
    {
        dft: [SIZE]complex64
        samples: [SIZE]f32
        for i in 0..<SIZE { samples[i] = f32(1) }

        stopwatch := time.Stopwatch{}
        plan := fft.create_fft_plan(SIZE)
        defer fft.destroy_fft_plan(plan)

        time.stopwatch_start(&stopwatch)
        for i in 0..<ITERATIONS do fft.run_fft_plan(plan, samples[:])
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("My radix-2 FFT: % 10.3f µs\n", µs / f64(ITERATIONS))
    }
}
