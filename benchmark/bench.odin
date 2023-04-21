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

main :: proc() {
    // Naive DFT
    {
        dft: [SIZE]complex64
        samples: [SIZE]f32
        for i in 0..<SIZE { samples[i] = f32(0) }

        using fft
        stopwatch := time.Stopwatch{}

        time.stopwatch_start(&stopwatch)
        naive_dft_real(dft[:], samples[:], SIZE)
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("Naive DFT real: {} µs\n", µs)
    }

    // Kiss FFT
    {
        cx_in: [SIZE]kissfft.kiss_fft_cpx
        cx_out: [SIZE]kissfft.kiss_fft_cpx

        // fill with zeroes
        for i in 0..<SIZE {
            cx_in[i] = {0, 0}
        }

        cfg := kissfft.kiss_fft_alloc(int(SIZE), false, nil, nil)
        // defer kissfft.kiss_fft_free(cfg)

        stopwatch := time.Stopwatch{}

        time.stopwatch_start(&stopwatch)
        kissfft.kiss_fft(cfg, raw_data(cx_in[:]), raw_data(cx_out[:]))

        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("KissFFT: {} µs\n", µs)
    }

    // Kiss FFT real
    {
        time_data: [SIZE]f32
        freq_data: [SIZE]kissfft.kiss_fft_cpx

        // fill with zeroes
        for i in 0..<SIZE {
            time_data[i] = f32(0)
        }

        cfg := kissfft.kiss_fftr_alloc(int(SIZE), false, nil, nil)
        // defer kissfft.kiss_fft_free(cfg)

        stopwatch := time.Stopwatch{}

        time.stopwatch_start(&stopwatch)
        kissfft.kiss_fftr(cfg, raw_data(time_data[:]), raw_data(freq_data[:]))

        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("KissFFT real: {} µs\n", µs)
    }

    // Meow FFT
    {
        using meow_fft

        freq_data: [SIZE]Meow_FFT_Complex
        samples: [SIZE]f32
        for i in 0..<SIZE { samples[i] = f32(0) }

        stopwatch := time.Stopwatch{}
        workset_bytes := meow_fft_generate_workset_real(int(SIZE), nil)
        workset := mem.alloc(int(workset_bytes))
        defer free(workset)
        meow_fft_generate_workset_real(int(SIZE), workset)

        time.stopwatch_start(&stopwatch)
        meow_fft_real(workset, raw_data(samples[:]), raw_data(freq_data[:]))
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("Meow FFT: {} µs\n", µs)
    }

    // Pffft
    {
        using pffft

        out : [SIZE*2]f32
        samples: [SIZE]f32
        for i in 0..<SIZE { samples[i] = f32(0) }

        stopwatch := time.Stopwatch{}

        setup := pffft_new_setup(int(SIZE), pffft_transform_t.PFFFT_REAL)
        defer pffft_destroy_setup(setup)

        time.stopwatch_start(&stopwatch)
        pffft_transform_ordered(setup, raw_data(samples[:]), raw_data(out[:]), nil, pffft_direction_t.PFFFT_FORWARD)
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("Pffft: {} µs\n", µs)
    }

    // Pocket FFT
    {
        using pocketfft

        stopwatch := time.Stopwatch{}

        out : [SIZE*2]f64
        for i in 0..<SIZE { out[i] = f64(0) }

        plan := make_rfft_plan(SIZE)
        defer destroy_rfft_plan(plan)

        time.stopwatch_start(&stopwatch)
        rfft_forward(plan, raw_data(out[:]), 1.0)
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("Pocket FFT: {} µs\n", µs)
    }

    // FFTW3
    {
        using fftw3
        input : [SIZE]f64
        out : [SIZE]fftw_complex

        plan := fftw_plan_dft_r2c_1d(int(SIZE), raw_data(input[:]), raw_data(out[:]), 0)
        defer fftw_destroy_plan(plan)

        stopwatch := time.Stopwatch{}

        time.stopwatch_start(&stopwatch)
        fftw_execute(plan)
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("FFTW3: {} µs\n", µs)
    }

    // MUFFT
    {
        using mufft

        dft: [SIZE]complex64
        samples: [SIZE]f32
        for i in 0..<SIZE { samples[i] = f32(0) }

        plan := mufft_create_plan_1d_r2c(SIZE, 0)
        defer mufft_free_plan_1d(plan)

        stopwatch := time.Stopwatch{}

        time.stopwatch_start(&stopwatch)
        mufft_execute_plan_1d(plan, raw_data(dft[:]), raw_data(samples[:]))
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("MUFFT: {} µs\n", µs)
    }


    // My radix-2 FFT
    {
        using fft

        dft: [SIZE]complex64
        samples: [SIZE]f32
        for i in 0..<SIZE { samples[i] = f32(0) }

        stopwatch := time.Stopwatch{}
        plan := create_fft_plan(SIZE)
        defer destroy_fft_plan(plan)

        time.stopwatch_start(&stopwatch)
        run_fft_plan(plan, samples[:])
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("My radix-2 FFT: {} µs\n", µs)
    }
}
