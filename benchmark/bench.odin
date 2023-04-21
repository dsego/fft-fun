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
import "../fft"


main :: proc() {
    FFT_SIZE :: uint(4096)
    dft: [FFT_SIZE]complex64
    samples: [FFT_SIZE]f32

    for i in 0..<FFT_SIZE {
        samples[i] = f32(0)
    }

    // Naive DFT
    {
        using fft
        stopwatch := time.Stopwatch{}

        time.stopwatch_start(&stopwatch)
        naive_dft_real(dft[:], samples[:], FFT_SIZE)
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("Naive DFT real: {} µs\n", µs)
    }

    // Kiss FFT
    {
        cx_in: [FFT_SIZE]kissfft.kiss_fft_cpx
        cx_out: [FFT_SIZE]kissfft.kiss_fft_cpx

        // fill with zeroes
        for i in 0..<FFT_SIZE {
            cx_in[i] = {0, 0}
        }

        cfg := kissfft.kiss_fft_alloc(int(FFT_SIZE), false, nil, nil)
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
        time_data: [FFT_SIZE]f32
        freq_data: [FFT_SIZE]kissfft.kiss_fft_cpx

        // fill with zeroes
        for i in 0..<FFT_SIZE {
            time_data[i] = f32(0)
        }

        cfg := kissfft.kiss_fftr_alloc(int(FFT_SIZE), false, nil, nil)
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

        stopwatch := time.Stopwatch{}
        freq_data: [FFT_SIZE]Meow_FFT_Complex
        workset_bytes := meow_fft_generate_workset_real(int(FFT_SIZE), nil)
        workset := mem.alloc(int(workset_bytes))
        defer free(workset)
        meow_fft_generate_workset_real(int(FFT_SIZE), workset)

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

        stopwatch := time.Stopwatch{}

        out : [FFT_SIZE*2]f32

        setup := pffft_new_setup(int(FFT_SIZE), pffft_transform_t.PFFFT_REAL)
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

        out : [FFT_SIZE*2]f64
        for i in 0..<FFT_SIZE {
            out[i] = f64(0)
        }

        plan := make_rfft_plan(FFT_SIZE)
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
        input : [FFT_SIZE]f64
        out : [FFT_SIZE]fftw_complex

        plan := fftw_plan_dft_r2c_1d(int(FFT_SIZE), raw_data(input[:]), raw_data(out[:]), 0)
        defer fftw_destroy_plan(plan)

        stopwatch := time.Stopwatch{}

        time.stopwatch_start(&stopwatch)
        fftw_execute(plan)
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("FFTW3: {} µs\n", µs)
    }


    // My radix-2 FFT
    {
        using fft

        stopwatch := time.Stopwatch{}
        plan := create_fft_plan(FFT_SIZE)
        defer destroy_fft_plan(plan)

        time.stopwatch_start(&stopwatch)
        run_fft_plan(plan, samples[:])
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("My radix-2 FFT: {} µs\n", µs)
    }
}
