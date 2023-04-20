package bench

import "core:time"
import "core:fmt"
import "core:mem"
import "core:math/bits"
import "../kissfft"
import "../fft"


main :: proc() {
    DFT_SIZE :: uint(4096)
    dft: [DFT_SIZE]complex64
    samples: [DFT_SIZE]f32

    for i in 0..<DFT_SIZE {
        samples[i] = f32(0)
    }

    // Naive DFT
    {
        using fft
        stopwatch := time.Stopwatch{}

        time.stopwatch_start(&stopwatch)
        naive_dft_real(dft[:], samples[:], DFT_SIZE)
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("Naive DFT real: {} µs\n", µs)
    }

    // KISS FFT
    {
        cx_in: [DFT_SIZE]kissfft.kiss_fft_cpx
        cx_out: [DFT_SIZE]kissfft.kiss_fft_cpx

        // fill with zeroes
        for i in 0..<DFT_SIZE {
            cx_in[i] = {0, 0}
        }

        cfg := kissfft.kiss_fft_alloc(int(DFT_SIZE), false, nil, 0)
        // defer kissfft.kiss_fft_free(cfg)

        stopwatch := time.Stopwatch{}

        time.stopwatch_start(&stopwatch)
        kissfft.kiss_fft(cfg, &cx_in, &cx_out)

        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("KissFFT: {} µs\n", µs)
    }

    // Radix-2 FFT
    {
        using fft

        stopwatch := time.Stopwatch{}
        plan := create_fft_plan(DFT_SIZE)
        defer destroy_fft_plan(plan)

        time.stopwatch_start(&stopwatch)
        run_fft_plan(plan, samples[:])
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("Radix-2 FFT: {} µs\n", µs)
    }
}
