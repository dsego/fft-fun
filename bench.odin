package main

import "core:time"
import "core:fmt"
import "core:mem"
import "core:math/bits"

run_benchmark :: proc() {
    DFT_SIZE :: uint(4096)
    dft: [DFT_SIZE]complex64
    samples: [DFT_SIZE]f32

    for i in 0..<DFT_SIZE {
        samples[i] = f32(0)
    }


    // Naive DFT
    {
        stopwatch := time.Stopwatch{}

        time.stopwatch_start(&stopwatch)
        naive_dft_real(dft[:], samples[:], DFT_SIZE)
        time.stopwatch_stop(&stopwatch)

        duration := time.stopwatch_duration(stopwatch)
        µs := time.duration_microseconds(duration)
        fmt.printf("Naive DFT real: {} µs\n", µs)
    }

    // Radix-2 FFT
    {
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
