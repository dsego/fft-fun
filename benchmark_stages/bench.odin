package bench_stages

/*
    ADDED BY ARTIFICIAL INTELLIGENCE: written by Claude, an AI model made by Anthropic, on
    2026-10-16. Times every pass of the AI FFT (fft/fft_ai.odin) and the extreme FFT
    (fft/fft_extreme.odin) on its own, to see where the time goes. LOG.md next to this file keeps
    what was tried and what these numbers were at each step.

        odin run benchmark_stages -o:speed -microarch:native

    Each pass runs on the plan's own work buffers over and over, its output isn't a real
    transform, only its timing counts. Microseconds per call, the fastest of 5 runs.
*/

import "core:fmt"
import "core:math"
import "core:time"
import "../fft"

ITERATIONS :: 50_000
RUNS :: 5

Args :: struct {
    ai:      ^fft.AI_Real_FFT_Plan,
    extreme: ^fft.Extreme_FFT_Plan,
    samples: []f32,
    out:     []complex64,
    stage:   int,
}

main :: proc() {
    for size in ([]int{1024, 4096, 8192}) {
        samples := make([]f32, size)
        defer delete(samples)
        for &sample, i in samples do sample = math.sin(f32(i) * 0.031) + 0.3 * math.sin(f32(i) * 0.47)
        out := make([]complex64, size / 2)
        defer delete(out)

        ai := fft.ai_create_real_fft_plan(size)
        defer fft.ai_destroy_real_fft_plan(&ai)
        extreme := fft.extreme_create_plan(size)
        defer fft.extreme_destroy_plan(&extreme)

        fmt.printf("AI FFT, N = %v\n", size)
        row("whole", best_of(proc(a: ^Args) { fft.ai_run_real_fft(a.ai, a.samples, a.out) }, &Args{ai = &ai, samples = samples, out = out}))
        row("first, stride 1", best_of(proc(a: ^Args) {
            fft.ai_first_stage(a.ai.stages[0], raw_data(a.samples), a.ai.buffers[0])
        }, &Args{ai = &ai, samples = samples}))
        for stage, index in ai.stages[1:] {
            name := fmt.tprintf("radix-%v, stride %v", 2 if stage.n == 2 else 4, stage.stride)
            row(name, best_of(proc(a: ^Args) {
                stage := a.ai.stages[a.stage]
                if stage.n == 2 {
                    fft.ai_radix2_stage(stage, a.ai.buffers[0], a.ai.buffers[1])
                } else {
                    fft.ai_radix4_stage(stage, a.ai.buffers[0], a.ai.buffers[1])
                }
            }, &Args{ai = &ai, stage = index + 1}))
        }
        row("split", best_of(proc(a: ^Args) {
            fft.ai_split(a.ai, a.ai.buffers[0], cast([^]f32)raw_data(a.out))
        }, &Args{ai = &ai, out = out}))

        fmt.printf("Extreme FFT, N = %v\n", size)
        row("whole", best_of(proc(a: ^Args) { fft.extreme_run(a.extreme, a.samples, a.out) }, &Args{extreme = &extreme, samples = samples, out = out}))
        row("first, stride 1", best_of(proc(a: ^Args) {
            fft.extreme_first_stage(a.extreme.stages[0], raw_data(a.samples), a.extreme.buffers[0])
        }, &Args{extreme = &extreme, samples = samples}))
        for stage, index in extreme.stages[1:] {
            row(fmt.tprintf("radix-4, stride %v", stage.stride), best_of(proc(a: ^Args) {
                fft.extreme_radix4_stage(a.extreme.stages[a.stage], a.extreme.buffers[0], a.extreme.buffers[1])
            }, &Args{extreme = &extreme, stage = index + 1}))
        }
        name := fmt.tprintf("radix-%v, stride %v + split", extreme.last_radix, extreme.last_stride)
        row(name, best_of(proc(a: ^Args) {
            out := cast([^]f32)raw_data(a.out)
            if a.extreme.last_radix == 2 {
                fft.extreme_radix2_split(a.extreme, a.extreme.buffers[0], out)
            } else {
                fft.extreme_radix4_split(a.extreme, a.extreme.buffers[0], out)
            }
        }, &Args{extreme = &extreme, out = out}))
        fmt.println()
    }
}

row :: proc(name: string, µs: f64) {
    fmt.printf("  %-30s % 7.3f µs\n", name, µs)
}

// Microseconds per call, the fastest of RUNS runs of ITERATIONS calls after a warm-up
best_of :: proc(run: proc(a: ^Args), args: ^Args) -> f64 {
    for _ in 0 ..< 2000 do run(args)
    best := max(f64)
    for _ in 0 ..< RUNS {
        stopwatch: time.Stopwatch
        time.stopwatch_start(&stopwatch)
        for _ in 0 ..< ITERATIONS do run(args)
        time.stopwatch_stop(&stopwatch)
        best = min(best, time.duration_microseconds(time.stopwatch_duration(stopwatch)) / ITERATIONS)
    }
    return best
}
