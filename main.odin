package main

import "core:math"
import "core:fmt"
import rl "vendor:raylib"

import "./kissfft"
import "./fft"


WIDTH :: 1024
HEIGHT :: 768

SIZE :: 4096
samples : [SIZE]f32
spectrum : [SIZE]f32
dft: [SIZE]complex64
points: [WIDTH]rl.Vector2


magnitude :: proc(c: complex64) -> f32 {
    return math.sqrt(real(c) * real(c) + imag(c) * imag(c))
}

main :: proc() {
    rl.InitWindow(WIDTH, HEIGHT, "FFT")
    defer rl.CloseWindow()

    generate_samples(samples[:])
    // fft.naive_dft_real(dft[:], samples[:], SIZE)


    // cx_in: [SIZE]kissfft.kiss_fft_cpx
    // cx_out: [SIZE]kissfft.kiss_fft_cpx

    // for i in 0..<SIZE {
    //     cx_in[i] = {samples[i], 0}
    // }
    // cfg := kissfft.kiss_fft_alloc(int(SIZE), false, nil, 0)
    // kissfft.kiss_fft(cfg, &cx_in, &cx_out)

    plan := fft.create_fft_plan(SIZE)
    defer fft.destroy_fft_plan(plan)
    fft.run_fft_plan(plan, samples[:])

    for i in 0..<SIZE {
        // dft[i] = complex(cx_out[i].r, cx_out[i].i)
        // spectrum[i] = magnitude(dft[i])
        spectrum[i] = magnitude(plan.buffer[i])
    }

    j := 0
    for i in 0..<WIDTH {
        avg := (spectrum[j]+spectrum[j+1]+spectrum[j+2]+spectrum[j+3]) / 4.0
        points[i] = {f32(i), f32(-avg/5 + HEIGHT/2.0)}
        j += 4
    }


    for !rl.WindowShouldClose() {
        draw_screen()
    }
}

// 440Hz + 550Hz
generate_samples :: proc(buffer: []f32) {
    SAMPLERATE :: 8000.00
    i := 0
    k := f32(math.TAU / SAMPLERATE)

    for i < len(buffer)/2 {
        buffer[i] = math.sin(k * f32(i) * 400)
        i += 1
    }
    for i < len(buffer) {
        buffer[i] += math.sin(k * f32(i) * 1000)
        i += 1
    }
}

draw_screen :: proc() {
    rl.BeginDrawing()
    defer rl.EndDrawing()

    rl.ClearBackground(rl.BLACK)
    rl.DrawLineStrip(raw_data(points[:]), WIDTH, rl.PINK)
}
