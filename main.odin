package main

import "core:c"
import "core:c/libc"
import "core:math"
import "core:fmt"
import "core:mem"
import rl "vendor:raylib"

import "external/kissfft"
import "external/meow_fft"
import "external/pffft"
import "external/pocketfft"
import "external/fftw3"
import "external/mufft"
import "fft"


WIDTH :: 1024
HEIGHT :: 768

SIZE :: 4096
samples : [SIZE]f32
spectrum : [SIZE]f32
dft: [SIZE]complex64
points: [WIDTH]rl.Vector2


magnitude :: proc(re: f32, im: f32) -> f32 {
    return math.sqrt(re * re + im * im)
}

run_fftw3 :: proc () {
    input: [SIZE]f64
    out : [SIZE]fftw3.fftw_complex

    for i in 0..<SIZE {
        input[i] = f64(samples[i])
    }

    plan := fftw3.fftw_plan_dft_r2c_1d(int(SIZE), raw_data(input[:]), raw_data(out[:]), 0)
    defer fftw3.fftw_destroy_plan(plan)

    fftw3.fftw_execute(plan)

    for i in 0..<SIZE {
        // fmt.println(out[i])
        spectrum[i] = magnitude(f32(real(out[i])), f32(imag(out[i])))
    }
}

run_pocketfft :: proc () {
    out : [SIZE*2]f64

    for i in 0..<SIZE {
        out[i] = f64(samples[i])
    }

    plan := pocketfft.make_rfft_plan(SIZE)
    defer pocketfft.destroy_rfft_plan(plan)

    pocketfft.rfft_forward(plan, raw_data(out[:]), 1.0)

    // TODO: not sure how to read the “halfcomplex”-format output
    spectrum[0] = magnitude(f32(out[0]), 0.0)
    for i in 1..<SIZE {
        spectrum[i] = magnitude(f32(out[i]), f32(out[2 * SIZE - i]))
    }
}

run_pffft :: proc() {
    out : [SIZE*2]f32

    setup := pffft.pffft_new_setup(SIZE, pffft.pffft_transform_t.PFFFT_REAL)
    defer pffft.pffft_destroy_setup(setup)
    pffft.pffft_transform_ordered(setup, raw_data(samples[:]), raw_data(out[:]), nil, pffft.pffft_direction_t.PFFFT_FORWARD)

    i := 0
    j := 0
    for i < 2*SIZE-1 {
        spectrum[j] = magnitude(out[i], out[i+1])
        i += 2
        j += 1
    }
}

run_kiss_fft :: proc() {
    freq_data: [SIZE]kissfft.kiss_fft_cpx
    cfg := kissfft.kiss_fftr_alloc(c.int(SIZE), false, nil, nil)
    kissfft.kiss_fftr(cfg, raw_data(samples[:]), raw_data(freq_data[:]))

    for i in 0..<SIZE {
        spectrum[i] = magnitude(freq_data[i].r, freq_data[i].i)
    }
}

run_meow_fft :: proc() {
    freq_data: [SIZE]meow_fft.Meow_FFT_Complex
    workset_bytes := meow_fft.meow_fft_generate_workset_real(SIZE, nil)
    workset, _ := mem.alloc(int(workset_bytes))
    defer free(workset)
    meow_fft.meow_fft_generate_workset_real(SIZE, workset)
    meow_fft.meow_fft_real(workset, raw_data(samples[:]), raw_data(freq_data[:]))

    for i in 0..<SIZE {
        spectrum[i] = magnitude(freq_data[i].r, freq_data[i].j)
    }
}

run_mufft :: proc() {
    plan := mufft.mufft_create_plan_1d_r2c(SIZE, 0)
    defer mufft.mufft_free_plan_1d(plan)
    mufft.mufft_execute_plan_1d(plan, raw_data(dft[:]), raw_data(samples[:]))
    for i in 0..<SIZE {
        spectrum[i] = magnitude(real(dft[i]), imag(dft[i]))
    }
}

run_fft :: proc() {
    plan := fft.create_fft_plan(SIZE)
    defer fft.destroy_fft_plan(plan)
    fft.run_fft_plan(plan, samples[:])

    for i in 0..<SIZE {
        spectrum[i] = magnitude(real(plan.buffer[i]), imag(plan.buffer[i]))
    }
}


main :: proc() {
    rl.InitWindow(WIDTH, HEIGHT, "FFT")
    defer rl.CloseWindow()

    generate_samples(samples[:])
    // run_mufft()
    run_pffft()

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
