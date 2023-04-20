package fft

import "core:math"

// A brute force implementation that executes the Fourier formula directly
naive_dft :: proc(
    dft: []complex64,
    samples: []f32,
    dft_size: uint
) #no_bounds_check {
    phase_angle:f32 = 2 * math.PI / f32(dft_size)

    for freq in 0..<dft_size {
        for sample, i in samples {
            time := phase_angle * f32(i)

            // Fourier formula: cos(2πft) - i×sin(2πft)
            ft := f32(freq) * time
            re := sample * math.cos(ft)
            im := -sample * math.sin(ft)
            dft[freq] = dft[freq] + complex(re, im)
        }
    }
}


// for real input the DFT output is symmetrical
naive_dft_real :: proc(
    dft: []complex64,
    samples: []f32,
    dft_size: uint
) #no_bounds_check {
    phase_angle:f32 = 2 * math.PI / f32(dft_size)
    real_size := dft_size % 2 == 0 ? dft_size / 2 : (dft_size+1) / 2

    for freq in 0..<real_size {
        for sample, i in samples {
            time := phase_angle * f32(i)

            // Fourier formula: cos(2πft) - i×sin(2πft)
            ft := f32(freq) * time
            re := sample * math.cos(ft)
            im := -sample * math.sin(ft)
            dft[freq] = dft[freq] + complex(re, im)
        }
        // exploit DFT symmetry
        if freq > 0 {
            dft[dft_size - freq] = conj(dft[freq])
        }
    }
}

