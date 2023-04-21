package fftw3

import "core:c"

// ./configure --enable-shared
// make
foreign import lib "libfftw3.3.dylib"

fftw_complex :: c.complex_double

foreign lib {
    fftw_malloc :: proc(size: uint) -> rawptr ---
    fftw_free :: proc(obj: rawptr) ---
    fftw_plan_dft_1d :: proc(n0: int, _in: [^]fftw_complex, out: [^]fftw_complex, sign: int, flags: uint) -> rawptr ---
    fftw_plan_dft_r2c_1d :: proc(n0: int, _in: [^]f64, out: [^]fftw_complex, flags: uint) -> rawptr ---
    fftw_destroy_plan :: proc(plan: rawptr) ---
    fftw_execute :: proc(plan: rawptr) ---
}
