package fftw3

import "core:c"

// FFTW 3.3.10, NEON in double precision needs the aarch64 build type, it guesses 'arm' on Apple Silicon
// ./configure --build=aarch64-apple-darwin --enable-shared --disable-static --disable-fortran --enable-neon
// make
// install_name_tool -id libfftw3.3.dylib .libs/libfftw3.3.dylib && codesign -f -s - .libs/libfftw3.3.dylib
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
