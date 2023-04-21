package meow_fft

// clang++ -fPIC -shared meow_fft.c -o meow_fft.dylib -O3
// ---------------------------------------------------
//  meow_fft.c
//    #define MEOW_FFT_IMPLEMENTATION
//    #include <meow_fft.h>

foreign import lib "meow_fft.dylib"

Meow_FFT_Complex :: struct {
    r: f32,
    j: f32,
}
Meow_FFT_Workset :: rawptr
Meow_FFT_Workset_Real :: rawptr


foreign lib {
    meow_fft_generate_workset :: proc (N: int, workset: Meow_FFT_Workset) -> uint ---
    meow_fft_generate_workset_real :: proc(N: int, workset: Meow_FFT_Workset_Real) -> uint ---
    meow_fft_real :: proc(workset: Meow_FFT_Workset_Real, _in: [^]f32, out: [^]Meow_FFT_Complex) ---
    meow_fft_real_i :: proc(workset: Meow_FFT_Workset_Real, _in: [^]f32, temp: [^]Meow_FFT_Complex, out: [^]f32) ---
    meow_fft :: proc(workset: Meow_FFT_Workset_Real, _in: [^]Meow_FFT_Complex, out: [^]Meow_FFT_Complex) ---
    meow_fft_i :: proc(workset: Meow_FFT_Workset_Real, _in: [^]Meow_FFT_Complex, out: [^]Meow_FFT_Complex) ---
}
