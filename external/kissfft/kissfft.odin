package kissfft

import "core:c"
foreign import lib "libkissfft-float.dylib"

kiss_fft_cfg :: rawptr
kiss_fftr_cfg :: rawptr
kiss_fft_cpx :: struct {
    r: f32,
    i: f32
}

kiss_fft_free :: free

foreign lib {
    kiss_fft_alloc :: proc(nfft: int, inverse_fft: c.bool, mem: rawptr, lenmem: ^uint) -> kiss_fft_cfg ---
    kiss_fft :: proc(cfg: kiss_fft_cfg, timedata: [^]kiss_fft_cpx, freqdata: [^]kiss_fft_cpx) ---
    kiss_fftr_alloc :: proc(nfft: int, inverse_fft: c.bool, mem: rawptr, lenmem: ^uint) -> kiss_fft_cfg ---
    kiss_fftr :: proc(cfg: kiss_fft_cfg, timedata: [^]f32, freqdata: [^]kiss_fft_cpx) ---
}
