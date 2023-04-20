package kissfft

import "core:c"
foreign import lib "../libkissfft-float.dylib"

MAXFACTORS :: 32

kiss_fft_state :: struct {
    nfft: int,
    inverse: int,
    factors: [MAXFACTORS]int,
    twiddles: [1]kiss_fft_cpx,
}

kiss_fft_cfg :: ^kiss_fft_state
kiss_fft_scalar :: f32

kiss_fft_cpx :: struct {
    r: kiss_fft_scalar,
    i: kiss_fft_scalar
}

kiss_fft_free :: free

foreign lib {
    kiss_fft_alloc :: proc(nfft: int, inverse_fft: c.bool, mem: rawptr, lenmem: uint) -> kiss_fft_cfg ---
    kiss_fft :: proc(cfg: kiss_fft_cfg, cx_in: rawptr, cx_out: rawptr) ---
}
