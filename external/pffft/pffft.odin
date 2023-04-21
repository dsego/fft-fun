package pffft

// clang -fPIC pffft.c -shared -o pffft.dylib -O3
foreign import lib "pffft.dylib"

pffft_direction_t :: enum {
    PFFFT_FORWARD,
    PFFFT_BACKWARD
}

pffft_transform_t :: enum {
    PFFFT_REAL,
    PFFFT_COMPLEX
}

foreign lib {
    pffft_new_setup :: proc(N: int, transform: pffft_transform_t) -> rawptr ---
    pffft_destroy_setup :: proc(setup: rawptr) ---
    pffft_transform :: proc(setup: rawptr, input: [^]f32, output: [^]f32, work: [^]f32, direction: pffft_direction_t) ---
    pffft_transform_ordered :: proc(setup: rawptr, input: [^]f32, output: [^]f32, work: [^]f32, direction: pffft_direction_t) ---
    pffft_zreorder :: proc(setup, input: [^]f32, output: [^]f32, direction: pffft_direction_t) ---
    pffft_zconvolve_accumulate :: proc(setup, dft_a: [^]f32, dft_b: [^]f32, dft_ab: [^]f32, scaling: f32) ---
    pffft_aligned_malloc :: proc(nb_bytes: uint) -> rawptr ---
    pffft_aligned_free :: proc(setup: rawptr) ---
    pffft_simd_size :: proc() -> int ---
}
