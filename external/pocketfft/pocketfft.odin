package pocketfft

// clang -fPIC pocketfft.c -shared -o pocketfft.dylib -O3
foreign import lib "pocketfft.dylib"

foreign lib {
    make_cfft_plan :: proc(length: uint) -> rawptr ---
    destroy_cfft_plan :: proc(plan: rawptr) ---
    cfft_backward :: proc(plan: rawptr, c: [^]f64, fct: f64) ---
    cfft_forward :: proc(plan: rawptr, c: [^]f64, fct: f64) ---
    cfft_length :: proc(plan: rawptr) -> uint ---
    make_rfft_plan :: proc(length: uint) -> rawptr ---
    destroy_rfft_plan :: proc(plan: rawptr) ---
    rfft_backward :: proc(plan: rawptr, c: [^]f64, fct: f64) ---
    rfft_forward :: proc(plan: rawptr, c: [^]f64, fct: f64) ---
    rfft_length :: proc(plan: rawptr) ---
}
