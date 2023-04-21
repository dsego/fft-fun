package mufft

// clang -fPIC cpu.c kernel.c fft.c -shared -o mufft.dylib -O3
// ---------------------------------------------------
foreign import lib "mufft.dylib"

foreign lib {
    mufft_create_plan_1d_c2c :: proc(N: uint, direction: int, flags: uint) -> rawptr ---
    mufft_create_plan_1d_r2c :: proc(N: uint, flags: uint) -> rawptr ---
    mufft_execute_plan_1d :: proc(plan: rawptr, output: rawptr, input: rawptr) ---
    mufft_free_plan_1d :: proc(plan: rawptr) ---
}
