### Having fun with FFTs & Odin

Having fun with [Odin](https://odin-lang.org/) and learning about FFT implementations based on the book “Understanding Digital Signal Processing” by R.G.Lyons.



#### C libraries

Imported some C libraries to benchmark and compare.


* fftw3 - http://fftw.org/
* kissfft - https://github.com/mborgerding/kissfft
* meow_fft - https://github.com/JodiTheTigger/meow_fft
* pffft - https://bitbucket.org/jpommier/pffft
* pocketfft - https://gitlab.mpcdf.mpg.de/mtr/pocketfft
* muFFT - https://github.com/Themaister/muFFT


#### Benchmark

Latest benchmark with Odin `dev-2025-08:accdd7c2a` (Aug 2025).

Real-to-complex 4096-point FFT, averaged over 1,000 iterations on Apple M1 8 Core.

```
➜ odin run benchmark -o:speed -microarch:native
Naive DFT real:  32623.043 µs
KissFFT:           115.396 µs
KissFFT real:       72.794 µs
Meow FFT:           10.621 µs
Pffft:               2.637 µs
Pocket FFT:          7.980 µs
FFTW3:               7.833 µs
MUFFT:              10.773 µs
My radix-2 FFT:     20.548 µs
```


#### AI FFT and extreme FFT

Added by artificial intelligence (Claude, by Anthropic) in October 2026, in new files only.

* [fft/fft_ai.odin](fft/fft_ai.odin), the AI FFT: the radix-2 FFT with a real input as a half size complex FFT, radix-4 Stockham stages, a twiddle table per stage, split real and imaginary arrays and `#simd[4]f32`.
* [fft/fft_extreme.odin](fft/fft_extreme.odin), the extreme FFT: the AI FFT with the last stage run inside a split that does bin k and N/2 − k together, the stride 4 stage's twiddles four at a time, and smaller changes to the instructions. A little faster than pffft's ordered transform. `extreme_run` lays the bins out like `pffft_transform_ordered`, `extreme_run_arrays` puts them in a real and an imaginary array like vDSP.
* [benchmark_ai/bench.odin](benchmark_ai/bench.odin): both against pffft, Apple's vDSP ([benchmark_ai/vdsp.odin](benchmark_ai/vdsp.odin), macOS only) and the radix-2 FFT.
* [benchmark_stages/bench.odin](benchmark_stages/bench.odin): every pass of both on its own. [benchmark_stages/LOG.md](benchmark_stages/LOG.md) has each step from the AI FFT to the extreme FFT and what it measured, the ones that didn't work too.

The comments in the files list each change. Both are forward only.

vDSP's FFT and DFT take the samples split into even and odd ones first, `vDSP_ctoz`, timed with them. Both return twice the DFT in two arrays.

Odin `dev-2026-10:84bc3fc21`, Apple M1 Pro, the fastest of 5 runs of 20,000 iterations. Errors are relative to the largest bin of pffft's result. Runs differ by a few percent.

```
➜ odin run benchmark_ai -o:speed -microarch:native
N = 1024
  error vs pffft:  AI 7.39e-08, extreme 7.37e-08, radix-2 1.02e-07
                   extreme arrays 7.37e-08
                   vDSP FFT 1.02e-07, vDSP DFT 1.02e-07
  radix-2:            5.376 µs
  AI FFT:             0.711 µs
  extreme FFT:        0.549 µs
  extreme, arrays:    0.533 µs
  pffft ordered:      0.557 µs
  pffft unordered:    0.491 µs
  vDSP FFT:           0.506 µs
  vDSP DFT:           0.713 µs

N = 4096
  error vs pffft:  AI 6.78e-08, extreme 6.78e-08, radix-2 8.47e-08
                   extreme arrays 6.78e-08
                   vDSP FFT 6.78e-08, vDSP DFT 6.78e-08
  radix-2:           24.239 µs
  AI FFT:             3.117 µs
  extreme FFT:        2.472 µs
  extreme, arrays:    2.365 µs
  pffft ordered:      2.493 µs
  pffft unordered:    2.230 µs
  vDSP FFT:           2.335 µs
  vDSP DFT:           2.277 µs

N = 8192
  error vs pffft:  AI 6.94e-08, extreme 6.94e-08, radix-2 1.39e-07
                   extreme arrays 6.94e-08
                   vDSP FFT 6.94e-08, vDSP DFT 6.94e-08
  radix-2:           51.770 µs
  AI FFT:             6.627 µs
  extreme FFT:        5.444 µs
  extreme, arrays:    5.406 µs
  pffft ordered:      5.796 µs
  pffft unordered:    5.279 µs
  vDSP FFT:           4.934 µs
  vDSP DFT:           5.065 µs
```
