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


#### AI FFT

Added by artificial intelligence (Claude, by Anthropic) in October 2026, in new files only: [fft/fft_ai.odin](fft/fft_ai.odin) and [benchmark_ai/bench.odin](benchmark_ai/bench.odin). The radix-2 FFT with a real input as a half size complex FFT, radix-4 Stockham stages, a twiddle table per stage, split real and imaginary arrays and `#simd[4]f32`. The comments in the file list each change.

Odin `dev-2026-10:84bc3fc21`, Apple M1 Pro, the fastest of 5 runs of 20,000 iterations. Errors are relative to the largest bin of pffft's result.

```
➜ odin run benchmark_ai -o:speed -microarch:native
N = 1024
  error vs pffft:  AI 7.39e-08, radix-2 1.02e-07
  radix-2:            5.377 µs
  AI FFT:             0.723 µs
  pffft ordered:      0.573 µs
  pffft unordered:    0.502 µs

N = 4096
  error vs pffft:  AI 6.78e-08, radix-2 8.47e-08
  radix-2:           25.081 µs
  AI FFT:             3.184 µs
  pffft ordered:      2.552 µs
  pffft unordered:    2.331 µs

N = 8192
  error vs pffft:  AI 6.94e-08, radix-2 1.39e-07
  radix-2:           54.173 µs
  AI FFT:             6.956 µs
  pffft ordered:      5.937 µs
  pffft unordered:    5.418 µs
```
