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

Latest benchmark with Odin `dev-2026-10:84bc3fc21` (Oct 2026).

Real-to-complex 4096-point FFT on Apple M1 Pro, the fastest of 5 runs of 1,000 iterations after a warm-up run. The naive DFT is a single run of 1,000.

```
➜ odin run benchmark -o:speed -microarch:native
Naive DFT real:  33662.202 µs
KissFFT real:       15.168 µs
Meow FFT:           10.310 µs
Pffft:               2.935 µs
Pocket FFT:          7.915 µs
FFTW3:               4.550 µs
MUFFT:              10.630 µs
My radix-2 FFT:     23.638 µs
```

The KISS FFT library was rebuilt with `-O3` in October 2026; the earlier one had been built without optimization and took 72.794 µs. FFTW was rebuilt with NEON at the same time, it took 7.822 µs without. FFTW and Pocket FFT run in double precision, the others in single.


#### AI FFT

Added by artificial intelligence (Claude, by Anthropic) in October 2026, in new files only.

* [fft/fft_ai.odin](fft/fft_ai.odin), the AI FFT: the radix-2 FFT with a real input as a half size complex FFT, radix-4 Stockham stages, a twiddle table per stage, split real and imaginary arrays and `#simd[4]f32`. The comments in the file list each change. Forward only.
* [benchmark_ai/bench.odin](benchmark_ai/bench.odin): against pffft, Apple's vDSP ([benchmark_ai/vdsp.odin](benchmark_ai/vdsp.odin), macOS only) and the radix-2 FFT.

The AI FFT went on to become the extreme FFT, a little faster than pffft and about level with vDSP, now a repository of its own: [github.com/dsego/extreme_fft](https://github.com/dsego/extreme_fft). Its `LOG.md` has each step and what it measured, the ones that didn't work too.

vDSP's FFT and DFT take the samples split into even and odd ones first, `vDSP_ctoz`, timed with them. Both return twice the DFT in two arrays.

Odin `dev-2026-10:84bc3fc21`, Apple M1 Pro, the fastest of 5 runs of 20,000 iterations. Errors are relative to the largest bin of pffft's result. Runs differ by a few percent.

```
➜ odin run benchmark_ai -o:speed -microarch:native
N = 1024
  error vs pffft:  AI 7.39e-08, radix-2 1.02e-07
                   vDSP FFT 1.02e-07, vDSP DFT 1.02e-07
  radix-2:            5.267 µs
  AI FFT:             0.716 µs
  pffft ordered:      0.576 µs
  pffft unordered:    0.507 µs
  vDSP FFT:           0.526 µs
  vDSP DFT:           0.544 µs

N = 4096
  error vs pffft:  AI 6.78e-08, radix-2 8.47e-08
                   vDSP FFT 6.78e-08, vDSP DFT 6.78e-08
  radix-2:           24.874 µs
  AI FFT:             3.229 µs
  pffft ordered:      2.617 µs
  pffft unordered:    2.232 µs
  vDSP FFT:           2.267 µs
  vDSP DFT:           2.298 µs

N = 8192
  error vs pffft:  AI 6.94e-08, radix-2 1.39e-07
                   vDSP FFT 6.94e-08, vDSP DFT 6.94e-08
  radix-2:           52.377 µs
  AI FFT:             6.761 µs
  pffft ordered:      5.837 µs
  pffft unordered:    5.915 µs
  vDSP FFT:           5.287 µs
  vDSP DFT:           5.272 µs
```
