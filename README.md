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
