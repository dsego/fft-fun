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

Real-to-complex 4096-point FFT, averaged over 1,000 iterations on Apple M1 8 Core (built with `odin build benchmark -o:speed`).

| Library        | µs        |
| ---------------|---------- |
| Brute force    | 49289.279 |
| KissFFT        | 69.463    |
| Meow FFT       | 10.521    |
| Pffft          | 2.529     |
| Pocket FFT     | 8.235     |
| FFTW3          | 7.957     |
| MUFFT          | 10.891    |
| My radix-2 FFT | 70.422    |
