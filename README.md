# nam-cpu-trainer

nam-cpu-trainer is a single-executable [neural-amp-modeler](https://github.com/sdatkinson/neural-amp-modeler) trainer with no external dependencies.

It is purely CPU-based - no graphics card is needed. It uses a high-performance, highly portable training implementation based on [cpugrad](https://github.com/mikeoliphant/cpugrad).

## Usage

Pre-built binaries for Windows x64, Linux x64 and Mac Arm64 can be downloaded from the [Releases](https://github.com/mikeoliphant/nam-cpu-trainer/releases) section. Note that the x64 binaries require a *reasonably* modern
CPU that supports the AVX2 instruction set. The Mac binary is only for Arm64 (not intel) Macs and will likely require some fiddling to let Apple's security let you run it.

```
Usage: nam-cpu-trainer [--help] [--version] --input <input.wav> --output <output.wav> [--channels <numChannels>] [--threads <numThreads>] [--epochs <maxEpochs>] [--rand <randomSeed>]

Optional arguments:
  -h, --help                    shows help message and exits
  -v, --version                 prints version information and exits
  -i, --input <input.wav>       Input .wav file use to capture [required]
  -o, --output <output.wav>     Output (captured) .wav file [required]
  -c, --channels <numChannels>  Number of channels [default: 3]
  -t, --threads <numThreads>    Number of threads (defaults to detected # cores)
  -e, --epochs <maxEpochs>      Maximum number of epochs to train for [default: 1000]
  -r, --rand <randomSeed>       Random seed for repeatability (by default a random value is used)
```

When training concludes, the resulting .nam file will be the same path/name as "output.wav", but with a .nam extension.

If an epoch has a new low ESR, you will see a "*" at the end of the line. This means that weights have been stored for this epoch, and it will be used to produce the model unless a better epoch comes later.

To end training early while still producing a model, hit "Ctrl-C". This will stop training after the current epoch ends. Hitting "Ctrl-C" again will immediately abort training - but will **not** create a model.

## Functionality

The trainer is currently capable of training NAM A2 models of 1, 2, 3 ("lite"), 4, 8 ("full"), or 16 channels.

It currently is designed for "input.wav" to be the default v3 NAM/Tone3000 sweep wav file. Other files will work, but training/validation sections will not get configured correctly.

Model "loudness" values are calculated and included in the output model.

Current differences/limitations with respect to the standard python NAM trainer:

- training loss is currently pre-emphasized MSE, rather than a combination of MSE and MR-STFT
- no auto-alignment is currently done for the input/output files
- no "gain" value is added to the model
- "packed" A2 (lite/full) is not currently supported

Given these limitations (particularly the lack of MR-STFT training loss), I consider this project to be in an experimental stage at this point.

## Performance

Performance optimization is still very much in progress, but it is already *much* better than trying to run the standard python trainer on CPU. In fact, performance stacks up very well against GPU training performance.

For example, my Ryzen 7 5700X CPU with 3200MHz RAM trains **A2 "lite"** models at **~1.5 seconds** per epoch, and **A2 "full"** models at **~4.5 seconds** per epoch.

Performance depends highly on the following factors:

- Raw CPU speed
- Number of CPU cores
- Size of memory caches
- Compiler used

## Building from source

First clone the repository:
```bash
git clone --recurse-submodules -j4 https://github.com/mikeoliphant/nam-cpu-trainer
cd nam-cpu-trainer/build
```

Then compile the plugin using:

**Linux/MacOS**
```bash
cmake .. -DCMAKE_BUILD_TYPE="Release"
make -j4
```

**Windows**
```bash
cmake.exe -G "Visual Studio 18 2026" -A x64 ..
cmake --build . --config=release -j4
```

Note - you'll have to change the Visual Studio version if you are using a different one. Also note that instead of "cmake --build", you can just load the .slnx file into Visual Studio and build it there.

Performance can be highly dependent on compiler. In my experience, the **best performance can be had with recent versions of clang++**.

To use clang on Windows, make sure you have the relevant packages installed using the Visual Studio installer, then add " -T ClangCL" to your CMake commandline.

## CMake Options

```-DTARGET_ARCH=<arch>```: For clang/gcc, you can use this to change the default "-march=<arch>" for the compiler. It defaults to "native".

