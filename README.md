# nam-cpu-trainer

nam-cpu-trainer is a single-executable [neural-amp-modeler](https://github.com/sdatkinson/neural-amp-modeler) trainer with no external dependencies.

It is purely CPU-based - no graphics card is needed. It uses a high-performance, highly portable training implementation based on [cpugrad](https://github.com/mikeoliphant/cpugrad).

## Getting it

Pre-built binaries for Windows x64, Linux x64 and Mac Arm64 can be downloaded from the [Releases](https://github.com/mikeoliphant/nam-cpu-trainer/releases) section. Note that the x64 binaries require a *reasonably* modern
CPU that supports the AVX2 instruction set. The Mac binary is only for Arm64 (not intel) Macs and will likely require some fiddling to let Apple's security let you run it.

## Usage

```
Usage: nam-cpu-trainer [--help] [--version] --input <input.wav> --output <output.wav>
                       [--channels <channels>] [--threads <numThreads>] [--epochs <maxEpochs>]
                       [--rand <randomSeed>] [--input-noise <dBFS>] [--loss <lossSpec>]
                       [--metadata <key=value>]...

Optional arguments:
  -h, --help                  shows help message and exits
  -v, --version               prints version information and exits
  -i, --input <input.wav>     Input .wav file used for the capture [required]
  -o, --output <output.wav>   Output (captured) .wav file [required]
  -c, --channels <channels>   Number of channels. A comma-separated list (like the default)
                              trains a model of each size and packs them into one .nam file,
                              like NAM's standard A2 (3 and 8 channels). A single value writes a
                              plain model. Supported sizes: 1, 2, 3, 4, 8 and 16
                                [default: "3,8"]
  -t, --threads <numThreads>  Number of threads (defaults to the number of physical cores)
  -e, --epochs <maxEpochs>    Maximum number of epochs to train for [default: 1000]
  -r, --rand <randomSeed>     Random seed, for repeatability (by default a random seed is used,
                              and recorded in the model)
  --input-noise <dBFS>        Add Gaussian noise at this RMS level (e.g. -75) to the training
                              input (off by default)
  -l, --loss <lossSpec>       Training loss, as comma-separated name[:weight][:key=value...]
                              terms. Losses:
                                  mse - mean squared error
                                  esr - error-to-signal ratio
                                  mrstft - multi-resolution STFT (sc = spectral convergence
                                    weight, logmag = log-magnitude weight)
                                    defaults: sc=0 logmag=1 [default: "mse:1,mrstft:0.0005"]
  -m, --metadata <key=value>  Add a key=value string to the model's metadata. Can be given more
                              than once, e.g.:
                                  -m name="Boss SD-1" -m modeled_by=Mike
                              NAM's standard keys include name, modeled_by, gear_make,
                              gear_model, gear_type and tone_type [may be repeated]
```

For example, to train for 100 epochs:

```
nam-cpu-trainer -i input.wav -o output.wav -e 100
```

By default, this trains a packed A2 model: a 3 channel ("lite") and an 8 channel ("full") model in one .nam file, the same format the standard NAM trainer produces (see [Packed models](#packed-models)). To train a single model instead, give a single channel count, for example just the full model:

```
nam-cpu-trainer -i input.wav -o output.wav -c 8 -e 100
```

When training concludes, the resulting .nam file will be the same path/name as "output.wav", but with a .nam extension.

If an epoch has a new low ESR, you will see a "*" at the end of the line. This means that weights have been stored for this epoch, and it will be used to produce the model unless a better epoch comes later.

To end training early while still producing a model, hit "Ctrl-C". This will stop training after the current epoch ends, keeping the best epoch so far. When training a packed model, any remaining model sizes are skipped, and the models trained so far are written: a packed model if there are two or more, otherwise a single model. Hitting "Ctrl-C" again before training has stopped will immediately abort training - but will **not** create a model.

### Packed models

When `-c` is given a comma-separated list of channel counts (the default is `3,8`), a model of each size is trained, smallest first, and they are written as one "SlimmableContainer" .nam file. Players that support packed models pick one of the models with a size or quality setting: with two models, a setting below 0.5 uses the smaller model and anything above that uses the larger one. Older players that don't support packed models can't load these files, so use a single channel count (such as `-c 8`) for those.

The smallest model trains fastest, so training it first gives the quickest indication of whether a capture is training well. Each model size starts from the same random seed, so each model in a packed file is the same as training that size on its own with `-c` and the same seed.

Each model in the file has its own loudness and `nam_cpu_trainer` metadata. The file's top-level metadata has the largest model's loudness and any `-m` metadata.

### Training loss

The loss used for training is set with `--loss`, as a comma-separated list of weighted terms. Each term is a loss name, optionally followed by `:weight` (default 1) and any number of `:key=value` parameters:

| Spec | Meaning |
|---|---|
| `mse:1,mrstft:0.0005` | MSE plus 0.0005 x MR-STFT (the default, similar to the standard NAM trainer) |
| `mse` | MSE only |
| `mse:1,mrstft:0.0005:sc=1` | Also include the spectral convergence part of MR-STFT |

By default, the MR-STFT loss uses only its log-magnitude part (`sc=0:logmag=1`). Names and keys are case-insensitive. The full loss, with all defaults filled in, is printed at the start of training.

Whatever the training loss, the best epoch is always chosen by validation ESR.

### Model metadata

Extra information can be added to the output .nam file's metadata with `-m key=value`, which can be repeated:

```
nam-cpu-trainer -i input.wav -o output.wav -m name="Boss SD-1" -m modeled_by=Mike -m gear_type=pedal
```

Values are stored as strings. NAM's standard metadata keys are `name`, `modeled_by`, `gear_make`, `gear_model`, `gear_type` and `tone_type`, which NAM players and model sites may show. Any other keys are stored as well. `date`, `loudness` and `nam_cpu_trainer` are written by the trainer and can't be set this way.

### Reproducibility

The random seed is printed at the start of training. Re-running with `-r <seed>` and the same other settings (including the number of threads) reproduces the same model.

The output .nam file records the training settings and results in a `nam_cpu_trainer` section of its metadata: trainer version, loss, seed, thread count, epochs (requested, completed and best), validation ESR, input noise (if used), target normalization, optimizer settings and the training/validation regions.

## Functionality

The trainer is currently capable of training NAM A2 models of 1, 2, 3 ("lite"), 4, 8 ("full"), or 16 channels, either as a single model or packed together into one model file. By default it trains a packed 3 and 8 channel model, like the standard NAM trainer.

It currently is designed for "input.wav" to be the default v3 NAM/Tone3000 sweep wav file. Other files will work, but training/validation sections will not get configured correctly.

Model "loudness" values are calculated and included in the output model.

Current differences/limitations with respect to the standard python NAM trainer:

- no auto-alignment is currently done for the input/output files
- no "gain" value is added to the model

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

Note - building with GCC 15 on a CPU that supports AVX-512 FP16 (such as recent Intel server CPUs) currently fails with errors in Eigen's `PacketMathFP16.h`, because "-march=native" enables Eigen's AVX-512 FP16 code. To work around this, use ```-DTARGET_ARCH=x86-64-v3```.

