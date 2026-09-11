# nam-cpu-trainer

nam-cpu-trainer is a single-executable [neural-amp-modeler](https://github.com/sdatkinson/neural-amp-modeler) trainer with no external dependencies.

It is purely CPU-based - no graphics card is needed. It uses a high-performance, highly portable training based on [cpugrad](https://github.com/mikeoliphant/cpugrad).

# Usage

```
nam-cpu-trainer [--help] [--version] --input <input.wav> --output <output.wav> [--channels <numChannels>] [--threads <numThreads>] [--epochs <maxEpochs>]

Optional arguments:
  -h, --help                    shows help message and exits
  -v, --version                 prints version information and exits
  -i, --input <input.wav>       Input .wav file use to capture [required]
  -o, --output <output.wav>     Output (captured) .wav file [required]
  -c, --channels <numChannels>  Number of channels [default: 3]
  -t, --threads <numThreads>    Number of threads (defaults to detected # cores)
  -e, --epochs <maxEpochs>      Maximum number of epochs to train for [default: 1000]
```

When training concludes, the result .nam file will be the same path/name as "output.wav", but with a .nam extension.

To end training early, hit "Ctrl-C". This will stop training after the current epoch ends. Hitting "Ctrl-C" again will immediately abort training - but will not create a model.

# Functionality

The trainer is currently capable of training NAM A2 models of 1, 2, 3 ("lite"), 4, 8 ("full"), or 16 channels.

It currently is designed for "input.wav" to be the default v3 NAM/Tone3000 sweep wav file. Other files will work, but training/validation sections will not get configured correctly.

Model "loudness" values are calculated and included in the output model.

Current differences/limitations with respect to the standard python nam trainer:

- training loss is currently pre-emphasized MSE, rather than a combination of MSE and MR-STFT
- no "gain" value is added to the model
- "packed" A2 (lite/full) is not currently supported

# Performance

Performance optimization is still very much in progress, but it is already *much* better than trying to run the standard python trainer on CPU.
