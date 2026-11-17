#include <filesystem>
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "NeuralModel.h"
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "argparse.hpp"
#include "Loss.h"
#include "MrStft.h"

// Prints the delay, MSE, ESR and MR-STFT of a model's output against the capture over the given stretch
static void EvaluateModel(NeuralAudio::NeuralModel& model, const float* input, const float* target, size_t numFrames)
{
	std::vector<float> namOutput(numFrames);

	model.Process(input, namOutput.data(), numFrames);

	MSELossT<float> mseLoss;

	int minDelay = 0;
	double minMSEErr = std::numeric_limits<double>::max();

	for (int delay = -128; delay < 128; delay++)
	{
		double delayErr = mseLoss.GetMeanLoss(namOutput.data(), target - delay, numFrames);

		if (delayErr < minMSEErr)
		{
			minMSEErr = delayErr;
			minDelay = delay;
		}
	}

	std::cout << "Delay samples: " << minDelay << std::endl;

	std::cout << "MSE: " << std::format("{:.8f}", minMSEErr) << std::endl;

	ESRLossT<float> esrLoss;

	double esrErr = esrLoss.GetMeanLoss(namOutput.data(), target - minDelay, numFrames);

	std::cout << "ESR: " << std::format("{:.8f}", esrErr) << std::endl;

	MultiResolutionStftLossT<float> mrstftLoss;
	mrstftLoss.SetWeights(1.0, 1.0);	// full auraloss-style MRSTFT (SC + log-mag) for comparability

	double mrstftErr = mrstftLoss.GetMeanLoss(namOutput.data(), target - minDelay, numFrames);

	std::cout << "MR-STFT: " << std::format("{:.8f}", mrstftErr) << std::endl;
}

int main(int argc, char* argv[])
{
	argparse::ArgumentParser program("nam-eval", "0.0.1");

	program.add_argument("-n", "--nam")
		.nargs(1)
		.required()
		.metavar("<model.nam>")
		.help("NAM model to evaluate");

	program.add_argument("-i", "--input")
		.nargs(1)
		.required()
		.metavar("<input.wav>")
		.help("Input .wav file used for the capture");

	program.add_argument("-o", "--output")
		.nargs(1)
		.required()
		.metavar("<output.wav>")
		.help("Output (captured) .wav file");

	program.add_argument("-q", "--quality")
		.default_value(1.0f)
		.nargs(1)
		.metavar("<0-1>")
		.help("Quality setting, which picks the model used from a packed model (as a player's setting would)")
		.scan<'g', float>();

	try
	{
		program.parse_args(argc, argv);
	}
	catch (const std::exception& e)
	{
		std::cerr << std::endl << "Commandline parse error: " << e.what() << std::endl << std::endl;
		std::cerr << program;

		return 1;
	}
	catch (...)
	{
		std::cerr << std::endl << "Commandline parse error" << std::endl << std::endl;

		std::cerr << program;

		return 1;
	}

	std::filesystem::path namPath = program.get("--nam");
	std::filesystem::path inputPath = program.get("--input");
	std::filesystem::path capturePath = program.get("--output");

	unsigned int channels;
	unsigned int sampleRate;
	drwav_uint64 numFrames;

	float* inData = drwav_open_file_and_read_pcm_frames_f32(inputPath.string().c_str(), &channels, &sampleRate, &numFrames, nullptr);
	float* targetData = drwav_open_file_and_read_pcm_frames_f32(capturePath.string().c_str(), &channels, &sampleRate, &numFrames, nullptr);

	if ((inData == nullptr) || (targetData == nullptr))
	{
		std::cerr << "Couldn't read " << ((inData == nullptr) ? inputPath : capturePath).string() << std::endl;

		return 1;
	}

	size_t verifyFrames = 48000 * 9;

	size_t verifyOffset = (size_t)numFrames - verifyFrames;

	NeuralAudio::NeuralModelLoader loader;

	// For a packed model, NeuralAudio uses this to pick which of its models is active
	loader.SetDefaultQualityScaleFactor(program.get<float>("--quality"));

	std::unique_ptr<NeuralAudio::NeuralModel> model(loader.CreateFromFile(namPath));

	if (!model)
	{
		std::cerr << "Couldn't load " << namPath.string() << std::endl;

		return 1;
	}

	if (model->HasQualityScaling())
		std::cout << "Packed model, quality setting: " << model->GetQualityScaleFactor() << std::endl;

	EvaluateModel(*model, inData + verifyOffset, targetData + verifyOffset, verifyFrames);

	drwav_free(inData, nullptr);
	drwav_free(targetData, nullptr);
}