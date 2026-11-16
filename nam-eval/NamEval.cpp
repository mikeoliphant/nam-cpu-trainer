#include <filesystem>
#include <format>
#include <iostream>
#include <limits>
#include <vector>

#include "NeuralModel.h"
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "argparse.hpp"
#include "Loss.h"
#include "MrStft.h"

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
		.help("Input .wav file use to capture");

	program.add_argument("-o", "--output")
		.nargs(1)
		.required()
		.metavar("<output.wav>")
		.help("Output (captured) .wav file");

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

	size_t verifyFrames = 48000 * 9;

	size_t verifyOffset = (size_t)numFrames - verifyFrames;

	NeuralAudio::NeuralModelLoader loader;

	auto namModel = loader.CreateFromFile(namPath);

	std::vector<float> namOutput(verifyFrames);

	namModel->Process(inData + verifyOffset, namOutput.data(), verifyFrames);

	float* targetPtr = targetData + verifyOffset;

	MSELossT<float> mseLoss;

	int minDelay = 0;
	double minMSEErr = std::numeric_limits<double>::max();

	for (int delay = -128; delay < 128; delay++)
	{
		double delayErr = mseLoss.GetMeanLoss(namOutput.data(), targetPtr - delay, verifyFrames);

		if (delayErr < minMSEErr)
		{
			minMSEErr = delayErr;
			minDelay = delay;
		}
	}

	std::cout << "Delay samples: " << minDelay << std::endl;

	std::cout << "MSE: " << std::format("{:.8f}", minMSEErr) << std::endl;

	ESRLossT<float> esrLoss;

	double esrErr = esrLoss.GetMeanLoss(namOutput.data(), targetPtr - minDelay, verifyFrames);

	std::cout << "ESR: " << std::format("{:.8f}", esrErr) << std::endl;

	MultiResolutionStftLossT<float> mrstftLoss;
	mrstftLoss.SetWeights(1.0, 1.0);	// full auraloss-style MRSTFT (SC + log-mag) for comparability

	double mrstftErr = mrstftLoss.GetMeanLoss(namOutput.data(), targetPtr - minDelay, verifyFrames);

	std::cout << "MR-STFT: " << std::format("{:.8f}", mrstftErr) << std::endl;
}