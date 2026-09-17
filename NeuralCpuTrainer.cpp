#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>

#include "NeuralModel.h"
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "argparse.hpp"
#include "ModelTrainer.h"
#include "NAM.h"
#include "Dataset.h"
#include "a2json.h"

using namespace NeuralAudio;
using namespace cpugrad;

static void TestNAM(std::filesystem::path modelPath)
{
	NeuralAudio::NeuralModelLoader loader;

	auto namModel = loader.CreateFromFile(modelPath);

	size_t numSamples = 24000; //48000 * 3;

	auto modelTrainer = new ModelTrainerT<float, A2BackpropT<float, 1, 3, A2KernelSizes, A2Dilations>>();

	DataGen dataGen;

	auto input = dataGen.GenerateSin(numSamples, numSamples);

	std::vector<float> namOutput(numSamples);

	namModel->Process(input.data(), namOutput.data(), numSamples);

	std::ifstream jsonStream(modelPath, std::ifstream::binary);

	nlohmann::json modelJson;
	jsonStream >> modelJson;

	std::vector<float> weights = modelJson.at("weights");

	auto it = weights.begin();

	modelTrainer->GetModel().SetWeights(it);
	modelTrainer->GetModel().SetHeadScale(*it);

	std::vector<float> verifyOutput(numSamples);

	modelTrainer->VerifyModel(input.data(), verifyOutput.data(), numSamples);

	MSELossT<float> mseLoss;

	size_t receptiveField = modelTrainer->GetReceptiveField();

	double err = mseLoss.GetMeanLoss(verifyOutput.data() + receptiveField, namOutput.data() + receptiveField, numSamples - receptiveField) / (double)(numSamples - receptiveField);

	std::cout << "MSE: " << err << std::endl;
}

volatile sig_atomic_t keepRunning = 1;

void SignalHandler(int sigNum)
{
	if (sigNum == SIGINT)
	{
		std::signal(sigNum, SIG_DFL);
		
		std::cout << std::endl << "Stopping after next epoch. Press ctl-c again to force immediate exit (model will not be saved)." << std::endl;

		keepRunning = 0; // Set flag to break the loop
	}
}

bool EpochCallback(size_t epoch, double loss)
{
	(void)epoch;
	(void)loss;

	return (keepRunning == 1);
}

std::vector<float> GeneratePinkNoise(size_t numSamples, float targetDbRms = -15.0f)
{
	if (numSamples == 0) return std::vector<float>();

	std::vector<float> noise(numSamples);

	// Convert target dB RMS to a linear RMS amplitude value
	const float targetLinearRms = std::pow(10.0f, targetDbRms / 20.0f);

	// Voss-McCartney algorithm setup (12 octaves)
	const int numRows = 12;
	std::vector<float> rows(numRows, 0.0f);
	float runningSum = 0.0f;

	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	for (int i = 0; i < numRows; ++i) {
		rows[i] = dist(generator);
		runningSum += rows[i];
	}

	// Generate raw pink noise and compute the sum for DC offset
	double sampleSum = 0.0;

	for (size_t i = 0; i < numSamples; ++i)
	{
		int rowToUpdate = 0;
		size_t index = i + 1;

		while ((index & 1) == 0 && rowToUpdate < numRows - 1)
		{
			rowToUpdate++;
			index >>= 1;
		}

		runningSum -= rows[rowToUpdate];
		rows[rowToUpdate] = dist(generator);
		runningSum += rows[rowToUpdate];

		float sample = runningSum + dist(generator);
		noise[i] = sample;
		sampleSum += sample;
	}

	// Remove DC offset (ensures RMS calculation reflects true AC energy)
	float mean = static_cast<float>(sampleSum / numSamples);
	double sumSquares = 0.0;

	for (size_t i = 0; i < numSamples; ++i)
	{
		noise[i] -= mean;
		sumSquares += static_cast<double>(noise[i]) * noise[i];
	}

	// Calculate current RMS and apply the target scaling factor
	float currentRms = std::sqrt(static_cast<float>(sumSquares / numSamples));

	if (currentRms > 0.0f)
	{
		float scaleFactor = targetLinearRms / currentRms;

		for (float& sample : noise)
		{
			sample *= scaleFactor;
		}
	}

	return noise;
}

void ReplaceVariable(std::string& str, const std::string& from, const std::string& to)
{
	if (from.empty()) return;

	size_t start_pos = 0;

	while ((start_pos = str.find(from, start_pos)) != std::string::npos)
	{
		str.replace(start_pos, from.length(), to);
		start_pos += to.length();
	}
}

double LinearToDbAmplitude(double linear)
{
	if (linear <= 0.0)
	{
		return -INFINITY;
	}

	return 20.0 * std::log10(linear);
}

double GetRMSLevel(float* data, size_t numSamples)
{
	double sumOfSquares = 0.0;

	for (size_t i = 0; i < numSamples; ++i)
	{
		sumOfSquares += static_cast<double>(data[i] * data[i]);
	}

	double meanSquare = sumOfSquares / static_cast<double>(numSamples);

	return LinearToDbAmplitude(std::sqrt(meanSquare));
}

template <typename ModelTrainer>
void TrainNAM(ModelTrainer& trainer, const std::filesystem::path inWavePath, const std::filesystem::path targetWavePath, size_t maxEpochs, size_t numChannels, std::filesystem::path outputNAMPath)
{
	unsigned int channels;
	unsigned int sampleRate;
	drwav_uint64 numFrames;

	std::signal(SIGINT, SignalHandler);

	float* inData = drwav_open_file_and_read_pcm_frames_f32(inWavePath.string().c_str(), &channels, &sampleRate, &numFrames, nullptr);
	float* targetData = drwav_open_file_and_read_pcm_frames_f32(targetWavePath.string().c_str(), &channels, &sampleRate, &numFrames, nullptr);

	size_t startOffset = 48000 * 13;

	size_t verifyFrames = 48000 * 9;

	size_t frameDelay = 0;

	//startOffset = 0;
	//verifyFrames = (size_t)(numFrames * 0.1f);

	size_t verifyOffset = (size_t)numFrames - verifyFrames;

	trainer.SetMaxEpochs(maxEpochs);
	trainer.SetEpochCallback(EpochCallback);

	trainer.TrainModel(inData + startOffset - frameDelay, targetData + startOffset, (size_t)numFrames - verifyFrames - startOffset - frameDelay, inData + verifyOffset - frameDelay, targetData + verifyOffset, verifyFrames - frameDelay);

	auto weights = trainer.GetBestWeights();

	weights.push_back(trainer.GetModel().GetHeadScale());

	double bestLoss = trainer.GetBestLoss();

	std::cout << std::endl << "Best ESR: " << std::format("{:.8f}", bestLoss) << std::endl;

	size_t receptiveField = trainer.GetModel().GetReceptiveField();

	std::vector<float> noise = GeneratePinkNoise(48000);
	noise.insert(noise.begin(), receptiveField, 0.0f);

	std::vector<float> noiseOutput(noise.size());

	trainer.VerifyModel(noise.data(), noiseOutput.data(), noise.size());

	double loudnessRMS = GetRMSLevel(noiseOutput.data() + receptiveField, noiseOutput.size() - receptiveField);

	std::cout << "Output RMS on -15dB input (\"loudness\"): " << loudnessRMS << std::endl;

	std::string weightStr;

	for (size_t i = 0; i < weights.size(); ++i) {
		std::format_to(std::back_inserter(weightStr), "{:.9g}", weights[i]);

		if (i < weights.size() - 1) {
			weightStr += ", ";
		}
	}

	auto now = std::chrono::system_clock::now();
	auto system_days = std::chrono::floor<std::chrono::days>(now);
	std::chrono::year_month_day ymd{ system_days };
	std::chrono::hh_mm_ss hms{ std::chrono::floor<std::chrono::seconds>(now - system_days) };

	std::string dateStr = std::format(
		"{{ "
		" \"year\": {},"
		" \"month\": {},"
		" \"day\": {},"
		" \"hour\": {},"
		" \"minute\": {},"
		" \"second\": {}"
		" }}",
		int(ymd.year()),
		unsigned(ymd.month()),
		unsigned(ymd.day()),
		hms.hours().count(),
		hms.minutes().count(),
		hms.seconds().count()
	);

	std::map<std::string, std::string> variables =
	{
		{"{{CHANNELS}}", std::to_string(numChannels) },
		{"{{HEADSCALE}}", std::to_string(weights[weights.size() - 1]) },
		{"{{DATE}}", dateStr },
		{"{{LOUDNESS}}", std::to_string(loudnessRMS) },
		{"{{WEIGHTS}}", weightStr }
	};

	std::string jsonOutStr = std::string(A2JsonData);

	for (const auto& [variable, value] : variables)
	{
		ReplaceVariable(jsonOutStr, variable, value);
	}

	std::ofstream namStream(outputNAMPath);

	namStream << jsonOutStr;
}

// Keeps the compiler happy
std::vector<float> TrainNAM(std::nullptr_t& trainer, const std::filesystem::path inWavePath, const std::filesystem::path targetWavePath, size_t maxEpochs, size_t numChannels, std::filesystem::path outputNAMPath)
{
	(void)trainer;
	(void)inWavePath;
	(void)targetWavePath;
	(void)maxEpochs;
	(void)numChannels;
	(void)outputNAMPath;

	return std::vector<float>();
}

template <int Channels>
using A2Backprop = ModelTrainerT<float, A2BackpropT<float, 1, Channels, A2KernelSizes, A2Dilations>>;

using A2Types = std::variant<std::nullptr_t, A2Backprop<1>, A2Backprop<2>, A2Backprop<3>, A2Backprop<4>, A2Backprop<8>, A2Backprop<16>>;

A2Types GetTrainer(size_t numChannels, size_t numThreads, std::mt19937& rand)
{
	switch (numChannels)
	{
		case 1:
			return A2Backprop<1>(numThreads, rand);
		case 2:
			return A2Backprop<2>(numThreads, rand);
		case 3:
			return A2Backprop<3>(numThreads, rand);
		case 4:
			return A2Backprop<4>(numThreads, rand);
		case 8:
			return A2Backprop<8>(numThreads, rand);
		case 16:
			return A2Backprop<16>(numThreads, rand);
	}

	return nullptr;
}


int main(int argc, char* argv[])
{	
	argparse::ArgumentParser program("nam-cpu-trainer", NCT_VERSION_STRING);

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

	program.add_argument("-c", "--channels")
		.default_value(3)
		.nargs(1)
		.metavar("<numChannels>")
		.help("Number of channels")
		.scan<'i', int>();

	program.add_argument("-t", "--threads")
		.nargs(1)
		.metavar("<numThreads>")
		.help("Number of threads (defaults to detected # cores)")
		.scan<'i', int>();

	program.add_argument("-e", "--epochs")
		.default_value(1000)
		.nargs(1, 1)
		.metavar("<maxEpochs>")
		.help("Maximum number of epochs to train for")
		.scan<'i', int>();

	program.add_argument("-r", "--rand")
		.nargs(1, 1)
		.metavar("<randomSeed>")
		.help("Random seed for repeatability (by default a random value is used)")
		.scan<'i', int>();

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

	std::filesystem::path inputPath = program.get("--input");
	std::filesystem::path capturePath = program.get("--output");

	std::filesystem::path outputNAMPath = capturePath;
	outputNAMPath.replace_extension(".nam");
	
	std::cout << std::endl << "nam-cpu-trainer v" << NCT_VERSION_STRING << std::endl;
	std::cout << "Copyright 2026 Mike Oliphant (https://github.com/mikeoliphant/nam-cpu-trainer)" << std::endl << std::endl;

	//TestNAM(R"(C:\Code\NeuralCpuTrainer\BossWN-a2lite.nam)");


	//std::cout << sizeof(A2BackpropT<float, 1, 8, A2KernelSizes, A2Dilations>) << std::endl;

	size_t numChannels = (size_t)program.get<int>("--channels");

	size_t numThreads = program.is_used("--threads") ? (size_t)program.get<int>("--threads") : 0;

	size_t maxEpochs = ((size_t)program.get<int>("--epochs"));

	std::mt19937 rand(std::random_device{}());

	if (program.is_used("--rand"))
	{
		unsigned int seed = (unsigned int)program.get<int>("--rand");

		rand.seed(seed);
	}

	std::cout << "Training NAM A2 with " << numChannels << " channels" << std::endl;

	A2Types modelTrainerObj = GetTrainer(numChannels, numThreads, rand);

	if (std::holds_alternative<std::nullptr_t>(modelTrainerObj))
	{
		std::cerr << std::endl << "Supported channel sizes are 1, 2, 4, 8 and 16" << std::endl;

		return 1;
	}

	std::visit([&](auto& modelTrainer)
	{
		//modelTrainer->TestBackprop(0, randData.data(), randData.data(), MAX_BATCH_SIZE);

		//modelTrainer->TrainIdentity(sinData);

		//TrainNAM(modelTrainer, R"(C:\Share\Recordings\NAM\NAMv3Input.wav)", R"(C:\Share\Recordings\NAM\BossSD1Capture.wav)");

		TrainNAM(modelTrainer, inputPath, capturePath, maxEpochs, numChannels, outputNAMPath);

	}, modelTrainerObj);

	return 0;
}

