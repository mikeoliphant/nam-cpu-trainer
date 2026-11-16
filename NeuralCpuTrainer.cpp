#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <optional>

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

// Pink noise from a fixed seed, so it is the same signal every run and on every platform (std::mt19937's output is
// fully specified, unlike std::uniform_real_distribution's)
std::vector<float> GeneratePinkNoise(size_t numSamples, float targetDbRms = -15.0f, unsigned int seed = 12345)
{
	if (numSamples == 0) return std::vector<float>();

	std::vector<float> noise(numSamples);

	// Convert target dB RMS to a linear RMS amplitude value
	const float targetLinearRms = std::pow(10.0f, targetDbRms / 20.0f);

	// Voss-McCartney algorithm setup (12 octaves)
	const int numRows = 12;
	std::vector<float> rows(numRows, 0.0f);
	float runningSum = 0.0f;

	std::mt19937 generator(seed);

	// Uniform in [-1, 1)
	auto dist = [](std::mt19937& gen) { return (float)(((double)gen() / 4294967296.0) * 2.0 - 1.0); };

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

// A float as the double it reads as (so 0.004f is written as 0.004, not 0.004000000189989805)
double FloatForJson(float value)
{
	return std::stod(std::format("{}", value));
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
void TrainNAM(ModelTrainer& trainer, const std::filesystem::path inWavePath, const std::filesystem::path targetWavePath, size_t maxEpochs, size_t numChannels, std::filesystem::path outputNAMPath, std::optional<float> inputNoiseDb, std::mt19937 noiseRand, const LossSpec& lossSpec, unsigned int seed)
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

	std::cout << "Activation: " << std::remove_reference_t<decltype(trainer.GetModel())>::ActivationJson << std::endl;

	float* trainInput = inData + startOffset - frameDelay;
	size_t trainFrames = (size_t)numFrames - verifyFrames - startOffset - frameDelay;

	if (inputNoiseDb)
	{
		// Only the training region gets noise - validation input stays clean
		std::cout << "Adding " << *inputNoiseDb << " dBFS RMS noise to training input" << std::endl;

		DataGen dataGen(noiseRand);
		dataGen.AddNoise(trainInput, trainFrames, std::pow(10.0f, *inputNoiseDb / 20.0f));
	}

	// Normalize the target (training and validation) so the training region is at -18 dBFS RMS, like NAM's
	// normalize_joint_dataset_output. The input is left alone. The scale is undone at export via head_scale.
	double targetSumSquares = 0;

	for (size_t i = 0; i < trainFrames; i++)
	{
		double sample = targetData[startOffset + i];

		targetSumSquares += sample * sample;
	}

	double targetRMS = std::sqrt(targetSumSquares / (double)trainFrames);

	if (!(targetRMS > 0) || !std::isfinite(targetRMS))
	{
		std::cerr << "Training target is silent or contains invalid samples - cannot normalize" << std::endl;

		return;
	}

	constexpr double targetNormalizeDb = -18.0;

	double targetScale = std::pow(10.0, targetNormalizeDb / 20.0) / targetRMS;

	std::cout << "Target training RMS: " << std::format("{:.2f}", LinearToDbAmplitude(targetRMS)) << " dBFS, scaling target by " << std::format("{:.6f}", targetScale) << " to " << targetNormalizeDb << " dBFS" << std::endl;

	for (size_t i = 0; i < (size_t)numFrames; i++)
	{
		targetData[i] = (float)(targetData[i] * targetScale);
	}

	trainer.SetLoss(lossSpec);
	trainer.SetMaxEpochs(maxEpochs);
	trainer.SetEpochCallback(EpochCallback);
	trainer.TrainModel(trainInput, targetData + startOffset, trainFrames, inData + verifyOffset - frameDelay, targetData + verifyOffset, verifyFrames - frameDelay);

	auto weights = trainer.GetBestWeights();

	// The model still has the last epoch's weights - load the best ones, which are what gets exported, so the loudness
	// measurement below is of the exported model
	auto weightIt = weights.begin();
	trainer.GetModel().SetWeights(weightIt);

	// Undo the target normalization so the exported model plays at the capture's real level
	// (this also makes the loudness measurement below come out at the real level)
	trainer.GetModel().SetHeadScale((float)(trainer.GetModel().GetHeadScale() / targetScale));

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

	// One "activation" entry per layer, from the model's activation type
	using ModelType = std::remove_reference_t<decltype(trainer.GetModel())>;

	std::string activationStr;

	for (size_t i = 0; i < ModelType::NumLayers; i++)
	{
		if (i > 0)
			activationStr += ",\n          ";

		activationStr += ModelType::ActivationJson;
	}

	// Our own metadata section (not part of NAM's format) - enough to reproduce the model and judge its quality
	nlohmann::ordered_json trainerInfo;

	trainerInfo["version"] = NCT_VERSION_STRING;
	trainerInfo["loss"] = FormatLossSpec(trainer.GetLossSpec());
	trainerInfo["seed"] = seed;
	trainerInfo["threads"] = trainer.GetNumThreads();
	trainerInfo["epochs"] = { { "requested", trainer.GetMaxEpochs() }, { "completed", trainer.GetEpochsCompleted() }, { "best", trainer.GetBestEpoch() } };
	trainerInfo["validation_esr"] = bestLoss;

	if (inputNoiseDb)
		trainerInfo["input_noise_dbfs"] = FloatForJson(*inputNoiseDb);

	trainerInfo["target_normalization"] = { { "rms_dbfs", LinearToDbAmplitude(targetRMS) }, { "scale", targetScale } };
	trainerInfo["optimizer"] = { { "lr", FloatForJson(trainer.GetLearningRate()) }, { "lr_decay", FloatForJson(trainer.GetLearningRateDecay()) }, { "segment", trainer.GetTrainingSize() }, { "mini_batch", trainer.GetMiniBatchSize() } };
	trainerInfo["data"] = { { "train_start_seconds", (double)startOffset / sampleRate }, { "validation_seconds", (double)verifyFrames / sampleRate } };

	// Indent to sit inside the template's "metadata" object
	std::string trainerInfoStr = trainerInfo.dump(2);

	ReplaceVariable(trainerInfoStr, "\n", "\n    ");

	std::map<std::string, std::string> variables =
	{
		{"{{CHANNELS}}", std::to_string(numChannels) },
		{"{{HEADSCALE}}", std::format("{:.9g}", weights[weights.size() - 1]) },	// full precision, so it matches the last weight
		{"{{DATE}}", dateStr },
		{"{{LOUDNESS}}", std::to_string(loudnessRMS) },
		{"{{WEIGHTS}}", weightStr },
		{"{{ACTIVATIONS}}", activationStr },
		{"{{TRAINER_INFO}}", trainerInfoStr }
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
std::vector<float> TrainNAM(std::nullptr_t& trainer, const std::filesystem::path inWavePath, const std::filesystem::path targetWavePath, size_t maxEpochs, size_t numChannels, std::filesystem::path outputNAMPath, std::optional<float> inputNoiseDb, std::mt19937 noiseRand, const LossSpec& lossSpec, unsigned int seed)
{
	(void)trainer;
	(void)inWavePath;
	(void)targetWavePath;
	(void)maxEpochs;
	(void)numChannels;
	(void)outputNAMPath;
	(void)inputNoiseDb;
	(void)noiseRand;
	(void)lossSpec;
	(void)seed;

	return std::vector<float>();
}

// Activation used for training: LeakyReLUT (standard A2) or FastTanhT (smooth, avoids zippering; exported as "Fasttanh")
template <int Channels>
using A2Backprop = ModelTrainerT<float, A2BackpropT<float, 1, Channels, A2KernelSizes, A2Dilations, LeakyReLUT>>;

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

static std::string LossHelp()
{
	std::string help = "Training loss: comma-separated name[:weight][:key=value...] terms, e.g. mse:1,mrstft:0.0005:sc=0. Losses:";

	for (auto& type : GetLossTypes())
	{
		help += "\n    " + type.Name + " - " + type.Description;

		if (!type.Params.empty())
		{
			help += " [defaults:";

			for (auto& [name, value] : type.Params)
				help += std::format(" {}={}", name, value);

			help += "]";
		}
	}

	return help;
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

	program.add_argument("--input-noise")
		.nargs(1, 1)
		.metavar("<dBFS>")
		.help("Add Gaussian noise at this RMS level (e.g. -75) to the training input (off by default)")
		.scan<'g', float>();

	program.add_argument("-l", "--loss")
		.default_value(std::string("mse:1,mrstft:0.0005"))
		.nargs(1)
		.metavar("<lossSpec>")
		.help(LossHelp());

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

	// Always use an explicit seed, so it can be recorded in the model and the run reproduced
	unsigned int seed = program.is_used("--rand") ? (unsigned int)program.get<int>("--rand") : std::random_device{}();

	std::mt19937 rand(seed);

	LossSpec lossSpec;

	try
	{
		lossSpec = ParseLossSpec(program.get("--loss"));
	}
	catch (const std::invalid_argument& e)
	{
		std::cerr << std::endl << "Invalid --loss: " << e.what() << std::endl;

		return 1;
	}

	std::cout << "Training NAM A2 with " << numChannels << " channels" << std::endl;
	std::cout << "Loss: " << FormatLossSpec(lossSpec) << std::endl;
	std::cout << "Random seed: " << seed << std::endl;

	A2Types modelTrainerObj = GetTrainer(numChannels, numThreads, rand);

	if (std::holds_alternative<std::nullptr_t>(modelTrainerObj))
	{
		std::cerr << std::endl << "Supported channel sizes are 1, 2, 4, 8 and 16" << std::endl;

		return 1;
	}

	std::optional<float> inputNoiseDb;

	if (program.is_used("--input-noise"))
	{
		inputNoiseDb = program.get<float>("--input-noise");
	}

	// Seeded from the (already copied) trainer seed, so --rand makes the noise repeatable too
	std::mt19937 noiseRand(rand());

	std::visit([&](auto& modelTrainer)
	{
		//modelTrainer->TestBackprop(0, randData.data(), randData.data(), MAX_BATCH_SIZE);

		//modelTrainer->TrainIdentity(sinData);

		//TrainNAM(modelTrainer, R"(C:\Share\Recordings\NAM\NAMv3Input.wav)", R"(C:\Share\Recordings\NAM\BossSD1Capture.wav)");

		TrainNAM(modelTrainer, inputPath, capturePath, maxEpochs, numChannels, outputNAMPath, inputNoiseDb, noiseRand, lossSpec, seed);

	}, modelTrainerObj);

	return 0;
}

