#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <optional>
#include <algorithm>
#include <cstdlib>

#if !defined(_WIN32)
#include <sys/ioctl.h>
#include <unistd.h>
#endif

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

[[maybe_unused]] static void TestNAM(std::filesystem::path modelPath)
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
		
		std::cout << std::endl << "Stopping after the current epoch (keeping the best epoch so far, and skipping any remaining model sizes)." << std::endl;
		std::cout << "Press ctrl-c again to abort immediately (no model will be saved)." << std::endl;

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

// User-supplied "key=value" strings for the model's metadata, in the order given
using MetadataPairs = std::vector<std::pair<std::string, std::string>>;

// Command-line text as UTF-8. On Windows, arguments arrive in the system code page rather than UTF-8.
std::string CommandLineToUtf8(const std::string& text)
{
#if defined(_WIN32)
	if (text.empty())
		return text;

	int wideLength = MultiByteToWideChar(CP_ACP, 0, text.data(), (int)text.size(), nullptr, 0);
	std::wstring wide(wideLength, L'\0');
	MultiByteToWideChar(CP_ACP, 0, text.data(), (int)text.size(), wide.data(), wideLength);

	int utf8Length = WideCharToMultiByte(CP_UTF8, 0, wide.data(), wideLength, nullptr, 0, nullptr, nullptr);
	std::string utf8(utf8Length, '\0');
	WideCharToMultiByte(CP_UTF8, 0, wide.data(), wideLength, utf8.data(), utf8Length, nullptr, nullptr);

	return utf8;
#else
	return text;
#endif
}

// Parses --metadata "key=value" arguments. Throws std::invalid_argument with a readable message.
MetadataPairs ParseMetadata(const std::vector<std::string>& args)
{
	// Keys the trainer writes itself
	static const std::vector<std::string> reservedKeys = { "date", "loudness", "nam_cpu_trainer" };

	MetadataPairs pairs;

	for (const std::string& arg : args)
	{
		size_t equals = arg.find('=');

		if (equals == std::string::npos)
			throw std::invalid_argument("\"" + arg + "\" should be key=value");

		std::string key = CommandLineToUtf8(arg.substr(0, equals));
		std::string value = CommandLineToUtf8(arg.substr(equals + 1));

		if (key.empty())
			throw std::invalid_argument("\"" + arg + "\" has an empty key");

		if (std::find(reservedKeys.begin(), reservedKeys.end(), key) != reservedKeys.end())
			throw std::invalid_argument("\"" + key + "\" is set by the trainer and can't be overridden");

		for (const auto& [existingKey, existingValue] : pairs)
		{
			if (existingKey == key)
				throw std::invalid_argument("\"" + key + "\" is given more than once");
		}

		pairs.emplace_back(std::move(key), std::move(value));
	}

	return pairs;
}

// JSON members for the metadata pairs, one per line, each followed by a comma, to sit in the template's metadata object
std::string MetadataToJson(const MetadataPairs& pairs)
{
	std::string json;

	for (const auto& [key, value] : pairs)
	{
		// Escaped by nlohmann; any invalid UTF-8 becomes U+FFFD instead of throwing
		auto quote = [](const std::string& text) { return nlohmann::json(text).dump(-1, ' ', false, nlohmann::json::error_handler_t::replace); };

		json += "\n    " + quote(key) + ": " + quote(value) + ",";
	}

	return json;
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

// The current date and time as a .nam "date" object
static std::string CurrentDateJson()
{
	auto now = std::chrono::system_clock::now();
	auto system_days = std::chrono::floor<std::chrono::days>(now);
	std::chrono::year_month_day ymd{ system_days };
	std::chrono::hh_mm_ss hms{ std::chrono::floor<std::chrono::seconds>(now - system_days) };

	return std::format(
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
}

// A trained model: its complete .nam JSON (with a {{USER_METADATA}} placeholder still in it, filled in when the file is
// written), and what a packed container needs to know about it
struct TrainedModel
{
	size_t Channels = 0;
	std::string Json;
	double Loudness = 0;
};

// Owns a buffer from drwav, freeing it when done
struct WavData
{
	float* Data = nullptr;

	~WavData() { drwav_free(Data, nullptr); }
};

// Trains one model and returns its .nam JSON, or nothing if training failed
template <typename ModelTrainer>
std::optional<TrainedModel> TrainNAM(ModelTrainer& trainer, const std::filesystem::path inWavePath, const std::filesystem::path targetWavePath, size_t maxEpochs, size_t numChannels, std::optional<float> inputNoiseDb, std::mt19937 noiseRand, const LossSpec& lossSpec, unsigned int seed)
{
	unsigned int channels;
	unsigned int sampleRate;
	drwav_uint64 numFrames;

	WavData inWav { drwav_open_file_and_read_pcm_frames_f32(inWavePath.string().c_str(), &channels, &sampleRate, &numFrames, nullptr) };
	WavData targetWav { drwav_open_file_and_read_pcm_frames_f32(targetWavePath.string().c_str(), &channels, &sampleRate, &numFrames, nullptr) };

	float* inData = inWav.Data;
	float* targetData = targetWav.Data;

	if ((inData == nullptr) || (targetData == nullptr))
	{
		std::cerr << "Couldn't read " << ((inData == nullptr) ? inWavePath : targetWavePath).string() << std::endl;

		return std::nullopt;
	}

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

		return std::nullopt;
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

	std::string dateStr = CurrentDateJson();

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

	// {{USER_METADATA}} is left in, to be filled in when the file is written: a plain model gets the user metadata, but the
	// models in a packed file don't (it goes in the container). Filling it in last also means placeholder-like text in
	// the user's values is left alone.
	std::vector<std::pair<std::string, std::string>> variables =
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

	TrainedModel model;

	model.Channels = numChannels;
	model.Json = jsonOutStr;
	model.Loudness = loudnessRMS;

	return model;
}

// Keeps the compiler happy
std::optional<TrainedModel> TrainNAM(std::nullptr_t& trainer, const std::filesystem::path inWavePath, const std::filesystem::path targetWavePath, size_t maxEpochs, size_t numChannels, std::optional<float> inputNoiseDb, std::mt19937 noiseRand, const LossSpec& lossSpec, unsigned int seed)
{
	(void)trainer;
	(void)inWavePath;
	(void)targetWavePath;
	(void)maxEpochs;
	(void)numChannels;
	(void)inputNoiseDb;
	(void)noiseRand;
	(void)lossSpec;
	(void)seed;

	return std::nullopt;
}

// Indents every line after the first, so multi-line JSON can be placed inside another JSON template
static std::string IndentJson(std::string json, const std::string& indent)
{
	// Templates start with a newline
	while (!json.empty() && ((json.front() == '\n') || (json.front() == '\r')))
		json.erase(0, 1);

	while (!json.empty() && ((json.back() == '\n') || (json.back() == '\r')))
		json.pop_back();

	ReplaceVariable(json, "\n", "\n" + indent);

	return json;
}

// A packed ("SlimmableContainer") .nam file holding the models, smallest first. A player picks the first model whose
// max_value is above its size setting (0 to 1), so the thresholds are spread evenly, with the largest model at 1.0.
// The top-level loudness is the largest model's.
static std::string PackedModelJson(const std::vector<TrainedModel>& models, const MetadataPairs& userMetadata, const LossSpec& lossSpec, unsigned int seed)
{
	std::string submodels;

	for (size_t i = 0; i < models.size(); i++)
	{
		const double maxValue = (i == models.size() - 1) ? 1.0 : (double)(i + 1) / (double)models.size();

		if (i > 0)
			submodels += ",\n";

		submodels += "      {\n";
		submodels += std::format("        \"max_value\": {:.9g},\n", maxValue);
		std::string modelJson = models[i].Json;

		ReplaceVariable(modelJson, "{{USER_METADATA}}", "");	// the user metadata goes in the container

		submodels += "        \"model\": " + IndentJson(modelJson, "        ") + "\n";
		submodels += "      }";
	}

	nlohmann::ordered_json trainerInfo;

	trainerInfo["version"] = NCT_VERSION_STRING;
	trainerInfo["loss"] = FormatLossSpec(lossSpec);
	trainerInfo["seed"] = seed;

	for (const TrainedModel& model : models)
		trainerInfo["channels"].push_back(model.Channels);

	std::string trainerInfoStr = trainerInfo.dump(2);

	ReplaceVariable(trainerInfoStr, "\n", "\n    ");

	// Substituted in this order. The submodels and user-supplied text go last, so placeholder-like text in them is left alone.
	std::vector<std::pair<std::string, std::string>> variables =
	{
		{"{{DATE}}", CurrentDateJson() },
		{"{{LOUDNESS}}", std::to_string(models.back().Loudness) },
		{"{{TRAINER_INFO}}", trainerInfoStr },
		{"{{SUBMODELS}}", submodels },
		{"{{USER_METADATA}}", MetadataToJson(userMetadata) }
	};

	std::string json = std::string(PackedJsonData);

	for (const auto& [variable, value] : variables)
	{
		ReplaceVariable(json, variable, value);
	}

	return json;
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

// Width to lay out --help for: the terminal's width, clamped to 80-120 columns (narrower is unreadable anyway, and
// much wider lines are hard to scan). When stdout isn't a terminal, the COLUMNS environment variable is used if set,
// otherwise 80.
static size_t TerminalWidth()
{
	size_t width = 0;

#if defined(_WIN32)
	CONSOLE_SCREEN_BUFFER_INFO info;

	if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info))
		width = (size_t)(info.srWindow.Right - info.srWindow.Left + 1);
#else
	struct winsize size;

	if (isatty(STDOUT_FILENO) && (ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0))
		width = size.ws_col;
#endif

	if (width == 0)
	{
		if (const char* columns = std::getenv("COLUMNS"))
			width = (size_t)std::strtoul(columns, nullptr, 10);
	}

	if (width == 0)
		width = 80;

	return std::clamp(width, (size_t)80, (size_t)120);
}

// argparse starts the help text at about column 32. Keep 2 columns spare at the right edge.
static constexpr size_t HelpColumn = 32;
static const size_t HelpWidth = TerminalWidth() - HelpColumn - 2;

// Word-wraps an argument's help text for --help. argparse lines up each line of the help under the help column, so
// lines are kept to HelpWidth characters to fit the terminal. Explicit newlines are kept, and a wrapped line keeps its
// leading indentation, plus two spaces. "annotation" is the note argparse appends to the last line
// ("[default: ...]", "[required]", "[may be repeated]"), so room is left for it there.

static std::string WrapHelp(std::string_view text, std::string_view annotation = "")
{
	// A placeholder word as long as the annotation (plus its space) is wrapped along with the text, then removed
	constexpr char placeholderChar = '\x1f';
	const std::string placeholder = annotation.empty() ? "" : std::string(annotation.size(), placeholderChar);

	std::string wrapped;
	size_t start = 0;

	while (start <= text.size())
	{
		size_t end = text.find('\n', start);

		if (end == std::string_view::npos)
			end = text.size();

		std::string_view paragraph = text.substr(start, end - start);
		const bool lastParagraph = (end == text.size());

		const size_t indentSize = paragraph.find_first_not_of(' ') == std::string_view::npos ? 0 : paragraph.find_first_not_of(' ');
		const std::string indent(indentSize, ' ');
		const std::string continuationIndent = indent + (indentSize > 0 ? "  " : "");

		std::vector<std::string> words;
		size_t pos = indentSize;

		while (pos < paragraph.size())
		{
			size_t wordEnd = paragraph.find(' ', pos);

			if (wordEnd == std::string_view::npos)
				wordEnd = paragraph.size();

			if (wordEnd > pos)
				words.emplace_back(paragraph.substr(pos, wordEnd - pos));

			pos = wordEnd + 1;
		}

		if (lastParagraph && !placeholder.empty())
			words.push_back(placeholder);

		std::string line = indent;
		bool lineHasWords = false;

		for (const std::string& word : words)
		{
			if (lineHasWords && (line.size() + 1 + word.size() > HelpWidth))
			{
				wrapped += line + "\n";
				line = continuationIndent;
				lineHasWords = false;
			}

			line += (lineHasWords ? " " : "") + word;
			lineHasWords = true;
		}

		wrapped += line;

		if (!lastParagraph)
			wrapped += "\n";

		start = end + 1;
	}

	if (!placeholder.empty())
	{
		const size_t placeholderPos = wrapped.find(placeholderChar);
		const size_t lineStart = (wrapped.rfind('\n', placeholderPos) == std::string::npos) ? 0 : (wrapped.rfind('\n', placeholderPos) + 1);

		if ((lineStart > 0) && (wrapped.find_first_not_of(' ', lineStart) == placeholderPos))
		{
			// The annotation didn't fit, so it gets its own line, at the help column whatever the indentation above it.
			// (argparse only lines up a final line that isn't empty, so it holds a single space.)
			wrapped.erase(lineStart);
			wrapped += ' ';
		}
		else
		{
			// Remove the placeholder and the space before it: argparse adds its own space before the annotation
			size_t removeFrom = placeholderPos;

			if ((removeFrom > 0) && (wrapped[removeFrom - 1] == ' '))
				removeFrom--;

			wrapped.erase(removeFrom);
		}
	}

	return wrapped;
}

// Parses --channels: one channel count, or a comma-separated list for a packed model. Returned smallest first.
// Throws std::invalid_argument with a readable message.
static std::vector<size_t> ParseChannels(const std::string& text)
{
	static const std::vector<size_t> supported = { 1, 2, 3, 4, 8, 16 };

	std::vector<size_t> channels;
	size_t start = 0;

	while (start <= text.size())
	{
		size_t end = text.find(',', start);

		if (end == std::string::npos)
			end = text.size();

		std::string item = text.substr(start, end - start);

		item.erase(0, item.find_first_not_of(' '));
		item.erase(item.find_last_not_of(' ') + 1);

		char* parseEnd = nullptr;
		unsigned long value = item.empty() ? 0 : std::strtoul(item.c_str(), &parseEnd, 10);

		if (item.empty() || (*parseEnd != 0))
			throw std::invalid_argument("\"" + text + "\" should be a channel count, or a comma-separated list of them");

		if (std::find(supported.begin(), supported.end(), (size_t)value) == supported.end())
			throw std::invalid_argument(std::to_string(value) + " channels isn't supported (supported: 1, 2, 3, 4, 8 and 16)");

		if (std::find(channels.begin(), channels.end(), (size_t)value) != channels.end())
			throw std::invalid_argument(std::to_string(value) + " channels is given more than once");

		channels.push_back((size_t)value);

		start = end + 1;
	}

	std::sort(channels.begin(), channels.end());

	return channels;
}

static std::string LossHelp()
{
	std::string help = "Training loss, as comma-separated name[:weight][:key=value...] terms. Losses:";

	for (auto& type : GetLossTypes())
	{
		help += "\n    " + type.Name + " - " + type.Description;

		if (!type.Params.empty())
		{
			help += "\n      defaults:";

			for (auto& [name, value] : type.Params)
				help += std::format(" {}={}", name, value);
		}
	}

	return help;
}

int main(int argc, char* argv[])
{	
	argparse::ArgumentParser program("nam-cpu-trainer", NCT_VERSION_STRING);

	program.set_usage_max_line_width(TerminalWidth() - 2);

	program.add_argument("-i", "--input")
		.nargs(1)
		.required()
		.metavar("<input.wav>")
		.help(WrapHelp("Input .wav file used for the capture", "[required]"));

	program.add_argument("-o", "--output")
		.nargs(1)
		.required()
		.metavar("<output.wav>")
		.help(WrapHelp("Output (captured) .wav file", "[required]"));

	program.add_argument("-c", "--channels")
		.default_value(std::string("3,8"))
		.nargs(1)
		.metavar("<channels>")
		.help(WrapHelp("Number of channels. A comma-separated list (like the default) trains a model of each size and packs them into one .nam file, like NAM's standard A2 (3 and 8 channels). A single value writes a plain model. Supported sizes: 1, 2, 3, 4, 8 and 16", "[default: \"3,8\"]"));

	program.add_argument("-t", "--threads")
		.nargs(1)
		.metavar("<numThreads>")
		.help(WrapHelp("Number of threads (defaults to the number of physical cores)"))
		.scan<'i', int>();

	program.add_argument("-e", "--epochs")
		.default_value(1000)
		.nargs(1, 1)
		.metavar("<maxEpochs>")
		.help(WrapHelp("Maximum number of epochs to train for", "[default: 1000]"))
		.scan<'i', int>();

	program.add_argument("-r", "--rand")
		.nargs(1, 1)
		.metavar("<randomSeed>")
		.help(WrapHelp("Random seed, for repeatability (by default a random seed is used, and recorded in the model)"))
		.scan<'i', int>();

	program.add_argument("--input-noise")
		.nargs(1, 1)
		.metavar("<dBFS>")
		.help(WrapHelp("Add Gaussian noise at this RMS level (e.g. -75) to the training input (off by default)"))
		.scan<'g', float>();

	program.add_argument("-l", "--loss")
		.default_value(std::string("mse:1,mrstft:0.0005"))
		.nargs(1)
		.metavar("<lossSpec>")
		.help(WrapHelp(LossHelp(), "[default: \"mse:1,mrstft:0.0005\"]"));

	program.add_argument("-m", "--metadata")
		.append()
		.metavar("<key=value>")
		.help(WrapHelp("Add a key=value string to the model's metadata. Can be given more than once, e.g.:\n"
			"    -m name=\"Boss SD-1\" -m modeled_by=Mike\n"
			"NAM's standard keys include name, modeled_by, gear_make, gear_model, gear_type and tone_type", "[may be repeated]"));

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

	std::vector<size_t> channelSizes;

	try
	{
		channelSizes = ParseChannels(program.get("--channels"));
	}
	catch (const std::invalid_argument& e)
	{
		std::cerr << std::endl << "Invalid --channels: " << e.what() << std::endl;

		return 1;
	}

	size_t numThreads = program.is_used("--threads") ? (size_t)program.get<int>("--threads") : 0;

	size_t maxEpochs = ((size_t)program.get<int>("--epochs"));

	// Always use an explicit seed, so it can be recorded in the model and the run reproduced
	unsigned int seed = program.is_used("--rand") ? (unsigned int)program.get<int>("--rand") : std::random_device{}();

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

	MetadataPairs userMetadata;

	try
	{
		if (program.is_used("--metadata"))
			userMetadata = ParseMetadata(program.get<std::vector<std::string>>("--metadata"));
	}
	catch (const std::invalid_argument& e)
	{
		std::cerr << std::endl << "Invalid --metadata: " << e.what() << std::endl;

		return 1;
	}

	const bool packed = (channelSizes.size() > 1);

	std::string channelList;

	for (size_t c : channelSizes)
		channelList += (channelList.empty() ? "" : ", ") + std::to_string(c);

	std::cout << "Training NAM A2 with " << channelList << " channels" << (packed ? " (packed into one model)" : "") << std::endl;
	std::cout << "Loss: " << FormatLossSpec(lossSpec) << std::endl;
	std::cout << "Random seed: " << seed << std::endl;

	for (const auto& [key, value] : userMetadata)
		std::cout << "Metadata: " << key << " = " << value << std::endl;

	std::optional<float> inputNoiseDb;

	if (program.is_used("--input-noise"))
	{
		inputNoiseDb = program.get<float>("--input-noise");
	}

	// Ctrl-C stops training after the current epoch, skipping any remaining model sizes. What has been trained is written.
	std::signal(SIGINT, SignalHandler);

	// Smallest first: it trains fastest, so a capture that isn't converging shows up soonest
	std::vector<TrainedModel> models;

	for (size_t i = 0; i < channelSizes.size(); i++)
	{
		const size_t numChannels = channelSizes[i];

		if (packed)
			std::cout << std::endl << "=== " << numChannels << " channels (model " << (i + 1) << " of " << channelSizes.size() << ") ===" << std::endl;

		// Each size starts from the same seed, so its model is the same as training that size on its own
		std::mt19937 rand(seed);

		A2Types modelTrainerObj = GetTrainer(numChannels, numThreads, rand);

		// Seeded from the (already copied) trainer seed, so --rand makes the noise repeatable too
		std::mt19937 noiseRand(rand());

		std::optional<TrainedModel> model;

		std::visit([&](auto& modelTrainer)
		{
			model = TrainNAM(modelTrainer, inputPath, capturePath, maxEpochs, numChannels, inputNoiseDb, noiseRand, lossSpec, seed);

		}, modelTrainerObj);

		if (!model)
			return 1;

		models.push_back(std::move(*model));

		if (keepRunning == 0)
			break;	// stopped with ctrl-c: skip any remaining sizes
	}

	// Two or more trained models are packed. A single one (including when a packed run was stopped during its first
	// model) is written as a plain model.
	std::string modelJson;

	if (models.size() > 1)
	{
		modelJson = PackedModelJson(models, userMetadata, lossSpec, seed);
	}
	else
	{
		modelJson = models.front().Json;

		ReplaceVariable(modelJson, "{{USER_METADATA}}", MetadataToJson(userMetadata));
	}

	std::ofstream namStream(outputNAMPath);

	namStream << modelJson;

	std::string writtenSizes;

	for (const TrainedModel& trained : models)
		writtenSizes += (writtenSizes.empty() ? "" : ", ") + std::to_string(trained.Channels);

	std::cout << std::endl;

	if (models.size() < channelSizes.size())
		std::cout << "Stopped early: only the " << writtenSizes << " channel model" << ((models.size() > 1) ? "s were" : " was") << " trained" << std::endl;

	std::cout << "Wrote " << ((models.size() > 1) ? "packed model" : "model") << " (" << writtenSizes << " channels) to " << outputNAMPath.string() << std::endl;

	return 0;
}

