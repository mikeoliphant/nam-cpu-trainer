#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <xmmintrin.h>

#include "NeuralModel.h"
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#include "WaveNet.h"
#include "WaveNetBackprop.h"
#include "ModelTrainer.h"
#include "NAM.h"
#include "Dataset.h"

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

	modelTrainer->GetModel()->SetWeights(it);
	modelTrainer->GetModel()->SetHeadScale(*it);

	std::vector<float> verifyOutput(numSamples);

	modelTrainer->VerifyModel(input.data(), verifyOutput.data(), numSamples);

	MSELossT<float> mseLoss;

	size_t receptiveField = modelTrainer->GetReceptiveField();

	double err = mseLoss.GetTotSquared(verifyOutput.data() + receptiveField, namOutput.data() + receptiveField, numSamples - receptiveField) / (double)(numSamples - receptiveField);

	std::cout << "MSE: " << err << std::endl;
}

template <typename ModelTrainer>
void TrainWav(ModelTrainer& trainer, const std::filesystem::path inWavePath, const std::filesystem::path targetWavePath)
{
	unsigned int channels;
	unsigned int sampleRate;
	drwav_uint64 numFrames;

	float* inData = drwav_open_file_and_read_pcm_frames_f32(inWavePath.string().c_str(), &channels, &sampleRate, &numFrames, nullptr);
	float* targetData = drwav_open_file_and_read_pcm_frames_f32(targetWavePath.string().c_str(), &channels, &sampleRate, &numFrames, nullptr);

	size_t startOffset = 48000 * 13;

	size_t verifyFrames = 48000 * 9;

	size_t frameDelay = 0;

	//startOffset = 0;
	//verifyFrames = (size_t)(numFrames * 0.1f);

	size_t verifyOffset = (size_t)numFrames - verifyFrames;

	trainer->TrainModel(inData + startOffset - frameDelay, targetData + startOffset, (size_t)numFrames - verifyFrames - startOffset - frameDelay, inData + verifyOffset - frameDelay, targetData + verifyOffset, verifyFrames - frameDelay);
}


int main()
{
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

	DataGen dataGen(123);

	size_t numSamples = 48000 * 180;

	auto randData = dataGen.GenerateRandom(numSamples);
	auto sinData = dataGen.GenerateSin(numSamples, 8192);
	auto delayData = dataGen.GenerateDelay(256, numSamples);
	auto xorData = dataGen.GenerateXOR(1, numSamples);

	//TestNAM(R"(C:\Code\NeuralCpuTrainer\BossWN-a2lite.nam)");

	using TestKernelSizes = std::integer_sequence<int, 6>;//, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6>;
	using TestDilations = std::integer_sequence<int, 1>;//, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239>;

	//auto a2 = new A2BackpropT<float, 1, 3, TestKernelSizes, TestDilations>();
	//auto a2 = new ();

	//std::cout << sizeof(A2BackpropT<float, 1, 8, A2KernelSizes, A2Dilations>) << std::endl;

	auto modelTrainer = new ModelTrainerT<float, A2BackpropT<float, 1, 3, A2KernelSizes, A2Dilations>>();

	//modelTrainer->TestBackprop(0, randData.data(), randData.data(), MAX_BATCH_SIZE);

	//modelTrainer->TrainIdentity(sinData);

	TrainWav(modelTrainer, R"(C:\Share\Recordings\NAM\NAMv3Input.wav)", R"(C:\Share\Recordings\NAM\BossSD1Capture.wav)");

	return 0;
}

