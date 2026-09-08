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
#include "WaveNet.h"
#include "WaveNetBackprop.h"
#include "ModelTrainer.h"
#include "NAM.h"
#include "Dataset.h"
#include "Tests.h"

using namespace NeuralAudio;
using namespace NeuralCpuTrain;

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


	//auto denseTrainer = new ModelTrainerT<float, DenseBackpropT<float, 1, 1, false>>();

	//denseTrainer->TestBackprop(0, randData.data(), randData.data(), MAX_BATCH_SIZE);

	//denseTrainer->TrainIdentity(randData);


	//denseTrainter->TestWav(R"(C:\Share\Recordings\NAM\v1_1_1.wav)", R"(C:\Share\Recordings\NAM\v1_1_1.wav)");

	//auto convTrainer = new ModelTrainerT<float, Conv1DBackpropT<float, 1, 1, 2, true, 128>>();

	//convTrainer->Train(delayData);

	////convTrainer->TestWav(R"(C:\Share\Recordings\NAM\v1_1_1.wav)", R"(C:\Share\Recordings\NAM\v1_1_1.wav)");

	//WaveNetLayerBackpropT<float, 1, 3, 1> wn;
	//TestModel(wn);

	//auto twoConvTrainer = new ModelTrainerT<float, TwoConvTestT<float, 1, 2, 128>>();

	//twoConvTrainer->Train(delayData);

	//auto convTestTrainer = new ModelTrainerT<float, ConvTestT<float, 1, 16, 3, 1, 1>>();

	//convTestTrainer->TrainIdentity(randData);

	//convTestTrainer->Train(xorData);

	//convTestTrainer->TestWav(R"(C:\Share\Recordings\NAM\v1_1_1.wav)", R"(C:\Share\Recordings\NAM\v1_1_1.wav)");
	//convTestTrainer->TestWav(R"(C:\Share\Recordings\NAM\v1_1_1.wav)", R"(C:\Share\Recordings\NAM\BossSD1.wav)");

	//auto a2 = new A2BackpropT<float, 1, 3, A2KernelSizes, A2Dilations>();

	using TestKernelSizes = std::integer_sequence<int, 6>;//, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6>;
	using TestDilations = std::integer_sequence<int, 1>;//, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239>;

	//auto a2 = new A2BackpropT<float, 1, 3, TestKernelSizes, TestDilations>();
	//auto a2 = new ();

	//std::cout << sizeof(A2BackpropT<float, 1, 8, A2KernelSizes, A2Dilations>) << std::endl;

	auto modelTrainer = new ModelTrainerT<float, A2BackpropT<float, 1, 3, A2KernelSizes, A2Dilations>>();

	//modelTrainer->TestBackprop(0, randData.data(), randData.data(), MAX_BATCH_SIZE);

	//modelTrainer->TrainIdentity(sinData);

	modelTrainer->TrainWav(R"(C:\Share\Recordings\NAM\NAMv3Input.wav)", R"(C:\Share\Recordings\NAM\BossSD1Capture.wav)");

	//ChainBackpropModelT<float, 1, 1> chainBackProp;

	//auto layer1 = std::make_unique<DenseBackpropT<float, 1, 2, false>>();
	//auto layer2 = std::make_unique<DenseBackpropT<float, 2, 1, false>>();

	//chainBackProp.AddLayer(std::move(layer1));
	//chainBackProp.AddLayer(std::move(layer2));

	//TestModel(chainBackProp);

	return 0;
}

