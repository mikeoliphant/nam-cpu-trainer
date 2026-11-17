#pragma once

#include "ModelTrainer.h"
#include "Dense.h"
#include "Conv1D.h"
#include "Activation.h"
#include "TemplateHelper.h"

#include <string_view>

using namespace cpugrad;

// .nam config "activation" entry for each supported activation
template <template <typename, int> class ActivationType>
struct NamActivationJson;

template <>
struct NamActivationJson<LeakyReLUT>
{
	static constexpr std::string_view Value = R"({ "type": "LeakyReLU", "negative_slope": 0.01 })";
};

template <>
struct NamActivationJson<FastTanhT>
{
	static constexpr std::string_view Value = R"({ "type": "Fasttanh" })";
};

template <typename T, int ConditionSize, int Channels, int KernelSize, int Dilation, template <typename, int> class ActivationType = LeakyReLUT>
class WaveNetLayerBackpropT : public BackpropModelT<T, Channels, Channels>
{
public:
	using BackpropModelT<T, Channels, Channels>::Forward;
	using BackpropModelT<T, Channels, Channels>::Backward;
	using BackpropModelT<T, Channels, Channels>::trainingContext;

	void Forward(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, ConditionSize>& condition, const ChannelRowSpan<T, Channels>& output, const ChannelRowSpan<T, Channels>& headOutput)
	{
		assert(input.GetNumCols() == (output.GetNumCols() + GetReceptiveField()));

		ForwardNoLayerOutput(input, condition, headOutput);

		const size_t numSamplesOut = input.GetNumCols() - conv.GetReceptiveField();

		// Layer output: 1x1 of the activation, plus the residual input
		oneByOne.Forward(activationOut, output);

		auto outputMap = output.GetEigenMap();
		size_t inputOffset = input.GetNumCols() - numSamplesOut;
		outputMap.noalias() += input.Slice(inputOffset, numSamplesOut).GetEigenMapConst();
	}

	// Forward without the layer output, for the last layer, whose output nothing uses (only its contribution to the
	// head). The matching backward pass is BackwardNoLayerOutput().
	void ForwardNoLayerOutput(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, ConditionSize>& condition, const ChannelRowSpan<T, Channels>& headOutput)
	{
		const size_t numSamplesIn = input.GetNumCols();
		const size_t numSamplesOut = numSamplesIn - conv.GetReceptiveField();

		trainingContext->GetBufferArena().template GetBuffer<Channels>(activationOut, numSamplesOut);

		// If the activation can do backward from its output, the pre-activation values aren't kept: the conv
		// is computed straight into activationOut and the activation applied in place
		if constexpr (StorePreActivation)
			trainingContext->GetBufferArena().template GetBuffer<Channels>(convOut, numSamplesOut);

		auto& preActivation = StorePreActivation ? convOut : activationOut;

		size_t conditionOffset = condition.GetNumCols() - numSamplesOut;
		conditionMixIn.Forward(condition.Slice(conditionOffset, numSamplesOut), preActivation);

		conv.Forward(input, preActivation);

		activation.Forward(preActivation, activationOut);

		const size_t headOutputSamples = headOutput.GetNumCols();

		auto headOutputMap = headOutput.GetEigenMap();
		headOutputMap.noalias() += activationOut.Slice(numSamplesOut - headOutputSamples, headOutputSamples).GetEigenMapConst();
	}

	void Backward(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, ConditionSize>& condition, const ChannelRowSpan<T, Channels>& dOutput, const ChannelRowSpan<T, Channels>& dHeadOutput, const ChannelRowSpan<T, Channels>& dInput)
	{
		assert((input.GetNumCols() == (dOutput.GetNumCols() + GetReceptiveField())) && (dInput.GetNumCols() == dInput.GetNumCols()));

		const size_t numSamplesIn = dInput.GetNumCols();
		const size_t numSamplesOut = numSamplesIn - conv.GetReceptiveField();

		// Doing the skip connection first means we don't need to clear dInput
		size_t inputOffset = dInput.GetNumCols() - numSamplesOut;
		auto dInputValidMap = dInput.Slice(inputOffset, numSamplesOut).GetEigenMap();
		dInputValidMap.noalias() = dOutput.GetEigenMapConst();
		dInput.Slice(0, inputOffset).SetZero(); // clear receptive field samples

		auto scratch = trainingContext->GetBufferArena().template GetScratchBuffer<Channels>(numSamplesOut);

		oneByOne.Backward(activationOut, dOutput, scratch);

		const size_t headOutputSize = dHeadOutput.GetNumCols();	// dHeadOutput is always smaller
		auto dOneByOneOutMap = scratch.Slice(numSamplesOut - headOutputSize, headOutputSize).GetEigenMap();
		dOneByOneOutMap.noalias() += dHeadOutput.GetEigenMapConst();

		ActivationBackward(0, numSamplesOut, scratch, scratch);	// activation backward can be done in place

		size_t conditionOffset = condition.GetNumCols() - numSamplesOut;
		conditionMixIn.BackwardNoDInput(condition.Slice(conditionOffset, numSamplesOut), scratch);

		conv.Backward(input, scratch, dInput);

		trainingContext->GetBufferArena().FreeScratchBuffer(scratch);
	}

	void BackwardNoLayerOutput(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, ConditionSize>& condition, const ChannelRowSpan<T, Channels>& dHeadOutput, const ChannelRowSpan<T, Channels>& dInput)
	{
		assert(input.GetNumCols() == dInput.GetNumCols());

		const size_t numSamplesIn = dInput.GetNumCols();
		const size_t offset = conv.GetReceptiveField();
		const size_t numSamplesOut = numSamplesIn - offset;

		dInput.SetZero();

		const size_t headOutputSize = dHeadOutput.GetNumCols();

		auto dActivationOut = trainingContext->GetBufferArena().template GetScratchBuffer<Channels>(numSamplesOut);

		ActivationBackward(numSamplesOut - headOutputSize, headOutputSize, dHeadOutput, dActivationOut.Slice(numSamplesOut - headOutputSize, headOutputSize));
		dActivationOut.Slice(0, numSamplesOut - headOutputSize).SetZero();	// Zero the samples we didn't touch

		conditionMixIn.BackwardNoDInput(condition.Slice(offset, numSamplesOut), dActivationOut);

		conv.Backward(input, dActivationOut, dInput);

		trainingContext->GetBufferArena().FreeScratchBuffer(dActivationOut);
	}

	size_t GetReceptiveField() override
	{
		return conv.GetReceptiveField();
	}

	size_t GetNumWeights() override
	{
		return conv.GetNumWeights() + oneByOne.GetNumWeights() + conditionMixIn.GetNumWeights();;
	}

	void RandomizeWeights(std::mt19937& rand) override
	{
		conv.RandomizeWeights(rand);
		oneByOne.RandomizeWeights(rand);
		conditionMixIn.RandomizeWeights(rand);
	}

	void SetWeights(std::vector<float>::iterator& inWeights) override
	{
		conv.SetWeights(inWeights);
		conditionMixIn.SetWeights(inWeights);
		oneByOne.SetWeights(inWeights);
	}

	void GetWeights(std::vector<float>::iterator& outWeights) override
	{
		conv.GetWeights(outWeights);
		conditionMixIn.GetWeights(outWeights);
		oneByOne.GetWeights(outWeights);
	}


	void SetTrainingContext(TrainingContextT<T>* context) override
	{
		BackpropModelT<T, Channels, Channels>::SetTrainingContext(context);

		conv.SetTrainingContext(context);
		conditionMixIn.SetTrainingContext(context);
		oneByOne.SetTrainingContext(context);
	}

private:
	using ActivationT = ActivationType<T, Channels>;

	// Keep the pre-activation values only if the activation's backward needs them
	static constexpr bool StorePreActivation = !ActivationT::CanBackwardFromOutput;

	// Activation gradient for columns [startCol, startCol + numCols) of this layer's activation
	void ActivationBackward(size_t startCol, size_t numCols, const ChannelRowSpan<T, Channels>& dOutput, const ChannelRowSpan<T, Channels>& dInput)
	{
		if constexpr (StorePreActivation)
			activation.Backward(convOut.Slice(startCol, numCols), dOutput, dInput);
		else
			activation.BackwardFromOutput(activationOut.Slice(startCol, numCols), dOutput, dInput);
	}

	Conv1DBackpropT<T, Channels, Channels, KernelSize, true, Dilation> conv;
	ChannelBufferDynamic<T, Channels> convOut;	// only used if StorePreActivation
	DenseBackpropT<T, ConditionSize, Channels, false> conditionMixIn;
	ActivationT activation;
	ChannelBufferDynamic<T, Channels> activationOut;
	DenseBackpropT<T, Channels, Channels, true> oneByOne;
};

template <typename T, int InOutChannels, int Channels, typename KernelSizeSequence, typename DilationsSequence, template <typename, int> class ActivationType = LeakyReLUT>
class A2BackpropT : public BackpropModelT<T, InOutChannels, InOutChannels>
{
	using BackpropModelT<T, InOutChannels, InOutChannels>::trainingContext;

	template <typename, typename>
	struct LayersHelper
	{};

	template <int... values>
	using Dilations = std::integer_sequence<int, values...>;

	template <int... values>
	using KernelSizes = std::integer_sequence<int, values...>;

	template <int... dilationVals, int... kernelSizeVals>
	struct LayersHelper<KernelSizes<kernelSizeVals...>, Dilations<dilationVals...>>
	{
		using type = std::tuple<WaveNetLayerBackpropT<T, InOutChannels, Channels, kernelSizeVals, dilationVals, ActivationType>...>;
	};

	using Layers = typename LayersHelper<KernelSizeSequence, DilationsSequence>::type;

public:
	Layers layers;
	static constexpr auto NumLayers = std::tuple_size_v<decltype (layers)>;

	// The per-layer "activation" entry for the .nam config
	static constexpr std::string_view ActivationJson = NamActivationJson<ActivationType>::Value;

	void Forward(const ChannelRowSpan<T, InOutChannels>& input, const ChannelRowSpan<T, InOutChannels>& output) override
	{
		assert(input.GetNumCols() == (output.GetNumCols() + GetReceptiveField()));

		if (headOutput.GetNumCols() == 0)
		{
			const size_t numSamplesIn = input.GetNumCols();

			trainingContext->GetBufferArena().template GetBuffer<Channels>(layerArrayRechannelOut, numSamplesIn);

			size_t currentSize = numSamplesIn;

			ForEachIndex<NumLayers>([&](auto layerIndex)
				{
					currentSize -= std::get<layerIndex>(layers).GetReceptiveField();

					// The last layer's output isn't used, so it doesn't get a buffer.
					// (GCC needs "this->" to find the base-class member inside "if constexpr" in a generic lambda.)
					if constexpr (layerIndex < NumLayers - 1)
						this->trainingContext->GetBufferArena().template GetBuffer<Channels>(layerOuts[layerIndex], currentSize);
				});

			trainingContext->GetBufferArena().template GetBuffer<Channels>(headOutput, currentSize);
		}
		
		headOutput.SetZero();

		layerArrayRechannel.Forward(input, layerArrayRechannelOut);

		ForEachIndex<NumLayers>([&](auto layerIndex)
			{
				auto& layer = std::get<layerIndex>(layers);

				ChannelBufferDynamic<T, Channels>* layerInputPtr;

				if constexpr (layerIndex == 0)
					layerInputPtr = &layerArrayRechannelOut;
				else
					layerInputPtr = &layerOuts[layerIndex - 1];

				auto& layerInput = *layerInputPtr;

				// Only the last layer's contribution to the head is used, not its output
				if constexpr (layerIndex == NumLayers - 1)
					layer.ForwardNoLayerOutput(layerInput, input, headOutput);
				else
					layer.Forward(layerInput, input, layerOuts[layerIndex], headOutput);
			});

		oneByOne.Forward(headOutput, output);

		auto outputMap = output.GetEigenMap();
		outputMap *= headScale;
	}

	void Backward(const ChannelRowSpan<T, InOutChannels>& input, const ChannelRowSpan<T, InOutChannels>& dOutput, const ChannelRowSpan<T, InOutChannels>& dInput) override
	{
		(void)dInput;

		size_t currentSize = dOutput.GetNumCols();

		auto dOutputMap = dOutput.GetEigenMap();
		dOutputMap *= headScale;

		currentSize += oneByOne.GetReceptiveField();

		trainingContext->GetBufferArena().template GetBuffer<Channels>(dHeadRechannelOut, currentSize);

		dHeadRechannelOut.SetZero();
		oneByOne.Backward(headOutput, dOutput, dHeadRechannelOut);

		trainingContext->GetBufferArena().template GetBuffer<Channels>(dLastLayerOut, input.GetNumCols());
		trainingContext->GetBufferArena().template GetBuffer<Channels>(dCurrentLayerOut, input.GetNumCols());
		ChannelBufferDynamic<T, Channels> dTmpLayerOut;

		size_t lastLayerSize = currentSize;

		ForEachIndex<NumLayers>([&](auto layerIndexForward)
			{
				constexpr auto layerIndexBackward = NumLayers - 1 - layerIndexForward;

				currentSize += std::get<layerIndexBackward>(layers).GetReceptiveField();

				if constexpr (layerIndexForward == 0)
				{
					std::get<layerIndexBackward>(layers).BackwardNoLayerOutput(layerOuts[layerIndexBackward - 1], input, dHeadRechannelOut,	dCurrentLayerOut.Slice(currentSize));
				}
				else if constexpr (layerIndexBackward > 0)
				{
					std::get<layerIndexBackward>(layers).Backward(layerOuts[layerIndexBackward - 1], input,	dLastLayerOut.Slice(lastLayerSize), dHeadRechannelOut, dCurrentLayerOut.Slice(currentSize));
				}
				else
				{
					std::get<layerIndexBackward>(layers).Backward(layerArrayRechannelOut, input, dLastLayerOut.Slice(lastLayerSize), dHeadRechannelOut, dCurrentLayerOut.Slice(currentSize));
				}

				dTmpLayerOut = dLastLayerOut;
				dLastLayerOut = dCurrentLayerOut;
				dCurrentLayerOut = dTmpLayerOut;
				lastLayerSize = currentSize;
			});

		layerArrayRechannel.BackwardNoDInput(input, dLastLayerOut);	// skip dInput gradient calculation since it isn't used
	}

	size_t GetReceptiveField() override
	{
		size_t fieldSize = 0;

		ForEachIndex<NumLayers>([&](auto layerIndex)
			{
				fieldSize += std::get<layerIndex>(layers).GetReceptiveField();
			});

		return fieldSize + oneByOne.GetReceptiveField();
	}

	size_t GetNumWeights() override
	{
		size_t numWeights = 0;

		ForEachIndex<NumLayers>([&](auto layerIndex)
			{
				numWeights += std::get<layerIndex>(layers).GetNumWeights();
			});

		return numWeights + oneByOne.GetNumWeights() + layerArrayRechannel.GetNumWeights();
	}

	size_t GetMaxScratchBufferSize(size_t inputBufferSize) override
	{
		return Channels * (inputBufferSize + GetReceptiveField());
	}

	void SetWeights(std::vector<float>::iterator& inWeights) override
	{
		layerArrayRechannel.SetWeights(inWeights);

		ForEachIndex<NumLayers>([&](auto layerIndex)
			{
				std::get<layerIndex>(layers).SetWeights(inWeights);
			});

		oneByOne.SetWeights(inWeights);
	}

	void GetWeights(std::vector<float>::iterator& outWeights) override
	{
		layerArrayRechannel.GetWeights(outWeights);

		ForEachIndex<NumLayers>([&](auto layerIndex)
			{
				std::get<layerIndex>(layers).GetWeights(outWeights);
			});

		oneByOne.GetWeights(outWeights);
	}

	void SetHeadScale(T scale)
	{
		this->headScale = scale;
	}

	T GetHeadScale()
	{
		return this->headScale;
	}

	void SetTrainingContext(TrainingContextT<T>* context) override
	{
		BackpropModelT<T, InOutChannels, InOutChannels>::SetTrainingContext(context);

		layerArrayRechannel.SetTrainingContext(context);
		ForEachIndex<NumLayers>([&](auto layerIndex)
			{
				std::get<layerIndex>(layers).SetTrainingContext(context);
			});

		oneByOne.SetTrainingContext(context);
	}

	void RandomizeWeights(std::mt19937& rand) override
	{
		layerArrayRechannel.RandomizeWeights(rand);

		ForEachIndex<NumLayers>([&](auto layerIndex)
			{
				std::get<layerIndex>(layers).RandomizeWeights(rand);
			});

		oneByOne.RandomizeWeights(rand);
	}

private:
	DenseBackpropT<T, InOutChannels, Channels, false> layerArrayRechannel;
	ChannelBufferDynamic<T, Channels> layerArrayRechannelOut;

	ChannelBufferDynamic<T, Channels> layerOuts[NumLayers];

	ChannelBufferDynamic<T, Channels> dLastLayerOut;
	ChannelBufferDynamic<T, Channels> dCurrentLayerOut;

	ChannelBufferDynamic<T, Channels> headOutput;
	Conv1DBackpropT<T, Channels, InOutChannels, 16, true, 1> oneByOne;
	ChannelBufferDynamic<T, Channels> dHeadRechannelOut;
	float headScale = 0.01f;	// matches NAM's head_scale
};

using A2KernelSizes = std::integer_sequence<int, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6>;
using A2Dilations = std::integer_sequence<int, 1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239>;

