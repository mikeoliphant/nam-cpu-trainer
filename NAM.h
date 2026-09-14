#pragma once

#include "ModelTrainer.h"
#include "Dense.h"
#include "Conv1D.h"

using namespace cpugrad;

template <typename T, int ConditionSize, int Channels, int KernelSize, int Dilation>
class WaveNetLayerBackpropT : public BackpropModelT<T, Channels, Channels>
{
public:
	using BackpropModelT<T, Channels, Channels>::Forward;
	using BackpropModelT<T, Channels, Channels>::Backward;
	using BackpropModelT<T, Channels, Channels>::trainingContext;

	void Forward(const ChannelRowSpan<T, Channels>& input, const ChannelRowSpan<T, ConditionSize>& condition, const ChannelRowSpan<T, Channels>& output, const ChannelRowSpan<T, Channels>& headOutput)
	{
		assert(input.GetNumCols() == (output.GetNumCols() + GetReceptiveField()));

		const size_t numSamplesIn = input.GetNumCols();
		const size_t numSamplesOut = numSamplesIn - conv.GetReceptiveField();

		if (convOut.GetNumCols() == 0)
			convOut = trainingContext->GetBufferArena().template GetBuffer<Channels>(numSamplesOut);

		size_t conditionOffset = condition.GetNumCols() - numSamplesOut;
		conditionMixIn.Forward(condition.Slice(conditionOffset, numSamplesOut), convOut);

		conv.Forward(input, convOut);

		if (reluOut.GetNumCols() == 0)
			reluOut = trainingContext->GetBufferArena().template GetBuffer<Channels>(numSamplesOut);
		relu.Forward(convOut, reluOut);

		const size_t headOutputSamples = headOutput.GetNumCols();

		auto headOutputMap = headOutput.GetEigenMap();
		headOutputMap.noalias() += reluOut.Slice(numSamplesOut - headOutputSamples, headOutputSamples).GetEigenMapConst();

		// Not needed on last layer - can optimize
		oneByOne.Forward(reluOut, output);	

		auto outputMap = output.GetEigenMap();
		size_t inputOffset = input.GetNumCols() - numSamplesOut;
		outputMap.noalias() += input.Slice(inputOffset, numSamplesOut).GetEigenMapConst();
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

		oneByOne.Backward(reluOut, dOutput, scratch);

		const size_t headOutputSize = dHeadOutput.GetNumCols();	// dHeadOutput is always smaller
		auto dOneByOneOutMap = scratch.Slice(numSamplesOut - headOutputSize, headOutputSize).GetEigenMap();
		dOneByOneOutMap.noalias() += dHeadOutput.GetEigenMapConst();

		// only need to know if convOut is < 0, so we could store it as a bitmask for better memory/performance
		relu.Backward(convOut, scratch, scratch);	// relu can be done in place

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

		auto dReluOut = trainingContext->GetBufferArena().template GetScratchBuffer<Channels>(numSamplesOut);

		relu.Backward(convOut.Slice(numSamplesOut - headOutputSize, headOutputSize), dHeadOutput, dReluOut.Slice(numSamplesOut - headOutputSize, headOutputSize));
		dReluOut.Slice(0, numSamplesOut - headOutputSize).SetZero();	// Zero the samples we didn't touch

		conditionMixIn.BackwardNoDInput(condition.Slice(offset, numSamplesOut), dReluOut);

		conv.Backward(input, dReluOut, dInput);

		trainingContext->GetBufferArena().FreeScratchBuffer(dReluOut);
	}

	size_t GetReceptiveField() override
	{
		return conv.GetReceptiveField();
	}

	size_t GetNumWeights() override
	{
		return conv.GetNumWeights() + oneByOne.GetNumWeights() + conditionMixIn.GetNumWeights();;
	}

	void RandomizeWeights() override
	{
		conv.RandomizeWeights();
		oneByOne.RandomizeWeights();
		conditionMixIn.RandomizeWeights();
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
	Conv1DBackpropT<T, Channels, Channels, KernelSize, true, Dilation> conv;
	ChannelBufferDynamic<T, Channels> convOut;
	DenseBackpropT<T, ConditionSize, Channels, false> conditionMixIn;
	LeakyReLUT<T, Channels> relu;
	ChannelBufferDynamic<T, Channels> reluOut;
	DenseBackpropT<T, Channels, Channels, true> oneByOne;
};

template <typename T, int InOutChannels, int Channels, typename KernelSizeSequence, typename DilationsSequence>
class A2BackpropT : public BackpropModelT<T, InOutChannels, InOutChannels>
{
	using BackpropModelT<T, InOutChannels, InOutChannels>::trainingContext;

	template <typename, typename>
	struct LayersHelper
	{};

	template <int... dilationVals, int... kernelSizeVals>
	struct LayersHelper<KernelSizes<kernelSizeVals...>, Dilations<dilationVals...>>
	{
		using type = std::tuple<WaveNetLayerBackpropT<T, InOutChannels, Channels, kernelSizeVals, dilationVals>...>;
	};

	using Layers = typename LayersHelper<KernelSizeSequence, DilationsSequence>::type;

public:
	Layers layers;
	static constexpr auto NumLayers = std::tuple_size_v<decltype (layers)>;

	void Forward(const ChannelRowSpan<T, InOutChannels>& input, const ChannelRowSpan<T, InOutChannels>& output) override
	{
		assert(input.GetNumCols() == (output.GetNumCols() + GetReceptiveField()));

		if (headOutput.GetNumCols() == 0)
		{
			const size_t numSamplesIn = input.GetNumCols();

			layerArrayRechannelOut = trainingContext->GetBufferArena().template GetBuffer<Channels>(numSamplesIn);

			size_t currentSize = numSamplesIn;

			ForEachIndex<NumLayers>([&](auto layerIndex)
				{
					currentSize -= std::get<layerIndex>(layers).GetReceptiveField();

					layerOuts[layerIndex] = trainingContext->GetBufferArena().template GetBuffer<Channels>(currentSize);
				});

			headOutput = trainingContext->GetBufferArena().template GetBuffer<Channels>(currentSize);
		}
		
		headOutput.SetZero();

		layerArrayRechannel.Forward(input, layerArrayRechannelOut);

		ForEachIndex<NumLayers>([&](auto layerIndex)
			{
				if constexpr (layerIndex == 0)
				{
					std::get<layerIndex>(layers).Forward(layerArrayRechannelOut, input, layerOuts[layerIndex], headOutput);
				}
				else
				{
					std::get<layerIndex>(layers).Forward(layerOuts[layerIndex - 1], input, layerOuts[layerIndex], headOutput);
				}
			});

		oneByOne.Forward(headOutput, output);

		auto outputMap = output.GetEigenMap();
		outputMap *= headScale;
	}

	void Backward(const ChannelRowSpan<T, InOutChannels>& input, const ChannelRowSpan<T, InOutChannels>& dOutput, const ChannelRowSpan<T, InOutChannels>& dInput) override
	{
		size_t currentSize = dOutput.GetNumCols();

		auto dOutputMap = dOutput.GetEigenMap();
		dOutputMap *= headScale;

		currentSize += oneByOne.GetReceptiveField();

		if (dHeadRechannelOut.GetNumCols() == 0)
			dHeadRechannelOut = trainingContext->GetBufferArena().template GetBuffer<Channels>(currentSize);

		dHeadRechannelOut.SetZero();
		oneByOne.Backward(headOutput, dOutput, dHeadRechannelOut);

		ChannelBufferDynamic<T, Channels> dLastLayerOut;
		ChannelBufferDynamic<T, Channels> dTmpLayerOut;

		ForEachIndex<NumLayers>([&](auto layerIndexForward)
			{
				constexpr auto layerIndexBackward = NumLayers - 1 - layerIndexForward;

				currentSize += std::get<layerIndexBackward>(layers).GetReceptiveField();
				auto dCurrentLayerOut = trainingContext->GetBufferArena().template GetScratchBuffer<Channels>(currentSize);

				if constexpr (layerIndexForward == 0)
				{
					std::get<layerIndexBackward>(layers).BackwardNoLayerOutput(layerOuts[layerIndexBackward - 1], input, dHeadRechannelOut,	dCurrentLayerOut);
				}
				else if constexpr (layerIndexBackward > 0)
				{
					std::get<layerIndexBackward>(layers).Backward(layerOuts[layerIndexBackward - 1], input,	dLastLayerOut, dHeadRechannelOut, dCurrentLayerOut);
				}
				else
				{
					std::get<layerIndexBackward>(layers).Backward(layerArrayRechannelOut, input, dLastLayerOut, dHeadRechannelOut, dCurrentLayerOut);
				}

				dTmpLayerOut = dLastLayerOut;
				dLastLayerOut = dCurrentLayerOut;

				if (dTmpLayerOut.GetNumCols() != 0)
				{
					trainingContext->GetBufferArena().FreeScratchBuffer(dTmpLayerOut);
				}
			});

		layerArrayRechannel.BackwardNoDInput(input, dLastLayerOut);	// skip dInput gradient calculation since it isn't used

		trainingContext->GetBufferArena().FreeScratchBuffer(dLastLayerOut);
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

	void RandomizeWeights() override
	{
		layerArrayRechannel.RandomizeWeights();

		ForEachIndex<NumLayers>([&](auto layerIndex)
			{
				std::get<layerIndex>(layers).RandomizeWeights();
			});

		oneByOne.RandomizeWeights();
	}

private:
	DenseBackpropT<T, InOutChannels, Channels, false> layerArrayRechannel;
	ChannelBufferDynamic<T, Channels> layerArrayRechannelOut;

	ChannelBufferDynamic<T, Channels> layerOuts[NumLayers];

	ChannelBufferDynamic<T, Channels> headOutput;
	Conv1DBackpropT<T, Channels, InOutChannels, 16, true, 1> oneByOne;
	ChannelBufferDynamic<T, Channels> dHeadRechannelOut;
	float headScale = 0.1f;
};

using A2KernelSizes = std::integer_sequence<int, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6>;
using A2Dilations = std::integer_sequence<int, 1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239>;

