///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2022-2023
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <Crozet/Core/Core.hpp>
#include <Crozet/Private/Private.hpp>

namespace crz
{
	double SoundSource::getCurrentTime() const
	{
		return static_cast<double>(getCurrentSample()) / getFrequency();
	}

	void SoundSource::getSamples(uint32_t frequency, uint16_t channelCount, int32_t* samples, uint64_t timeFrom, uint64_t timeTo)
	{
		// Clip timeFrom and timeTo

		if (timeFrom > timeTo)
		{
			std::swap(timeFrom, timeTo);
		}

		const uint64_t sampleCount = getSampleCount();
		if (timeFrom >= sampleCount)
		{
			std::fill_n(samples, timeTo - timeFrom, 0);
			return;
		}
		else if (timeTo > sampleCount)
		{
			std::fill_n(samples + sampleCount - timeFrom, timeTo - sampleCount, 0);
			timeTo = sampleCount;
		}

		// If no sample required, shortcut the call

		if (timeFrom == timeTo)
		{
			return;
		}

		// If frequency and channel count are the same, shortcut the call as well

		const uint32_t realFrequency = getFrequency();
		const uint16_t realChannelCount = getChannelCount();

		if (frequency == realFrequency && channelCount == realChannelCount)
		{
			getRawSamples(samples, timeFrom, timeTo);
			return;
		}

		// Compute real sample indices (based on sound frequency instead of output frequency)

		const uint64_t realTimeFrom = timeFrom * realFrequency / frequency;
		uint64_t realTimeTo = timeTo * realFrequency / frequency;

		if (realTimeFrom == realTimeTo)
		{
			++realTimeTo;
		}

		// TODO: Comment this

		int32_t* rawSamples = new int32_t[(realTimeTo - realTimeFrom) * std::max(channelCount, realChannelCount)];
		getRawSamples(rawSamples, realTimeFrom, realTimeTo);

		if (channelCount > realChannelCount)
		{
			const uint16_t srcDecrement = 2 * realChannelCount;
			const uint16_t dstDecrement = 2 * channelCount;


			int32_t* itDst = rawSamples + (realTimeTo - realTimeFrom - 1) * channelCount;
			const int32_t* itSrc = rawSamples + (realTimeTo - realTimeFrom - 1) * realChannelCount;
			const int32_t* const itEnd = rawSamples - realChannelCount;

			for (; itSrc != itEnd; itSrc -= srcDecrement, itDst -= dstDecrement)
			{
				uint16_t i = 0;
				for (; i < realChannelCount; ++i, ++itSrc, ++itDst)
				{
					*itDst = *itSrc;
				}

				for (; i < channelCount; ++i, ++itDst)
				{
					*itDst = *(itDst - realChannelCount);
				}
			}
		}
		else if (realChannelCount > channelCount)
		{
			const uint16_t srcIncrement = realChannelCount - channelCount;

			int32_t* itDst = rawSamples + channelCount;
			const int32_t* itSrc = rawSamples + realChannelCount;
			const int32_t* const itEnd = rawSamples + (realTimeTo - realTimeFrom) * realChannelCount;

			for (; itSrc != itEnd; itSrc += srcIncrement)
			{
				for (uint16_t i = 0; i < channelCount; ++i, ++itSrc, ++itDst)
				{
					*itDst = *itSrc;
				}
			}
		}

		scp::Matrix<int32_t>* samplesMatrix = scp::Matrix<int32_t>::createAroundMemory(timeTo - timeFrom, channelCount, samples);
		scp::Matrix<int32_t>* rawSamplesMatrix = scp::Matrix<int32_t>::createAroundMemory(realTimeTo - realTimeFrom, channelCount, rawSamples);

		samplesMatrix->resize<float, scp::InterpolationMethod::Linear>(*rawSamplesMatrix);

		delete samplesMatrix;
		delete rawSamplesMatrix;
		delete[] rawSamples;
	}
}
