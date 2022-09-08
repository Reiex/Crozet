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

		// Retrieve sound-base samples

		scp::Vector<int32_t> sampleVector(timeTo - timeFrom);
		scp::Vector<int32_t> rawSampleVector(realTimeTo - realTimeFrom);

		std::vector<int32_t> buffer((realTimeTo - realTimeFrom) * realChannelCount);
		getRawSamples(buffer.data(), realTimeFrom, realTimeTo);

		// Convert them to output-frequency samples

		const uint16_t commonChannelCount = std::min(realChannelCount, channelCount);
		for (uint16_t i = 0; i < commonChannelCount; ++i)
		{
			int32_t* it = buffer.data() + i;
			for (uint64_t j = 0; j < rawSampleVector.getSize(0); ++j, it += realChannelCount)
			{
				rawSampleVector[j] = *it;
			}

			rawSampleVector.interpolation(sampleVector, scp::InterpolationMethod::Cubic);

			it = samples + i;
			for (uint64_t j = 0; j < sampleVector.getSize(0); ++j, it += channelCount)
			{
				*it = sampleVector[j];
			}
		}

		// Then to output-channelCount samples

		for (uint16_t i = realChannelCount; i < channelCount; ++i)
		{
			int32_t* it = samples + i;
			for (uint64_t j = 0; j < sampleVector.getSize(0); ++j, it += channelCount)
			{
				*it = *(it - realChannelCount);
			}
		}
	}
}
