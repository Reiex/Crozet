#include <Crozet/Core/Core.hpp>
#include <Crozet/Private/Private.hpp>

namespace crz
{
	SoundBase::SoundBase() :
		_frequency(0),
		_channelCount(0),
		_sampleCount(0),
		_currentSample(0),
		_isPlaying(false)
	{
	}

	uint32_t SoundBase::getFrequency() const
	{
		return _frequency;
	}

	uint16_t SoundBase::getChannelCount() const
	{
		return _channelCount;
	}

	uint64_t SoundBase::getSampleCount() const
	{
		return _sampleCount;
	}

	uint64_t SoundBase::getCurrentSample() const
	{
		return _currentSample;
	}

	double SoundBase::getCurrentTime() const
	{
		return static_cast<double>(_currentSample) / _frequency;
	}

	bool SoundBase::isPlaying() const
	{
		return _isPlaying;
	}

	void SoundBase::setIsPlaying(bool isPlaying)
	{
		_isPlaying = isPlaying;
	}

	uint64_t SoundBase::getSampleCount(uint32_t frequency) const
	{
		return _sampleCount * frequency / _frequency;
	}

	void SoundBase::getSamples(uint32_t frequency, uint16_t channelCount, int32_t* samples, uint64_t timeFrom, uint64_t timeTo)
	{
		// If no samples required, shortcut the call

		if (timeFrom == timeTo)
		{
			return;
		}

		// Compute real sample indices (based on sound frequency instead of output frequency)

		const uint64_t realTimeFrom = timeFrom * _frequency / frequency;
		uint64_t realTimeTo = timeTo * _frequency / frequency;

		if (realTimeFrom == realTimeTo)
		{
			++realTimeTo;
		}

		// Retrieve sound-base samples

		scp::Vector<int32_t> sampleVector(timeTo - timeFrom);
		scp::Vector<int32_t> rawSampleVector(realTimeTo - realTimeFrom);

		std::vector<int32_t> buffer((realTimeTo - realTimeFrom) * _channelCount);
		getRawSamples(buffer.data(), realTimeFrom, realTimeTo);
		_currentSample = realTimeTo;

		// Convert them to output-frequency samples

		const uint16_t commonChannelCount = std::min(_channelCount, channelCount);
		for (uint16_t i = 0; i < commonChannelCount; ++i)
		{
			int32_t* it = buffer.data() + i;
			for (uint64_t j = 0; j < rawSampleVector.getSize(0); ++j, it += _channelCount)
			{
				rawSampleVector[j] = *it;
			}

			rawSampleVector.interpolation(sampleVector, scp::InterpolationMethod::Cubic);

			it = samples + i;
			for (uint64_t j = 0; j < sampleVector.getSize(0); ++j, it += channelCount)
			{
				*it = rawSampleVector[j];
			}
		}

		// Then to output-channelCount samples

		for (uint16_t i = _channelCount; i < channelCount; ++i)
		{
			int32_t* it = samples + i;
			for (uint64_t j = 0; j < sampleVector.getSize(0); ++j, it += channelCount)
			{
				*it = *(it - _channelCount);
			}
		}
	}
}
