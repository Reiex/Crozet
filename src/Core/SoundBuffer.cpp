///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Pélérin Marius
//! \copyright The MIT License (MIT)
//! \date 2022-2023
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <Crozet/Core/Core.hpp>
#include <Crozet/Private/Private.hpp>

namespace crz
{
	SoundBuffer::SoundBuffer(uint32_t frequency, uint16_t channelCount, uint64_t sampleCount, const int32_t* samples) : SoundBase(),
		_samples(samples)
	{
		assert(frequency != 0);
		assert(channelCount != 0);

		_frequency = frequency;
		_channelCount = channelCount;
		_sampleCount = sampleCount;
	}

	void SoundBuffer::getRawSamples(int32_t* samples, uint64_t timeFrom, uint64_t timeTo)
	{
		std::copy_n(_samples + timeFrom * _channelCount, (timeTo - timeFrom) * _channelCount, samples);
		_currentSample = timeTo;
	}
}
