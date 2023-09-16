///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Pélégrin Marius
//! \copyright The MIT License (MIT)
//! \date 2022-2023
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <Crozet/Core/Core.hpp>
#include <Crozet/Private/Private.hpp>

namespace crz
{
	FilterPlaySpeed::FilterPlaySpeed(double speedRatio) : FilterBase(),
		_speedRatio(speedRatio)
	{
		assert(speedRatio > 0.0);	// TODO: Reverse play ?
	}

	uint32_t FilterPlaySpeed::getFrequency() const
	{
		return _source->getFrequency() * _speedRatio;
	}

	uint16_t FilterPlaySpeed::getChannelCount() const
	{
		return _source->getChannelCount();
	}

	uint64_t FilterPlaySpeed::getSampleCount() const
	{
		return _source->getSampleCount();
	}

	uint64_t FilterPlaySpeed::getCurrentSample() const
	{
		return _source->getCurrentSample();
	}

	void FilterPlaySpeed::getRawSamples(int32_t* samples, uint64_t timeFrom, uint64_t timeTo)
	{
		_source->getSamples(_source->getFrequency(), _source->getChannelCount(), samples, timeFrom, timeTo);
	}
}
