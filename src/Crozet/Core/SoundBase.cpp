#include <Crozet/Core/Core.hpp>
#include <Crozet/Private/Private.hpp>

namespace crz
{
	SoundBase::SoundBase() :
		_frequency(0),
		_channelCount(0),
		_sampleCount(0),
		_currentSample(0)
	{
	}

	const FilterBase* SoundBase::getFilter(uint64_t filterId) const
	{
		if (filterId < _filters.size())
		{
			return _filters[filterId];
		}
		else
		{
			return nullptr;
		}
	}

	FilterBase* SoundBase::getFilter(uint64_t filterId)
	{
		if (filterId < _filters.size())
		{
			return _filters[filterId];
		}
		else
		{
			return nullptr;
		}
	}

	const SoundSource* SoundBase::getFilteredSource() const
	{
		if (_filters.empty())
		{
			return this;
		}
		else
		{
			return _filters.back();
		}
	}

	SoundSource* SoundBase::getFilteredSource()
	{
		if (_filters.empty())
		{
			return this;
		}
		else
		{
			return _filters.back();
		}
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

	SoundBase::~SoundBase()
	{
		for (FilterBase* filter : _filters)
		{
			delete filter;
		}
	}
}
