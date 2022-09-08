#pragma once

#include <Crozet/Core/types.hpp>
#include <Crozet/Core/SoundSource.hpp>

namespace crz
{
	class SoundBase : public SoundSource
	{
		public:

			SoundBase(const SoundBase& sound) = delete;
			SoundBase(SoundBase&& sound) = delete;

			SoundBase& operator=(const SoundBase& sound) = delete;
			SoundBase& operator=(SoundBase&& sound) = delete;

			template<std::derived_from<FilterBase> TFilter, typename... Args> uint64_t addFilter(Args&&... args);

			const FilterBase* getFilter(uint64_t filterId) const;
			FilterBase* getFilter(uint64_t filterId);

			const SoundSource* getFilteredSource() const;
			SoundSource* getFilteredSource();

			virtual uint32_t getFrequency() const override final;
			virtual uint16_t getChannelCount() const override final;
			virtual uint64_t getSampleCount() const override final;
			virtual uint64_t getCurrentSample() const override final;

			virtual ~SoundBase();

		protected:

			SoundBase();

			virtual void getRawSamples(int32_t* samples, uint64_t timeFrom, uint64_t timeTo) override = 0;

			uint32_t _frequency;
			uint16_t _channelCount;
			uint64_t _sampleCount;
			uint64_t _currentSample;

			std::vector<FilterBase*> _filters;
	};
}

#include <Crozet/Core/templates/SoundBase.hpp>
