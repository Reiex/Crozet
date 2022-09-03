#pragma once

#include <Crozet/Core/types.hpp>

namespace crz
{
	class SoundBase
	{
		public:

			SoundBase(const SoundBase& sound) = delete;
			SoundBase(SoundBase&& sound) = delete;

			SoundBase& operator=(const SoundBase& sound) = delete;
			SoundBase& operator=(SoundBase&& sound) = delete;

			uint32_t getFrequency() const;
			uint16_t getChannelCount() const;
			uint64_t getSampleCount() const;

			virtual ~SoundBase() = default;

		protected:

			SoundBase();

			virtual void getRawSamples(int32_t* samples, uint64_t timeFrom, uint64_t timeTo) = 0;

			uint32_t _frequency;
			uint16_t _channelCount;
			uint64_t _sampleCount;

		private:

			uint64_t getSampleCount(uint32_t frequency) const;
			void getSamples(uint32_t frequency, uint16_t channelCount, int32_t* samples, uint64_t timeFrom, uint64_t timeTo);
	
		friend class AudioOutput;
	};
}
