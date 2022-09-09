#pragma once

#include <Crozet/Core/types.hpp>

namespace crz
{
	class SoundBuffer : public SoundBase
	{
		public:

			SoundBuffer(uint32_t frequency, uint16_t channelCount, uint64_t sampleCount, const int32_t* samples);
			SoundBuffer(const SoundBuffer& sound) = delete;
			SoundBuffer(SoundBuffer&& sound) = delete;

			SoundBuffer& operator=(const SoundBuffer& sound) = delete;
			SoundBuffer& operator=(SoundBuffer&& sound) = delete;

			~SoundBuffer() = default;

		private:

			void getRawSamples(int32_t* samples, uint64_t timeFrom, uint64_t timeTo) override final;

			const int32_t* _samples;
	};
}
