///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Pélégrin Marius
//! \copyright The MIT License (MIT)
//! \date 2022-2023
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Crozet/Core/CoreTypes.hpp>

namespace crz
{
	class CRZ_API SoundSource
	{
		public:

			SoundSource(const SoundSource& source) = delete;
			SoundSource(SoundSource&& source) = delete;

			SoundSource& operator=(const SoundSource& source) = delete;
			SoundSource& operator=(SoundSource&& source) = delete;

			virtual uint32_t getFrequency() const = 0;
			virtual uint16_t getChannelCount() const = 0;
			virtual uint64_t getSampleCount() const = 0;
			virtual uint64_t getCurrentSample() const = 0;

			double getCurrentTime() const;

			void getSamples(uint32_t frequency, uint16_t channelCount, int32_t* samples, uint64_t timeFrom, uint64_t timeTo);

			virtual ~SoundSource() = default;

		protected:

			SoundSource() = default;

			virtual void getRawSamples(int32_t* samples, uint64_t timeFrom, uint64_t timeTo) = 0;
	};
}
