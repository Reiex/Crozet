///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Pélégrin Marius
//! \copyright The MIT License (MIT)
//! \date 2022-2023
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Crozet/Core/CoreTypes.hpp>
#include <Crozet/Core/SoundBase.hpp>

namespace crz
{
	class CRZ_API AudioInput : public SoundBase
	{
		public:

			AudioInput();
			AudioInput(int deviceIndex);
			AudioInput(const AudioInput& input) = delete;
			AudioInput(AudioInput&& input) = delete;

			AudioInput& operator=(const AudioInput& input) = delete;
			AudioInput& operator=(AudioInput&& input) = delete;

			void setStoredLength(double storedLength);
			double getStoredLength() const;

			bool isValid() const;

			~AudioInput();

		private:

			int internalCallback(const int32_t* input, unsigned long frameCount);
			virtual void getRawSamples(int32_t* samples, uint64_t timeFrom, uint64_t timeTo) override final;

			static constexpr uint64_t _frameCount = 1024;

			void* _stream;

			uint64_t _storedSamples;

			std::mutex _samplesMutex;
			std::deque<int32_t> _samples;

		friend int audioInputMidCallback(const int32_t* input, unsigned long frameCount, AudioInput* audioInput);
	};
}
