///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2022-2023
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <Crozet/Core/types.hpp>

namespace crz
{
	class AudioOutput
	{
		public:

			AudioOutput();
			AudioOutput(int deviceIndex);
			AudioOutput(const AudioOutput& output) = delete;
			AudioOutput(AudioOutput&& output) = delete;

			AudioOutput& operator=(const AudioOutput& output) = delete;
			AudioOutput& operator=(AudioOutput&& output) = delete;

			template<std::derived_from<SoundBase> TSound, typename... Args> uint64_t createSound(Args&&... args);
			void scheduleSound(uint64_t soundId, double delay = 0.0, double startTime = 0.0, double duration = -1.0, bool removeWhenFinished = true);
			void unscheduleSound(uint64_t soundId);
			void removeSound(uint64_t soundId);

			const SoundBase* getSound(uint64_t soundId) const;
			SoundBase* getSound(uint64_t soundId);

			uint32_t getFrequency() const;
			uint16_t getChannelCount() const;
			bool isValid() const;

			~AudioOutput();

		private:

			bool canScheduleSound(uint64_t soundId, double delay, double startTime, double duration, bool removeWhenFinished) const;
			int internalCallback(int32_t* output, unsigned long frameCount);
			void samplesComputationLoop();

			struct ScheduleInfo
			{
				uint64_t scheduleTime;
				uint64_t timeFrom;
				uint64_t timeTo;
				bool removeWhenFinished;
			};

			static constexpr uint64_t _frameCount = 1024;

			void* _stream;

			uint32_t _frequency;
			uint16_t _channelCount;

			uint64_t _nextSoundId;
			std::unordered_map<uint64_t, SoundBase*> _sounds;

			std::mutex _scheduleMutex;
			uint64_t _currentTime;
			std::unordered_map<uint64_t, std::deque<ScheduleInfo>> _schedule;

			std::thread _samplesThread;
			std::mutex _samplesMutex;
			std::condition_variable _samplesCondition;
			std::vector<int32_t> _samples;
			bool _samplesReady;

		friend int audioOutputMidCallback(int32_t* output, unsigned long frameCount, AudioOutput* audioOutput);
	};
}

#include <Crozet/Core/templates/AudioOutput.hpp>
