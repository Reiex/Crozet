#pragma once

#include <Crozet/Core/types.hpp>

namespace crz
{
	class AudioOutput
	{
		public:

			AudioOutput(int deviceIndex);
			AudioOutput(const AudioOutput& stream) = delete;
			AudioOutput(AudioOutput&& stream) = delete;

			AudioOutput& operator=(const AudioOutput& stream) = delete;
			AudioOutput& operator=(AudioOutput&& stream) = delete;

			template<std::derived_from<SoundBase> TSound, typename... Args> uint64_t createSound(Args&&... args);
			// TSound getSound(uint64_t soundId);
			void playSound(uint64_t soundId, double delay = 0.0);

			uint32_t getFrequency() const;
			uint16_t getChannelCount() const;
			bool isValid() const;

			~AudioOutput();

		private:

			int internalCallback(int32_t* output, unsigned long frameCount);
			void samplesComputationLoop();

			static constexpr uint64_t _frameCount = 512;

			void* _stream;

			uint32_t _frequency;
			uint16_t _channelCount;

			uint64_t _nextSoundIndex;
			std::unordered_map<uint64_t, SoundBase*> _sounds;

			std::mutex _timelineMutex;
			uint64_t _timelineIndex;
			std::multimap<uint64_t, SoundBase*> _timeline;

			std::thread _samplesThread;
			std::mutex _samplesMutex;
			std::vector<int32_t> _samples;
			bool _samplesReady;
			bool _stopThread;

		friend int audioOutputMidCallback(int32_t* output, unsigned long frameCount, AudioOutput* outputStream);
	};
}

#include <Crozet/Core/templates/AudioOutput.hpp>
