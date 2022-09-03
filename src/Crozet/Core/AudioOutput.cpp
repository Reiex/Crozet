#include <Crozet/Core/Core.hpp>
#include <Crozet/Private/Private.hpp>

#include <iostream>

namespace crz
{
	namespace
	{
		int audioOutputCallback(const void* input, void* output, unsigned long frameCount, const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData)
		{
			return audioOutputMidCallback(reinterpret_cast<int32_t*>(output), frameCount, reinterpret_cast<AudioOutput*>(userData));
		}
	}

	int audioOutputMidCallback(int32_t* output, unsigned long frameCount, AudioOutput* audioOutput)
	{
		return audioOutput->internalCallback(output, frameCount);
	}


	AudioOutput::AudioOutput(int deviceIndex) :
		_stream(nullptr),
		_frequency(0),
		_channelCount(0),

		_nextSoundIndex(0),
		_sounds(),

		_timelineMutex(),
		_timelineIndex(0),
		_timeline(),

		_samplesThread(),
		_samplesMutex(),
		_samples(),
		_samplesReady(false),
		_stopThread(false)
	{
		// Initialize PortAudio (can be done multiple times, each time will require one more Pa_Terminate)

		PaError error = Pa_Initialize();
		if (error)
		{
			return;
		}

		// Create stream

		PaStream* paStream = reinterpret_cast<PaStream*>(_stream);

		const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(deviceIndex);
		_frequency = deviceInfo->defaultSampleRate;
		_channelCount = deviceInfo->maxOutputChannels;

		_samples.resize(_channelCount * _frameCount, 0);

		PaStreamParameters parameters;
		parameters.device = deviceIndex;
		parameters.channelCount = _channelCount;
		parameters.sampleFormat = paInt32;
		parameters.suggestedLatency = deviceInfo->defaultLowOutputLatency;
		parameters.hostApiSpecificStreamInfo = nullptr;

		error = Pa_OpenStream(&paStream, nullptr, &parameters, _frequency, _frameCount, paNoFlag, audioOutputCallback, this);
		if (error)
		{
			std::cout << Pa_GetErrorText(error) << std::endl;
			Pa_Terminate();
			_stream = nullptr;
			return;
		}

		// Start stream

		error = Pa_StartStream(paStream);
		if (error)
		{
			Pa_CloseStream(paStream);
			Pa_Terminate();
			_stream = nullptr;
			return;
		}

		// Start samples thread

		_samplesThread = std::thread(&AudioOutput::samplesComputationLoop, this);
	}

	void AudioOutput::playSound(uint64_t soundId, double delay)
	{
		assert(_sounds.find(soundId) != _sounds.end());

		_timelineMutex.lock();

		uint64_t index = _timelineIndex + uint64_t(delay * _frequency);
		auto it = _sounds.find(soundId);

		_timeline.emplace(index, it->second);
		_sounds.erase(it);

		_timelineMutex.unlock();
	}

	uint32_t AudioOutput::getFrequency() const
	{
		return _frequency;
	}

	uint16_t AudioOutput::getChannelCount() const
	{
		return _channelCount;
	}

	bool AudioOutput::isValid() const
	{
		return _stream;
	}

	AudioOutput::~AudioOutput()
	{
		if (_stream)
		{
			PaStream* paStream = reinterpret_cast<PaStream*>(_stream);

			Pa_StopStream(paStream);
			Pa_CloseStream(paStream);
			Pa_Terminate();

			_stopThread = true;
			_samplesThread.join();

			for (std::pair<const uint64_t, SoundBase*>& elt : _sounds)
			{
				delete elt.second;
			}

			for (std::pair<const uint64_t, SoundBase*>& elt : _timeline)
			{
				delete elt.second;
			}
		}
	}

	int AudioOutput::internalCallback(int32_t* output, unsigned long frameCount)
	{
		assert(frameCount * _channelCount == _samples.size());

		_samplesMutex.lock();

		if (_samplesReady)
		{
			std::copy(_samples.begin(), _samples.end(), output);
			_samplesReady = false;
		}
		else
		{
			std::fill_n(output, _samples.size(), 0);
		}

		_samplesMutex.unlock();

		return 0;
	}

	void AudioOutput::samplesComputationLoop()
	{
		while (!_stopThread)
		{
			if (!_samplesReady)
			{
				_timelineMutex.lock();
				_samplesMutex.lock();

				std::fill(_samples.begin(), _samples.end(), 0);
				std::vector<int32_t> buffer(_samples.size());

				const uint64_t range[2] = { _timelineIndex, _timelineIndex + _frameCount };

				auto itTimeline = _timeline.begin();
				const auto itTimelineEnd = _timeline.cend();
				for (; itTimeline != itTimelineEnd; ++itTimeline)
				{
					const uint64_t& timeStart = itTimeline->first;
					SoundBase* sound = itTimeline->second;
					const uint64_t sampleCount = sound->getSampleCount(_frequency);

					if (timeStart >= range[1])
					{
						break;
					}

					std::fill(buffer.begin(), buffer.end(), 0);
					
					const uint64_t offset = timeStart > range[0] ? timeStart - range[0] : 0;
					const uint64_t timeFrom = range[0] > timeStart ? range[0] - timeStart : 0;
					const uint64_t timeTo = std::min(timeFrom + _frameCount, sampleCount);

					sound->getSamples(_frequency, _channelCount, buffer.data() + offset, timeFrom, timeTo);

					auto itSamples = _samples.begin();
					auto itBuffer = buffer.begin();
					const auto itBufferEnd = buffer.cend();
					for (; itBuffer != itBufferEnd; ++itSamples, ++itBuffer)
					{
						*itSamples = static_cast<int32_t>(std::clamp<int64_t>(static_cast<int64_t>(*itSamples) + static_cast<int64_t>(*itBuffer), INT32_MIN, INT32_MAX));
					}

					if (timeTo == sampleCount)
					{
						auto itErased = itTimeline;
						--itTimeline;
						delete sound;
						_timeline.erase(itErased);
					}
				}

				_timelineIndex += _frameCount;
				_samplesReady = true;

				_samplesMutex.unlock();
				_timelineMutex.unlock();
			}
		}
	}
}
