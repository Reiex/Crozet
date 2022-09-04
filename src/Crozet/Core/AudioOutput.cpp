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
		_samplesCondition(),
		_samples(),
		_samplesReady(false)
	{
		// Initialize PortAudio (can be done multiple times, each time will require one more Pa_Terminate)

		PaError error = Pa_Initialize();
		if (error)
		{
			return;
		}

		// Retrieve and store device infos

		const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(deviceIndex);
		_frequency = deviceInfo->defaultSampleRate;
		_channelCount = deviceInfo->maxOutputChannels;

		_samples.resize(_channelCount * _frameCount, 0);

		// Open stream from device infos

		PaStreamParameters parameters;
		parameters.device = deviceIndex;
		parameters.channelCount = _channelCount;
		parameters.sampleFormat = paInt32;
		parameters.suggestedLatency = deviceInfo->defaultLowOutputLatency;
		parameters.hostApiSpecificStreamInfo = nullptr;

		PaStream* paStream = reinterpret_cast<PaStream*>(_stream);
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

	void AudioOutput::playSound(uint64_t soundId, double delay, double startTime, double duration, bool removeWhenFinished)
	{
		assert(_sounds.find(soundId) != _sounds.end());
		assert(!_sounds.find(soundId)->second->isPlaying());
		assert(startTime >= _sounds.find(soundId)->second->getCurrentTime());

		_timelineMutex.lock();

		// Compute where to insert sound in _timeline

		auto it = _sounds.find(soundId);
		SoundBase* sound = it->second;
		const uint64_t index = _timelineIndex + uint64_t(delay * _frequency);

		// Compute sound play info

		const uint64_t sampleCount = sound->getSampleCount(_frequency);
		SoundPlayInfo info;
		info.soundId = soundId;
		info.timeFrom = std::min<uint64_t>(startTime * _frequency, sampleCount);
		if (duration < 0.0)
		{
			info.timeTo = sampleCount;
		}
		else
		{
			info.timeTo = std::min<uint64_t>((startTime + duration) * _frequency, sampleCount);
		}
		info.removeWhenFinished = removeWhenFinished;

		// If timeline is empty, we need to restart the stream (stopped in samplesComputationLoop)

		if (_timeline.empty())
		{
			PaStream* paStream = reinterpret_cast<PaStream*>(_stream);
			Pa_StartStream(paStream);
		}

		// Add sound to timeline

		_timeline.emplace(index, info);

		_timelineMutex.unlock();
	}

	void AudioOutput::stopSound(uint64_t soundId)
	{
		assert(_sounds.find(soundId) != _sounds.end());

		_timelineMutex.lock();

		// Find sound in timeline

		auto it = _timeline.begin();
		const auto itEnd = _timeline.cend();
		for (; it->second.soundId != soundId && it != itEnd; ++it);

		// Remove it from timeline

		if (it != itEnd)
		{
			_timeline.erase(it);
			_sounds.find(soundId)->second->setIsPlaying(false);
		}

		_timelineMutex.unlock();
	}

	void AudioOutput::removeSound(uint64_t soundId)
	{
		assert(_sounds.find(soundId) != _sounds.end());

		// If sound is playing, stop it

		stopSound(soundId);

		// Delete and remove sound

		auto it = _sounds.find(soundId);
		delete it->second;
		_sounds.erase(it);
	}

	const SoundBase* AudioOutput::getSound(uint64_t soundId) const
	{
		auto it = _sounds.find(soundId);
		if (it == _sounds.end())
		{
			return nullptr;
		}
		else
		{
			return it->second;
		}
	}

	SoundBase* AudioOutput::getSound(uint64_t soundId)
	{
		auto it = _sounds.find(soundId);
		if (it == _sounds.end())
		{
			return nullptr;
		}
		else
		{
			return it->second;
		}
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

			Pa_AbortStream(paStream);
			Pa_CloseStream(paStream);
			Pa_Terminate();

			for (std::pair<const uint64_t, SoundBase*>& elt : _sounds)
			{
				delete elt.second;
			}
		}
	}

	int AudioOutput::internalCallback(int32_t* output, unsigned long frameCount)
	{
		assert(frameCount * _channelCount == _samples.size());

		std::unique_lock lock(_samplesMutex);

		if (_samplesReady)
		{
			std::copy(_samples.begin(), _samples.end(), output);
			_samplesReady = false;
		}
		else
		{
			std::fill_n(output, _samples.size(), 0);
		}

		_samplesCondition.notify_one();

		return 0;
	}

	namespace
	{
		int32_t samplesStackFunc(int32_t x, int32_t y)
		{
			return std::clamp<int64_t>(static_cast<int64_t>(x) + static_cast<int64_t>(y), INT32_MIN, INT32_MAX);
		}
	}

	void AudioOutput::samplesComputationLoop()
	{
		// This function runs while *this exists

		PaStream* paStream = reinterpret_cast<PaStream*>(_stream);

		while (true)
		{
			// Wait for samples to be emptied by the audio callback

			std::unique_lock lock(_samplesMutex);
			_samplesCondition.wait(lock, [&] { return !_samplesReady; });

			// Prepare samples range to be computed

			_timelineMutex.lock();

			const uint64_t range[2] = { _timelineIndex, _timelineIndex + _frameCount };

			std::fill(_samples.begin(), _samples.end(), 0);
			std::vector<int32_t> buffer(_samples.size());

			// For each sound currently playing in the timeline

			auto itTimeline = _timeline.begin();
			const auto itTimelineEnd = _timeline.cend();
			for (; itTimeline != itTimelineEnd;)
			{
				const uint64_t& index = itTimeline->first;

				if (index >= range[1])
				{
					break;
				}

				// Compute samples to retrieve and retrieve them

				const SoundPlayInfo& info = itTimeline->second;
				SoundBase* sound = _sounds.find(info.soundId)->second;
				const uint64_t sampleCount = sound->getSampleCount(_frequency);
					
				const uint64_t timeFrom = info.timeFrom + (range[0] > index ? range[0] - index : 0);
				const uint64_t offset = index > range[0] ? index - range[0] : 0;
				const uint64_t timeTo = std::min(timeFrom + _frameCount - offset, info.timeTo);

				if (timeTo - timeFrom != _frameCount)
				{
					std::fill(buffer.begin(), buffer.end(), 0);
				}
				sound->getSamples(_frequency, _channelCount, buffer.data() + offset, timeFrom, timeTo);

				// Stack them to the output samples

				std::transform(_samples.begin(), _samples.end(), buffer.begin(), _samples.begin(), samplesStackFunc);

				// Remove the sound if the end was reached or set isPlaying if necessary

				if (timeTo == info.timeTo)
				{
					if (info.removeWhenFinished)
					{
						delete sound;
						_sounds.erase(info.soundId);
					}
					else
					{
						sound->setIsPlaying(false);
					}

					auto itErased = itTimeline;
					++itTimeline;
					_timeline.erase(itErased);
				}
				else
				{
					if (!sound->isPlaying())
					{
						sound->setIsPlaying(true);
					}

					++itTimeline;
				}
			}

			// Mark the samples as ready

			_timelineIndex += _frameCount;
			_samplesReady = true;

			// Stop the stream if timeline is empty

			if (_timeline.empty())
			{
				Pa_StopStream(paStream);
			}

			_timelineMutex.unlock();
		}
	}
}
