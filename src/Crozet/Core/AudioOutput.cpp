///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2022-2023
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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


	AudioOutput::AudioOutput() : AudioOutput(AudioDevice::getDefaultOutputDeviceIndex())
	{
	}

	AudioOutput::AudioOutput(int deviceIndex) :
		_stream(nullptr),

		_frequency(0),
		_channelCount(0),

		_nextSoundId(0),
		_sounds(),

		_scheduleMutex(),
		_currentTime(0),
		_schedule(),

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
		assert(_channelCount > 0);

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
			Pa_Terminate();
			return;
		}

		// Start stream

		error = Pa_StartStream(paStream);
		if (error)
		{
			Pa_CloseStream(paStream);
			Pa_Terminate();
			return;
		}

		// Start samples thread

		_stream = reinterpret_cast<void*>(paStream);
		_samplesThread = std::thread(&AudioOutput::samplesComputationLoop, this);
	}

	void AudioOutput::scheduleSound(uint64_t soundId, double delay, double startTime, double duration, bool removeWhenFinished)
	{
		assert(isValid());

		_scheduleMutex.lock();

		assert(canScheduleSound(soundId, delay, startTime, duration, removeWhenFinished));

		// Compute schedule info

		ScheduleInfo info;
		info.scheduleTime = _currentTime + uint64_t(delay * _frequency);
		info.timeFrom = startTime * _frequency;
		info.timeTo = duration < 0.0 ? UINT64_MAX : (startTime + duration) * _frequency;
		info.removeWhenFinished = removeWhenFinished;

		// Start stream if it was stopped

		if (_schedule.empty())
		{
			PaStream* paStream = reinterpret_cast<PaStream*>(_stream);
			Pa_StartStream(paStream);
		}

		// Insert info into _schedule

		std::deque<ScheduleInfo>& infos = _schedule[soundId];
		auto it = infos.begin();
		const auto itEnd = infos.cend();
		for (;; ++it)
		{
			if (it == itEnd || info.scheduleTime < it->scheduleTime)
			{
				infos.insert(it, info);
				break;
			}
		}

		_scheduleMutex.unlock();
	}

	void AudioOutput::unscheduleSound(uint64_t soundId)
	{
		assert(isValid());
		assert(_sounds.find(soundId) != _sounds.end());

		_scheduleMutex.lock();

		auto it = _schedule.find(soundId);
		if (it != _schedule.end())
		{
			_schedule.erase(it);
		}

		_scheduleMutex.unlock();
	}

	void AudioOutput::removeSound(uint64_t soundId)
	{
		assert(isValid());
		assert(_sounds.find(soundId) != _sounds.end());

		unscheduleSound(soundId);

		auto it = _sounds.find(soundId);
		delete it->second;
		_sounds.erase(it);
	}

	const SoundBase* AudioOutput::getSound(uint64_t soundId) const
	{
		assert(isValid());

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
		assert(isValid());

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
		assert(isValid());

		return _frequency;
	}

	uint16_t AudioOutput::getChannelCount() const
	{
		assert(isValid());

		return _channelCount;
	}

	bool AudioOutput::isValid() const
	{
		return _stream;
	}

	AudioOutput::~AudioOutput()
	{
		if (isValid())
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

	bool AudioOutput::canScheduleSound(uint64_t soundId, double delay, double startTime, double duration, bool removeWhenFinished) const
	{
		// Check sound exists

		auto itSounds = _sounds.find(soundId);
		if (itSounds == _sounds.end())
		{
			return false;
		}

		// Check sound can be played starting at desired time

		const SoundSource* source = itSounds->second->getFilteredSource();
		if (startTime < source->getCurrentTime())
		{
			return false;
		}

		// If sound isn't already scheduled, shortcut the call

		auto itSchedule = _schedule.find(soundId);
		if (itSchedule == _schedule.end())
		{
			return true;
		}

		const std::deque<ScheduleInfo>& infos = itSchedule->second;

		// Check the same sound isn't playing twice at the same time and that there is no "rewind"

		const uint64_t timeFrom = startTime * _frequency;
		const uint64_t timeTo = duration < 0.0 ? UINT64_MAX : (startTime + duration) * _frequency;
		const uint64_t scheduleTime = _currentTime + uint64_t(delay * _frequency);

		auto itInfo = infos.begin();
		const auto itInfoEnd = infos.cend();
		for (; itInfo != itInfoEnd; ++itInfo)
		{
			if (scheduleTime < itInfo->scheduleTime)
			{
				if (removeWhenFinished)
				{
					return false;
				}

				if (scheduleTime + timeTo - timeFrom > itInfo->scheduleTime || timeTo > itInfo->timeFrom)
				{
					return false;
				}

				break;
			}
			else
			{
				if (itInfo->removeWhenFinished)
				{
					return false;
				}
			}
		}

		if (itInfo != infos.begin())
		{
			--itInfo;
			if (itInfo->scheduleTime + itInfo->timeTo - itInfo->timeFrom > scheduleTime || itInfo->timeTo > timeFrom)
			{
				return false;
			}
			++itInfo;
		}

		return true;
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

		return paContinue;
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

			_scheduleMutex.lock();

			std::fill(_samples.begin(), _samples.end(), 0);
			std::vector<int32_t> buffer(_samples.size());

			// For each sound currently playing in the schedule

			auto itSchedule = _schedule.begin();
			const auto itScheduleEnd = _schedule.cend();
			for (; itSchedule != itScheduleEnd;)
			{
				ScheduleInfo& info = itSchedule->second.front();

				if (info.scheduleTime > _currentTime + _frameCount)
				{
					++itSchedule;
					continue;
				}

				// Compute samples to retrieve and retrieve them

				SoundSource* source = _sounds.find(itSchedule->first)->second->getFilteredSource();
				const uint64_t sampleCount = source->getSampleCount() * _frequency / source->getFrequency();
				const uint64_t timeFrom = info.timeFrom + (_currentTime > info.scheduleTime ? _currentTime - info.scheduleTime : 0);
				const uint64_t offset = info.scheduleTime > _currentTime ? info.scheduleTime - _currentTime : 0;
				const uint64_t timeTo = std::min(timeFrom + _frameCount - offset, info.timeTo);

				std::fill_n(buffer.begin(), offset, 0);
				source->getSamples(_frequency, _channelCount, buffer.data() + offset, timeFrom, timeTo);

				// Stack them to the output samples

				std::transform(_samples.begin(), _samples.end(), buffer.begin(), _samples.begin(), samplesStackFunc);

				// Remove the sound if the end was reached

				if (timeTo == info.timeTo || timeTo >= sampleCount)
				{
					if (info.removeWhenFinished)
					{
						delete _sounds.find(itSchedule->first)->second;
						_sounds.erase(itSchedule->first);
					}

					itSchedule->second.pop_front();

					if (itSchedule->second.empty())
					{
						auto itErased = itSchedule;
						++itSchedule;

						_schedule.erase(itErased);
					}
					else
					{
						++itSchedule;
					}
				}
				else
				{
					++itSchedule;
				}
			}

			// Mark the samples as ready

			_currentTime += _frameCount;
			_samplesReady = true;

			// Stop the stream if timeline is empty

			if (_schedule.empty())
			{
				Pa_StopStream(paStream);
			}

			_scheduleMutex.unlock();
		}
	}
}
