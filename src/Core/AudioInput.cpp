///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Pélérin Marius
//! \copyright The MIT License (MIT)
//! \date 2022-2023
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <Crozet/Core/Core.hpp>
#include <Crozet/Private/Private.hpp>

namespace crz
{
	namespace
	{
		int audioInputCallback(const void* input, void* output, unsigned long frameCount, const PaStreamCallbackTimeInfo* timeInfo, PaStreamCallbackFlags statusFlags, void* userData)
		{
			return audioInputMidCallback(reinterpret_cast<const int32_t*>(input), frameCount, reinterpret_cast<AudioInput*>(userData));
		}
	}

	int audioInputMidCallback(const int32_t* input, unsigned long frameCount, AudioInput* audioInput)
	{
		return audioInput->internalCallback(input, frameCount);
	}


	AudioInput::AudioInput() : AudioInput(AudioDevice::getDefaultInputDeviceIndex())
	{
	}

	AudioInput::AudioInput(int deviceIndex) : SoundBase(),
		_stream(nullptr),
		_storedSamples(0),
		_samplesMutex(),
		_samples()
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
		_channelCount = deviceInfo->maxInputChannels;
		_sampleCount = 0;
		_currentSample = 0;
		_storedSamples = _frequency * _channelCount;
		assert(_channelCount > 0);

		// Open stream from device infos

		PaStreamParameters parameters;
		parameters.device = deviceIndex;
		parameters.channelCount = _channelCount;
		parameters.sampleFormat = paInt32;
		parameters.suggestedLatency = deviceInfo->defaultLowInputLatency;
		parameters.hostApiSpecificStreamInfo = nullptr;

		PaStream* paStream = reinterpret_cast<PaStream*>(_stream);
		error = Pa_OpenStream(&paStream, &parameters, nullptr, _frequency, _frameCount, paNoFlag, audioInputCallback, this);
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
	}

	void AudioInput::setStoredLength(double storedLength)
	{
		_storedSamples = storedLength * _frequency * _channelCount;
	}

	double AudioInput::getStoredLength() const
	{
		return static_cast<double>(_storedSamples) / (_frequency * _channelCount);
	}

	bool AudioInput::isValid() const
	{
		return _stream;
	}

	int AudioInput::internalCallback(const int32_t* input, unsigned long frameCount)
	{
		_samplesMutex.lock();

		_samples.insert(_samples.end(), input, input + frameCount * _channelCount);
		_sampleCount += frameCount;

		const int64_t throwedSamples = _samples.size() - _storedSamples;
		if (throwedSamples > 0)
		{
			_samples.erase(_samples.begin(), _samples.begin() + throwedSamples * _channelCount);
			_currentSample += throwedSamples;
		}

		_samplesMutex.unlock();

		return paContinue;
	}

	void AudioInput::getRawSamples(int32_t* samples, uint64_t timeFrom, uint64_t timeTo)
	{
		assert(isValid());

		_samplesMutex.lock();

		if (timeFrom > _currentSample)
		{
			_samples.erase(_samples.begin(), _samples.begin() + (timeFrom - _currentSample) * _channelCount);
			_samples.shrink_to_fit();
		}

		auto it = _samples.begin();
		auto itEnd = it + (timeTo - timeFrom) * _channelCount;
		std::copy(it, itEnd, samples);

		_samples.erase(it, itEnd);
		_currentSample = timeTo;

		_samplesMutex.unlock();
	}

	AudioInput::~AudioInput()
	{
		if (isValid())
		{
			PaStream* paStream = reinterpret_cast<PaStream*>(_stream);

			Pa_AbortStream(paStream);
			Pa_CloseStream(paStream);
			Pa_Terminate();
		}
	}
}
