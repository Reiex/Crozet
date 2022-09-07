#include <Crozet/Core/Core.hpp>
#include <Crozet/Private/Private.hpp>

namespace crz
{
	int AudioDevice::getDefaultInputDeviceIndex()
	{
		Pa_Initialize();
		PaDeviceIndex index = Pa_GetDefaultInputDevice();
		Pa_Terminate();

		return index;
	}

	int AudioDevice::getDefaultOutputDeviceIndex()
	{
		Pa_Initialize();
		PaDeviceIndex index = Pa_GetDefaultOutputDevice();
		Pa_Terminate();

		return index;
	}

	int AudioDevice::getDeviceCount()
	{
		Pa_Initialize();
		PaDeviceIndex count = Pa_GetDeviceCount();
		Pa_Terminate();

		return count;
	}

	AudioDevice AudioDevice::getAudioDevice(int deviceIndex)
	{
		Pa_Initialize();

		assert(deviceIndex < getDeviceCount());

		AudioDevice device;
		const PaDeviceInfo* info = Pa_GetDeviceInfo(deviceIndex);

		device.name = info->name;
		device.maxInputChannels = info->maxInputChannels;
		device.maxOutputChannels = info->maxOutputChannels;
		device.defaultLowInputLatency = info->defaultLowInputLatency;
		device.defaultLowOutputLatency = info->defaultLowOutputLatency;
		device.defaultHighInputLatency = info->defaultHighInputLatency;
		device.defaultHighOutputLatency = info->defaultHighOutputLatency;
		device.defaultSampleRate = info->defaultSampleRate;

		Pa_Terminate;

		return device;
	}
}
