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
	struct AudioDevice
	{
		std::string name;
		int maxInputChannels;
		int maxOutputChannels;
		double defaultLowInputLatency;
		double defaultLowOutputLatency;
		double defaultHighInputLatency;
		double defaultHighOutputLatency;
		double defaultSampleRate;

		static int getDefaultInputDeviceIndex();
		static int getDefaultOutputDeviceIndex();

		static int getDeviceCount();
		static AudioDevice getAudioDevice(int deviceIndex);
	};
}
