///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Reiex
//! \copyright The MIT License (MIT)
//! \date 2022-2023
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <SciPP/SciPPTypes.hpp>
#include <Diskon/DiskonTypes.hpp>

#include <condition_variable>
#include <map>
#include <mutex>
#include <thread>

namespace crz
{
	struct AudioDevice;

	class AudioOutput;
	class AudioInput;


	class SoundSource;

	class SoundBase;
	class SoundFile;
	class SoundBuffer;

	class FilterBase;
	class FilterPlaySpeed;
	// TODO: class FilterEnvelope;
}
