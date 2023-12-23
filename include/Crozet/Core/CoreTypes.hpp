///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Pélérin Marius
//! \copyright The MIT License (MIT)
//! \date 2022-2023
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#define _CRT_SECURE_NO_WARNINGS

#include <cstdio>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <SciPP/SciPPTypes.hpp>
#include <Diskon/DiskonTypes.hpp>

#if defined(_WIN32)
	#ifdef CROZET_EXPORTS
		#define CRZ_API __declspec(dllexport)
	#else
		#define CRZ_API __declspec(dllimport)
	#endif
#elif defined(__linux__)
	#define CRZ_API
#else
	#error "Unrecognized platform"
#endif

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
