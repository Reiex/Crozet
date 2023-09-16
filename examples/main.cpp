///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//! \file
//! \author Pélégrin Marius
//! \copyright The MIT License (MIT)
//! \date 2022-2023
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <Crozet/Crozet.hpp>
#include <portaudio.h>
#include <iostream>

int main()
{
	/*
	
	TODO:
		- Create a debug manager for
			- Retrieving PA errors
			- Retrieving callback CPU load
			- Retrieve other datas such as frame dropped... status of different streams...

		- Allow for reverse playing / jump back in time
			- Add bool SoundBase::_reversible
			- Add open mode in SoundFile : Stream/Store (not enum, just boolean in constructor)
			- Add handle mode in SoundBuffer : Stream/Copy (not enum, just boolean in constructor)
	*/

	crz::AudioOutput audioOutput;
	audioOutput.createSound<crz::SoundFile>("examples/assets/Balavoine.wav", crz::SoundFileFormat::Wave);
	audioOutput.createSound<crz::AudioInput>();
	audioOutput.getSound(0)->addFilter<crz::FilterPlaySpeed>(0.75);
	
	audioOutput.scheduleSound(0, 1.0, 10.0, 2.5, false);
	audioOutput.scheduleSound(0, 4.0, 12.5, 2.5);
	audioOutput.scheduleSound(1, 7.0, 13.0);

	std::this_thread::sleep_for(std::chrono::seconds(1000));

	return 0;
}
