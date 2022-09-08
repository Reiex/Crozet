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

		- AudioInput

		- Make sure filters cannot be created once sounds started to play (not just scheduled ? use _currentTime ?)
	*/

	crz::AudioOutput audioOutput;
	audioOutput.createSound<crz::AudioInput>();
	audioOutput.createSound<crz::SoundFile>("build/test.wav", crz::SoundFileFormat::Wave);
	audioOutput.getSound(1)->addFilter<crz::FilterPlaySpeed>(0.75);
	
	audioOutput.scheduleSound(1, 1.0, 10.0, 5.0, false);
	audioOutput.scheduleSound(1, 7.0, 15.0, 5.0);
	audioOutput.scheduleSound(0, 13.0, 13.0);

	std::this_thread::sleep_for(std::chrono::seconds(1000));


	return 0;
}
