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
	*/

	crz::AudioOutput audioOutput(4);

	audioOutput.createSound<crz::SoundFile>("build/test.wav", crz::SoundFileFormat::Wave);
	audioOutput.scheduleSound(0, 1.0, 10.0, 10.0, false);
	audioOutput.scheduleSound(0, 12.0, 20.0, 10.0);

	std::this_thread::sleep_for(std::chrono::seconds(1000));

	return 0;
}
