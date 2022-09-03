#include <Crozet/Crozet.hpp>
#include <portaudio.h>
#include <iostream>

int main()
{
	/*
	
	TODO:
		- Activate and deactivate stream (or stop/start) for performances
		- Create a debug manager for
			- Retrieving PA errors
			- Retrieving callback CPU load
			- Retrieve other datas such as frame dropped... status of different streams...
	
	*/

	crz::AudioOutput audioOutput(4);

	audioOutput.createSound<crz::SoundFile>("build/test.wav", crz::SoundFileFormat::Wave);
	audioOutput.playSound(0);

	while (true);

	return 0;
}
