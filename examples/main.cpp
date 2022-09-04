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

		- Rename "playSound" "scheduleSound"
		- Do the difference between playing and scheduled !
		- stopSound -> unscheduleSound
		
		- Better structure than std::multimap<index, soundplayinfo> ?
	*/

	crz::AudioOutput audioOutput(4);

	audioOutput.createSound<crz::SoundFile>("build/test.wav", crz::SoundFileFormat::Wave);
	audioOutput.playSound(0, 1.0, 10.0, 5.0, false);
	std::this_thread::sleep_for(std::chrono::milliseconds(7000));
	audioOutput.playSound(0, 0.0, 15.0, 5.0);

	std::this_thread::sleep_for(std::chrono::seconds(1000));

	return 0;
}
