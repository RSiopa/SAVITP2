import pyttsx3 
#sudo apt-get install libespeak1


speaker = pyttsx3.init()
speaker.say('This room has: One mug and two hats.')
speaker.runAndWait()