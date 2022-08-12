import speech_recognition as sr

print("Version of speech_recognition: ", sr.__version__)

# recognizer object that has seven different sources (methods) for speech recognition
r = sr.Recognizer()

# loading an audio file to an audio file object
harvard_noise = sr.AudioFile('Speech_Recognition\harvard_noise.wav')
with harvard_noise as source:
    # dealing with noise using this method
    # changing the default duration of 1 sec to 0.5 for calibration so we won't miss any speech data between 0.5-1 sec
    r.adjust_for_ambient_noise(source, duration=0.5)
    # recording the content of the audio file
    audio = r.record(source)

# our recognizer will be set to google
# printing the content of the audio file
# print(r.recognize_google(audio, show_all=True))
print(r.recognize_google(audio))
