import speech_recognition as sr

print("Version of speech_recognition: ", sr.__version__)

# recognizer object that has seven different sources (methods) for speech recognition
r = sr.Recognizer()

# loading an audio file to an audio file object
harvard = sr.AudioFile('Speech_Recognition\harvard.wav')
with harvard as source:
    # recording the content of the audio file
    # audio = r.record(source)
    # audio3 = r.record(source, offset=4, duration=3)       # offset of the file                          # all the duration
    audio1 = r.record(source, duration=4)                   # first 4 seconds
    audio2 = r.record(source, duration=4)                   # next 4 seconds

# our recognizer will be set to google
# printing the content of the audio file
print(r.recognize_google(audio1))
print(r.recognize_google(audio2))