import speech_recognition as sr

r = sr.Recognizer()
mic = sr.Microphone()

"""
# observe the list of microphones of the pc
print(sr.Microphone.list_microphone_names())
# choosing a specific microphone
mic = sr.Microphone(device_index=3)
"""
duration_calibration = 1
print("Note - Before speaking please wait for the duration configured for calibration, which is: ", duration_calibration)
print("Recording Starts")
with mic as source:
    # to handle ambient noise 
    r.adjust_for_ambient_noise(source, duration=duration_calibration)
    # takes an audio source and 
    # records input from the source until silence is detected
    # note - if we use "adjust_for_ambient_noise", we need to wait 1 sec before speaking beacause the first sec will be
    # for calibration of the noise and we we do not wait, the speech recognition will be only from sec 1 and we miss the first sec
    audio = r.listen(source)
print("Recording Stoped")

try:
    # for english by default
    print("Recognized: ", r.recognize_google(audio))
    # for hebrew
    # print("Recognized: ", r.recognize_google(audio, language='he'))
except:
    print("Did not recognize anything valid")
