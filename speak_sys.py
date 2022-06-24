import pyaudio
import wave
import librosa, librosa.feature, librosa.feature.inverse
from scipy.io.wavfile import write
import numpy as np

def speak_clean(audio):
    CHUNK = 1024
    wf = wave.open(audio, 'rb')

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)
    data = wf.readframes(CHUNK)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(CHUNK)
    stream.stop_stream()
    stream.close()
    p.terminate()

def smooth(data,window_width):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0)) 
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec

def voice_in_the_machine(audio):
    x,sr=librosa.load(audio)
    mfccs = librosa.feature.mfcc(y=x,sr=sr,n_mfcc=20)
    wave = librosa.feature.inverse.mfcc_to_audio(mfccs)
    amplitude = np.iinfo(np.int16).max
    data = amplitude*np.sin(-1*wave*np.pi)
    data = smooth(data,2)
    write('example.wav',sr,data.astype(np.int16))

ask = input('please select:\n1)\tspeak_clean\n2)\tvoice_in_the_machine\n... ')
if ask == '1':
    audio = input('audio path... ')
    speak_clean(audio)
elif ask == '2':
    audio = input('audio path... ')
    voice_in_the_machine(audio)
    speak_clean('example.wav')