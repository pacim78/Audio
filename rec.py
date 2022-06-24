import pyaudio
import wave
from threading import Thread

thread_running = True

def take_input():
    user_input = input('Type user input: ')


def rec(name,func,where):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100
    p = pyaudio.PyAudio()
    frames = []

    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
    print("start... recording")

#    seconds = 5
#    for i in range(0, int(rate/chunk * seconds)):

    def my_forever_while():
        global thread_running
        while thread_running:
            data = stream.read(chunk)
            frames.append(data)

    t1 = Thread(target=my_forever_while)
    t2 = Thread(target=func)

    t1.start()
    t2.start()

    t2.join()
    thread_running = False
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(where+'%s.wav'%(name), 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print('finished')

def ask():
    name = input('name... ')
    rec(name,take_input,'')
    
if __name__ == "__main__":
    ask()