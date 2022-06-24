import librosa, numpy, math
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

def frames(y,sr):
    #y,sr = librosa.load(f'{str(input_path)}/{str(name)}.wav')
    utterance_length = sr*0.02      # 20 milliseconds sections
    padding = int(math.ceil(len(y)/(utterance_length))*(utterance_length)-len(y))
    y = numpy.concatenate((numpy.array([0]*padding),y),axis=0)
    return [y[int(utterance_length*i):int((utterance_length*i)+utterance_length)] for i in range(int(len(y)/utterance_length))],sr

def shave_wav(y,sr):
    #y,sr = librosa.load(f'{str(input_path)}/{str(name)}.wav')
    mercy = int(numpy.round(0.2*sr))
    rough_silence_mask = numpy.round((abs(y)+1)**10-.6)
    mask = numpy.ones(shape=len(y))
    for verse in [1,-1]:
        count = 0
        while rough_silence_mask[verse*count] == 0:
            mask[verse*count] = 0
            count += 1
    mask = list(mask)
    for verse in [1,-1]:
        for i,el in enumerate(mask[::verse]):
            if el==1:
                if verse == 1:
                    mask[i-mercy+1:i+1] = [1]*mercy
                else:
                    if len(y)-i+mercy > len(y):
                        mask[(len(y)-i-1)+1:] = [1]*(i)
                    else:
                        mask[len(y)-i:len(y)-i+mercy] = [1]*mercy
                break
    mask = numpy.array(mask)
    return y[mask==True],sr

def pad_shave_frame(name):
    y,sr = librosa.load(name)
    y,sr = shave_wav(y,sr)
    y = numpy.concatenate(
        (y,numpy.zeros(shape=(25*sr)-len(y)))
    )
    y,sr = frames(y,sr)
    return y,sr

def plot_frames(y,sr):
    # y as list of frames
    for i in range(len(y)):
        plt.plot(
            list(range(int(sr*0.02)*(i),int(sr*0.02)*(i+1))),
            y[i]
        )
    plt.show()