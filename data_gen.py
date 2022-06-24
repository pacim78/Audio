from threading import Thread
import random, os, sys, time, pyaudio, wave
from rec import rec
import pandas as pd

def main():

    def get_corpus():
        global corpus,corpus_words
        with open('text_1.txt','r',encoding='utf8') as txt:
            corpus = txt.read()
            corpus_words = corpus.lower().split(' ')

        return corpus, corpus_words

    global counter,phrase, thread_running, corpus, corpus_words
    corpus,corpus_words = get_corpus()
    phrase = corpus_words[0]
    counter = 1
    thread_running = True    
    
    def forward(n):
        global counter,phrase
        if counter+n > len(corpus_words):
            pass
        else:
            counter+=n
        phrase = ''
        for i in range(counter):
            if len(' '.join([phrase.split('\n')[-1],corpus_words[i]])) < 75:
                phrase = ' '.join([phrase,corpus_words[i]])
            else:
                phrase = '\n'.join([phrase,corpus_words[i]])

    def backward(n):
        global counter,phrase
        if counter-n < 1:
            pass
        else:
            counter-=n
        phrase = ''
        for i in range(counter):
            if len(' '.join([phrase.split('\n')[-1],corpus_words[i]])) < 75:
                phrase = ' '.join([phrase,corpus_words[i]])
            else:
                phrase = '\n'.join([phrase,corpus_words[i]])
        
    def save():
        global counter
        with open('text_1.txt', 'w', encoding='utf8') as file:
            file.write(' '.join(corpus_words[counter:]))
    
    def delete_str():
        global counter,phrase,corpus,corpus_words
        save()
        counter = 1
        phrase = ''
        corpus,corpus_words = get_corpus()

    def stop_recording():
        input('choose to stop')

    def save_recording():
        global phrase,corpus,corpus_words,counter,thread_running
        num = 1
        thread_running = True
        while os.path.exists('rec_data/%s.wav'%(str(num))):
            num+=1
        
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
        
        def my_forever_while():
            global thread_running
            while thread_running:
                data = stream.read(chunk)
                frames.append(data)

        t1 = Thread(target=my_forever_while)
        t2 = Thread(target=stop_recording)
        
        print("START RECORDING")

        t1.start()
        t2.start()

        t2.join()
        thread_running = False

        stream.stop_stream()
        stream.close()
        p.terminate()
        print('FINISHED')

        still = True
        while still == True:
            wf = wave.open('rec_data/%s.wav'%(str(num)), 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
            wf.close()

            newfile = pd.DataFrame.from_dict({'audiofile':['%s.wav'%str(num)],'transcript':[phrase]})
            if os.path.exists('rec_data/dataset.csv'):
                newfile.to_csv('rec_data/dataset.csv', mode='a', header=False,encoding='utf8')
            else:
                newfile.to_csv('rec_data/dataset.csv',encoding='utf8')
                
            save()
            counter = 1
            phrase = ''
            corpus,corpus_words = get_corpus()
            still = False
            
        

    print('DataGen')
    while True:
        inp = input(f'\r<< < rec | del > >>\n\n{phrase}\n')
        if inp in ['<<','1']:
            backward(5)
        elif inp in ['<','2']:
            backward(1)
        elif inp in ['>','4']:
            forward(1)
        elif inp in ['>>','5']:
            forward(5)
        elif inp in ['rec','']:
            save_recording()
        elif inp in ['del']:
            delete_str()

if __name__ == '__main__':
    main()