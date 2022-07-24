import os
import sounddevice as sd
from scipy.io.wavfile import write

fs = 48000
s = 3

def memulaiRecord(fs, s):
    myrecording = sd.rec(int(s * fs), samplerate=fs, channels=2)
    sd.wait()
    return myrecording

def simpanRecord(i):
    if(i-1 < 9):
        write("./dataNew/" + "newData (00"+str(i)+")" + ".wav", fs, memulaiRecord(fs, s))    
    elif(i-1 >= 9):
        write("./dataNew/" + "newData (0"+str(i)+")" + ".wav", fs, memulaiRecord(fs, s))
    elif(i-1 >= 99):
        write("./dataNew/" + "newData ("+str(i)+")" + ".wav", fs, memulaiRecord(fs, s))
    elif(i-1 >= 999):
        print("Tidak dapat melanjutkan")        
        print("File audio history mencapai 1000, mohon bersihkan")
        print(exit())
