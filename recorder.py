import pyaudio
import wave
import numpy as np
import datetime
import os

FRAMES_PER_BUFFER = 1024*3 
FORMAT = pyaudio.paInt16
#Mono channel 
CHANNELS = 1
#RATE
RATE = 16000 # 44100
p = pyaudio.PyAudio()

def record_audio():
    dst_folder = "recorded_bad"
    # dst_folder = "recorded_good"
    
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    print("start recording...")

    frames = []
    # records for one second 
    seconds = 1.25
    for i in range(0, int(RATE / FRAMES_PER_BUFFER * seconds)):
        data = stream.read(FRAMES_PER_BUFFER)
        frames.append(data)

    print("recording stopped")

    stream.stop_stream()
    stream.close()
    
    
     # save the recorded data as a WAV file with date and time in the filename
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # old_filename = "recorded_file.wav"
    new_filename = f"recorded_file_{now}_BAD.wav"
    # new_filename = f"recorded_file_{now}_GOOD.wav"
    
    
    wf = wave.open(os.path.join(dst_folder , new_filename), "wb")
    # set the channels
    wf.setnchannels(CHANNELS)
    # set the sample format
    wf.setsampwidth(p.get_sample_size(FORMAT))
    # set the sample rate
    wf.setframerate(RATE)
    # write the frames as bytes
    wf.writeframes(b"".join(frames))
    # close the file
    wf.close()
    
    return os.path.join(dst_folder , new_filename)

    #return  np.frombuffer(b''.join(frames), dtype=np.int16),RATE,frames

def combineframes(frames):
    
    wf = wave.open("test.wav", "wb")
    # set the channels
    wf.setnchannels(CHANNELS)
    # set the sample format
    wf.setsampwidth(p.get_sample_size(FORMAT))
    # set the sample rate
    wf.setframerate(RATE)
    # write the frames as bytes
    wf.writeframes(b"".join(frames))
    # close the file
    wf.close()


def terminate():
    p.terminate()


