import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import os
from pydub import AudioSegment,effects
from matplotlib import pyplot as plt
import torch
from pydub import AudioSegment, effects


def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    # plt.title("Signal Wave...")
    # plt.plot(wav)
    # plt.show()
    return wav

#spec 
def preprocess(file_path): 
    rawsound = AudioSegment.from_wav(file_path)
    normalizedsound = effects.compress_dynamic_range(rawsound)
    normalizedsound.export(os.path.join("C:/Users/David/Documents/Audio_Test/test.wav"), format="wav")
    wav = load_wav_16k_mono("test.wav")
    wav = wav[:16000] # 35000 sample length (can make smaller to reduce processing time but will affect accuracy)
    zero_padding = tf.zeros([16000] - tf.shape(wav), dtype=tf.float32) 
    wav = tf.concat([zero_padding, wav],0) # pad clips longer than 25000 samples wiht zeros
    spectrogram = tf.signal.stft(wav, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    spectrogram = tf.expand_dims(spectrogram,0)
    
    return spectrogram


# def preprocessing_Audio(waveform, rate):
#     waveform = waveform/32768
#     waveform = tf.convert_to_tensor(waveform, dtype = tf.float32)
#     sample_rate = tf.convert_to_tensor(rate, dtype = tf.float32)
#     spec = preprocess(waveform, sample_rate)
#     spec = tf.expand_dims(spec,0)

#     return spec

