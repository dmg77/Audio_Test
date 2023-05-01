from keras.models import load_model
import sounddevice as sd
import numpy as np
from simpleaudio import get_spectrogram, label_names
import sys

model = load_model('C:/Users/David/Documents/Audio_Test/saved_as_keras')
sys.path.insert(0, 'C:/Users/David/Documents/Audio_Test')

duration = 5  # seconds
sample_rate = 16000  # Hz

audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()

spectrogram = get_spectrogram(audio)
spectrogram = np.expand_dims(spectrogram, axis=0)  # add batch dimension

predictions = model.predict(spectrogram)

predicted_class_index = np.argmax(predictions)
predicted_class = label_names[predicted_class_index]
