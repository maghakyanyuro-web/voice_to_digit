from telebot import TeleBot
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import librosa
from pydub import AudioSegment
import torchaudio
import dotenv


dotenv.load_dotenv()

SAVE_DIR = 'voices'
os.makedirs(SAVE_DIR, exist_ok=True)
TOKEN = os.getenv('TOKEN')
bot = TeleBot(token=TOKEN)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

def audio_to_mel_spectrogram(file_path, sr=16000, n_mels=128, max_len=10):
    y, _ = librosa.load(file_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # (n_mels, time)
    log_mel_spec = log_mel_spec.T
    if log_mel_spec.shape[0] < max_len:
        pad_width = max_len - log_mel_spec.shape[0]
        log_mel_spec = np.pad(log_mel_spec, ((0, pad_width), (0, 0)))
    else:
        log_mel_spec = log_mel_spec[:max_len, :]
    return log_mel_spec


def predict(voice):
    data = audio_to_mel_spectrogram(voice)
    data = data.reshape(1, 10, 128, 1)
    y_pred = model.predict(data)
    return np.argmax(y_pred)



@bot.message_handler(commands=['start'])
def start_bot(message):
    bot.send_message(message.chat.id, 'Ուղարկեք ձայնային հաղորդագրություն, ես կվերադարձնեմ ինֆորմացիան 🤖🤖🤖')


@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    file_info = bot.get_file(message.voice.file_id)
    file_path = file_info.file_path
    downloaded_file = bot.download_file(file_path)
    filename = os.path.join(SAVE_DIR, f'{message.message_id}.ogg')
    with open(filename, 'wb') as new_file:
        new_file.write(downloaded_file)
    waveform, sample_rate = torchaudio.load(filename)
    torchaudio.save("output.wav", waveform, sample_rate)
    bot.send_message(message.chat.id, f'Ձեր թիվը {predict("output.wav")} է')


bot.polling()