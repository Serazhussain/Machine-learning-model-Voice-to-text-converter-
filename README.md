# Machine-learning-model-Voice-to-text-converter-
In this machine learning model  (voice to text ) we have done some test with our own voice and found that how this model works.

!pip install librosa
!pip install transformers
!pip install SoundFile
!pip3 install torch torchvision torchaudio
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
C:\Users\hp\New folder (2)\lib\site-packages\paramiko\transport.py:219: CryptographyDeprecationWarning: Blowfish has been deprecated
  "class": algorithms.Blowfish,
import IPython.display as display
display.Audio("C:\\Users\\hp\\taken.wav", autoplay=True)
# Importing Wav2Vec pretrained model

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
The tokenizer class you load from this checkpoint is not the same type as the class this function is called from. It may result in unexpected tokenization. 
The tokenizer class you load from this checkpoint is 'Wav2Vec2CTCTokenizer'. 
The class this function is called from is 'Wav2Vec2Tokenizer'.
C:\Users\hp\New folder (2)\lib\site-packages\transformers\models\wav2vec2\tokenization_wav2vec2.py:752: FutureWarning: The class `Wav2Vec2Tokenizer` is deprecated and will be removed in version 5 of Transformers. Please use `Wav2Vec2Processor` or `Wav2Vec2CTCTokenizer` instead.
  warnings.warn(
Some weights of Wav2Vec2ForCTC were not initialized from the model checkpoint at facebook/wav2vec2-base-960h and are newly initialized: ['wav2vec2.masked_spec_embed']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# Loading the audio file

filename = "taken.wav" # "my-audio.wav" or "taken.wav"

audio, rate = librosa.load(filename, sr = 16000)
# Taking an input value
input_values = tokenizer(audio, return_tensors = "pt").input_values
input_values
tensor([[-0.0001, -0.0001, -0.0001,  ..., -0.0305, -0.0323, -0.0001]])
# feeding input values to our model - storing logits
logits = model(input_values).logits
# max probab values - storing predicted id's
prediction = torch.argmax(logits, dim = -1)
# Passing the prediction to the tokenzer decode to get the transcription
transcription = tokenizer.batch_decode(prediction)[0]
transcription
"BUT IF YOU DON'T I WILL LOOK FOR YOU I WILL FIND YOU AND I WILL KILL YOU"
