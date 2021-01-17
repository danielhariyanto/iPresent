from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage
from google.cloud import language_v1

import spacy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Reshape, Bidirectional
from tensorflow.keras import backend as K

from tqdm import tqdm

import string
import json 
import re

import pydub

import IPython.display as ipd

import librosa
import librosa.display
from librosa import feature

import soundfile as sf
import io

import operator


def transcribe_gcs(gcs_uri):
    """Asynchronously transcribes the audio file specified by the gcs_uri."""
    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        audio_channel_count=1,
        language_code="en-US",
        enable_word_time_offsets=True, 
        enable_word_confidence=True,
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    response = operation.result(timeout=90)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        alternative = result.alternatives[0]
        print(u"Transcript: {}".format(result.alternatives[0].transcript))
        print("Confidence: {}".format(result.alternatives[0].confidence))
        
    return response.results[0].alternatives[0]


'''
This function returns a count of how many times 
each phrase in the `phrases` list is in the 
`transcript` string. 
'''
def count_phrases(transcript, phrases, return_phrase_counts=True): 
    space_transcript = ' ' + transcript + ' '
    phrase_counts = {}
    all_counts = 0 
    for phrase in phrases: 
        space_phrase = ' ' + phrase + ' '
        count = space_transcript.count(space_phrase)
        if count > 0: 
            phrase_counts[phrase] = count
            all_counts = all_counts + count
    
    if return_phrase_counts: 
        return phrase_counts, all_counts # Return the counts for each filler phrase
    else:
        return all_counts # Only return the total number of counts

'''
Compute the total duration of the speech in seconds, given
the dict of words. 
'''
def speech_time(words): 
    first_word = words[0]
    last_word = words[-1]
    
    start_time = first_word.start_time.total_seconds()
    end_time = last_word.end_time.total_seconds()
    
    return end_time - start_time

'''
Computes the number of words spoken in the speech per 
second, given `words` as a dict. 
'''
def words_per_second(words, time): 
    return len(words) / time

'''
Cadence algorithm (import this one, not the two above)
'''
def cadence(words):
    total_time = speech_time(words)
    pace = words_per_second(words, total_time)

    return pace

'''
Common words
'''
def vocabulary(transcript):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(transcript)

    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    verb_phrases = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    adjectives = [token.lemma_ for token in doc if token.pos_ == "ADJ"]

    print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
    print("Adjectives:", [token.lemma_ for token in doc if token.pos_ == "ADJ"])


"""
Analyzing Sentiment in a String

Returns dict of scores for the entire document, and a list of
dicts representing scores for each sentence in the document. 

Args:
    text_content The text content to analyze
"""
def sample_analyze_sentiment(text_content):
    client = language_v1.LanguageServiceClient()

    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT

    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_sentiment(request = {'document': document, 'encoding_type': encoding_type})
    
    # Get overall sentiment of the input document
    print(u"Document sentiment score: {}".format(response.document_sentiment.score))
    print(
        u"Document sentiment magnitude: {}".format(
            response.document_sentiment.magnitude
        )
    )
    
    return response.document_sentiment.score, response.document_sentiment.magnitude


def create_embedding(transcript, wv):
#     print(transcript)
    X = []
    found_words = []
    transcript = clean(transcript) # remove any punctuation
    
    print(transcript)
    words = transcript.split()
    for word in words: 
        try:
            found_words.append(wv[word])
        except: 
            continue
            
    embedding = np.asarray(found_words)
    mean = np.mean(embedding, axis=0)
    mean = mean.tolist()

    if type(mean) == list: 
        X.append(mean)

    X = np.array(X)
    return X


def ai_impression(transcript, wv):
    wv = load_wv()
    embedding = create_embedding(transcript, wv)

    model_filepath = './static/models/ted_analysis_model'
    model = tf.keras.models.load_model(model_filepath)
    
    pred = model.predict(embedding)[0]
    cols = ['Beautiful', 'Confusing', 'Courageous', 'Funny', 'Informative', 'Ingenious', 'Inspiring', 'Longwinded', 'Unconvincing', 'Fascinating', 'Jaw-dropping', 'Persuasive', 'OK', 'Obnoxious']

    ted_dict = {}
    for val, col in zip(pred, cols):
        ted_dict[col] = val
    
    word_result = max(stats.iteritems(), key=operator.itemgetter(1))[0]

    return word_result


'''
This function reads in a .wav file. 

Set cloud to true if data file path is from the cloud.
'''
def read_wav(bucket, file_name):
    # read a blob
    
    blob = bucket.blob(file_name)
    file_as_string = blob.download_as_string()

    # convert the string to bytes and then finally to audio samples as floats 
    # and the audio sample rate
    data, sample_rate = sf.read(io.BytesIO(file_as_string))

    # print(len(data.shape))
    if len(data.shape) == 1: 
        left_channel = data[:]  # I assume the left channel is column zero
    elif len(data.shape) == 2: 
        left_channel = data[:, 0]  # I assume the left channel is column zero

    # enable play button in datalab notebook
    aud = ipd.Audio(left_channel, rate=sample_rate)
    
    return aud, data, sample_rate

'''
Get spectrogram for each audio file. 
'''
def get_spectrogram(y, sr):
    spectrogram = librosa.feature.melspectrogram(y, sr)
    return spectrogram

# Intensity analysis (0 = neutral, 1 = strong/passionate)
def get_intensity_analysis(emotion_intensity):
    emotions = []
    total = 0
    sums = {0: 0, 1: 0}
    for emotion in emotion_intensity: 
        if emotion[0] >= 0.5:
            emotions.append('Passionate')
            sums[1] += 1
        else:         
            emotions.append('Neutral')
            sums[0] += 1
        total += 1
    percentages = {'Neutral': sums[0] / total, 'Passionate': sums[1] / total}
    return emotions, percentages

def perform_audio_analysis(bucket_name, filename):
    audio_model_filepath = './static/models/audio_analysis_model'
    emotion_intensity_model_filepath = './static/models/emotion_intensity_model'

    # Create a Cloud Storage client.
    gcs = storage.Client()

    # Get the bucket that the file will be uploaded to.
    bucket = gcs.get_bucket(bucket_name)
    aud, y, sr = read_wav(bucket, filename)
    
    # Convert into spectrogram
    spectrogram = get_spectrogram(y[:, 0], sr)
    
    # Load models
    audio_model = tf.keras.models.load_model(audio_model_filepath)
    emotion_intensity_model = tf.keras.models.load_model(emotion_intensity_model_filepath)
    
    # Break into time-based chunks, every ~3 seconds 
    num_chunks = spectrogram.shape[-1] // 124
    sound_chunks = []
    for i in range(num_chunks): 
        chunk = spectrogram[:, i * 124 : (i+1) * 124]
        sound_chunks.append(chunk)
    sound_chunks = np.array(sound_chunks)
    sound_chunks = np.expand_dims(sound_chunks, axis=-1)
    
    # Make predictions
    audio_analysis = audio_model.predict(sound_chunks)
    emotion_intensity = emotion_intensity_model.predict(sound_chunks)

    emotions, emotion_percentages = get_intensity_analysis(emotion_intensity)
    
    # Classifies audio based on neutrality vs. strength (aka passion/intensity)
    emote_stats = (emotions, emotion_percentages)
    return emote_stats