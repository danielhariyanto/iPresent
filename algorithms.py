from google.cloud import speech_v1p1beta1 as speech
from google.cloud import storage

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

import gensim.downloader as api

import string
import json 
import re


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

    return total_time, pace

'''
Compute the number of seconds for each word in `words`
'''
def seconds_per_word(words, verbose=False):
    word_times = {}
    for word_info in words:
        word = word_info.word
        start_time = word_info.start_time.total_seconds()
        end_time = word_info.end_time.total_seconds()
        confidence = word_info.confidence

        if verbose: 
            print(
                f"{word}, start_time: {start_time}, end_time: {end_time}, confidence: {confidence}"
            )

        time_of_word = round(end_time - start_time, 3)
        word_times[word] = time_of_word
        
    return word_times