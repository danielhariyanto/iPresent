from flask import Flask, render_template, url_for, request, redirect
from google.cloud import storage
import algorithms
from data import filler_words, hedging_language
import gensim.downloader as api


app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/present', methods=['GET'])
def start():
    #return render_template("testaudio.html")
    return render_template("present.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    return render_template("upload.html")


@app.route('/results', methods=['GET'])
def results():
    bucket_name = request.args.get("bucket")
    blob_name = request.args.get("key")

    gcs_uri = "gs://"+bucket_name+"/"+blob_name

    #algorithms go here
    #speech-to-text
    result = algorithms.transcribe_gcs(gcs_uri)
    words = result.words
    #transcript
    transcript = result.transcript
    #clarity
    clarity = result.confidence
    #brevity
    filler_phrase_counts, filler_all_counts = algorithms.count_phrases(transcript, filler_words)
    hedging_phrase_counts, hedging_all_counts = algorithms.count_phrases(transcript, hedging_language)
    #cadence
    wpm = algorithms.cadence(words)
    #common words
    #common_words = algorithms.vocabulary(transcript)
    common_words = {
        "first_word": "first",
        "first_word_count": 8,
        "second_word": "second",
        "second_word_count": 5,
        "third_word": "third",
        "third_word_count": 3,
        "fourth_word": "fourth",
        "fourth_word_count": 2
    }
    first_word = common_words["first_word"]
    first_word_count = common_words["first_word_count"]
    second_word = common_words["second_word"]
    second_word_count = common_words["second_word_count"]
    third_word = common_words["third_word"]
    third_word_count = common_words["third_word_count"]
    fourth_word = common_words["fourth_word"]
    fourth_word_count = common_words["fourth_word_count"]
    #sentiment analysis
    sentiment_score, sentiment_magnitude = algorithms.sample_analyze_sentiment(transcript)
    #AI impression (JSON)
    #ai_impression = algorithms.ai_impression(transcript, wv)
    ai_impression = "Beautiful"
    #expressiveness
    #emote_stats = algorithms.perform_audio_analysis(bucket_name, blob_name)
    emote_stats = {'Neutral': 0.7777777777777778, 'Passionate': 0.2222222222222222}


    ### DELETE AUDIO FILE ###
    delete_blob(bucket_name, blob_name)

    return render_template("metrics.html", clarity=clarity, brevity=filler_all_counts+hedging_all_counts, wpm=wpm*60, common_words=common_words, first_word=first_word, first_word_count=first_word_count, second_word=second_word, second_word_count=second_word_count, third_word=third_word, third_word_count=third_word_count, fourth_word=fourth_word, fourth_word_count=fourth_word_count, sentiment_score=sentiment_score, sentiment_magnitude=sentiment_magnitude, ai_impression=ai_impression, expressiveness=emote_stats["Passionate"])

def load_model():
    global wv
    wv = api.load('word2vec-google-news-300')

def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket after algorithms"""
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()

    print("Blob {} deleted.".format(blob_name))


if __name__ == "__main__":
    #load_model()
    app.run(debug=True)