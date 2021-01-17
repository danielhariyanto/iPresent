from flask import Flask, render_template, url_for, request, redirect
from google.cloud import storage
import algorithms
from data import filler_words, hedging_language
import gensim.downloader as api
from heapq import nlargest
import asyncio


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
    try:
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
        clarity = int(round(result.confidence, 2) * 100)
        #brevity
        filler_phrase_counts, filler_all_counts = algorithms.count_phrases(transcript, filler_words)
        hedging_phrase_counts, hedging_all_counts = algorithms.count_phrases(transcript, hedging_language)
        
        #cadence
        wpm = int(algorithms.cadence(words))
        #common words
        #common_words = algorithms.vocabulary(transcript)
        common_words = {
            "common_1": "first",
            "common_1_count": 8,
            "common_2": "second",
            "common_2_count": 5,
            "common_3": "third",
            "common_3_count": 3,
            "common_4": "fourth",
            "common_4_count": 2
        }
        common_1 = common_words["common_1"]
        common_1_count = common_words["common_1_count"]
        common_2 = common_words["common_2"]
        common_2_count = common_words["common_2_count"]
        common_3 = common_words["common_3"]
        common_3_count = common_words["common_3_count"]
        common_4 = common_words["common_4"]
        common_4_count = common_words["common_4_count"]
        #sentiment analysis
        sentiment_score, sentiment_magnitude = asyncio.run(algorithms.sample_analyze_sentiment(transcript))
        #AI impression (JSON)
        ai_impression = asyncio.run(algorithms.ai_impression(transcript, wv))
        #expressiveness
        emote_stats = asyncio.run(algorithms.perform_audio_analysis(bucket_name, blob_name))


        ### DELETE AUDIO FILE ###
        delete_blob(bucket_name, blob_name)

        return render_template("metrics.html", clarity=clarity, brevity=filler_all_counts+hedging_all_counts, wpm=wpm*60, common_words=common_words, common_1=common_1, common_1_count=common_1_count, common_2=common_2, common_2_count=common_2_count, common_3=common_3, common_3_count=common_3_count, common_4=common_4, common_4_count=common_4_count, sentiment_score=sentiment_score, sentiment_magnitude=sentiment_magnitude, ai_impression=ai_impression, expressiveness=emote_stats[1].get("Passionate"))
    
    except:
        return render_template("404.html")

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
    load_model()
    app.run(debug=True)