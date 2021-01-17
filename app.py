from flask import Flask, render_template, url_for, request, redirect
from google.cloud import storage
import algorithms
from data import filler_words, hedging_language


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
    clarity = int(round(result.confidence, 2) * 100)
    #brevity
    filler_phrase_counts, filler_all_counts = algorithms.count_phrases(transcript, filler_words)
    hedging_phrase_counts, hedging_all_counts = algorithms.count_phrases(transcript, hedging_language)
    #cadence
    total_time, pace = algorithms.cadence(words)
    pace = int(pace)
    #seconds / word
    word_times = algorithms.seconds_per_word(words, verbose=True)
    #sentiment
    sentiment = "negative"
    magnitude = int(round(0.67888, 2) * 100)
    #ai impression
    impression = "beautiful"
    #common words
    common_1 = "their"
    common_2 = "gone"
    common_3 = "present"

    ### DELETE AUDIO FILE ###
    #delete_blob(bucket_name, blob_name)

    
    return render_template("metrics.html", sentiment=sentiment, megnitude=magnitude, impression=impression, clarity=clarity, brevity=filler_all_counts+hedging_all_counts, cadence=pace, common_1=common_1, common_2=common_2, common_3=common_3)


def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket after algorithms"""
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()

    print("Blob {} deleted.".format(blob_name))


if __name__ == "__main__":
    app.run(debug=True)