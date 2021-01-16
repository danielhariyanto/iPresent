from flask import Flask, render_template, url_for, request, redirect
from google.cloud import storage

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
    #algorithms go here

    """Deletes a blob from the bucket after algorithms"""
    bucket_name = request.args.get("bucket")
    blob_name = request.args.get("key")

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()

    print("Blob {} deleted.".format(blob_name))
    return render_template("metrics.html")


if __name__ == "__main__":
    app.run(debug=True)