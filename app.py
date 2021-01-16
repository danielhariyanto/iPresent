from flask import Flask, render_template, url_for, request, redirect
from google.cloud import storage

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/start', methods=['GET', 'POST'])
def start():
    return render_template("present.html")

@app.route('/upload', methods=['POST'])
def upload():
    filenamebytes = request.data
    filename = filenamebytes.decode("utf-8")
    bucket_name = "hack_the_ne"
    source_file_name = filename
    destination_blob_name = filename.split('/')[-1]

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )

    return redirect("/results")

@app.route('/results', methods=['GET'])
def results():
    return render_template("metrics.html")

if __name__ == "__main__":
    app.run(debug=True)