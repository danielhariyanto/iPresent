from flask import Flask, render_template, url_for, request, redirect
from google.cloud import storage

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/start', methods=['GET', 'POST'])
def start():
    return render_template("testaudio.html")
    # return render_template("present.html")

@app.route('/upload', methods=['GET'])
def upload():
    return render_template("upload.html")

@app.route('/results', methods=['GET','POST'])
def results():
    if request.method == 'POST':
        file = request.form['file-upload']
        filepath = request.form['file-path']

        bucket_name = "hack_the_ne"
        source_file_name = filepath+"/"+file
        destination_blob_name = file

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print(
            "File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
            )
        )
    else:
        return redirect("/upload")
    return render_template("metrics.html")


if __name__ == "__main__":
    app.run(debug=True)
