from flask import Flask, render_template, url_for, request, redirect

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('start', methods=['GET', 'POST'])
def start():
    if request.method == "POST":
        f = request.files['audio_data']
        with open('audio.wav', 'wb') as audio:
            f.save(audio)
        print('file uploaded successfully')

        return render_template('present.html', request="POST")
    else:
        return render_template("present.html")


if __name__ == "__main__":
    app.run(debug=True)