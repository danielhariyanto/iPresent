//webkitURL is deprecated but nevertheless
URL = window.URL || window.webkitURL;

var gumStream;                      //stream from getUserMedia()
var rec;                            //Recorder.js object
var input;                          //MediaStreamAudioSourceNode we'll be recording

// shim for AudioContext when it's not avb. 
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //audio context to help us record

var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");
var pauseButton = document.getElementById("pauseButton");

//add events to those 2 buttons
recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", (stopRecording));
pauseButton.addEventListener("click", pauseRecording);

var durationLabel = document.getElementById("duration");
let isPaused = false;
let totalSeconds = 0;

function startRecording() {
    console.log("recordButton clicked");
    setInterval(incrementDuration, 1000);
    document.querySelector('.present-header').innerHTML = 'you got this!'
    recording = true;

    /*
        Simple constraints object, for more advanced audio features see
        https://addpipe.com/blog/audio-constraints-getusermedia/
    */

    var constraints = { audio: true, video: false }

    /*
        Disable the record button until we get a success or fail from getUserMedia() 
    */

    recordButton.disabled = true;
    stopButton.disabled = false;
    pauseButton.disabled = false;
    recordButton.innerHTML = "REC...";


    /*
        We're using the standard promise based getUserMedia() 
        https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
    */

    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

        /*
            create an audio context after getUserMedia is called
            sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
            the sampleRate defaults to the one set in your OS for your playback device

        */
        audioContext = new AudioContext();

        //update the format 
        //document.getElementById("formats").innerHTML = "Format: 1 channel pcm @ " + audioContext.sampleRate / 1000 + "kHz"

        /*  assign to gumStream for later use  */
        gumStream = stream;

        /* use the stream */
        input = audioContext.createMediaStreamSource(stream);

        /* 
            Create the Recorder object and configure to record mono sound (1 channel)
            Recording 2 channels  will double the file size
        */
        rec = new Recorder(input, { numChannels: 1 })

        //start the recording process
        rec.record()

        console.log("Recording started");

    }).catch(function (err) {
        //enable the record button if getUserMedia() fails
        recordButton.disabled = false;
        stopButton.disabled = true;
        pauseButton.disabled = true
    });
}

function pauseRecording() {
    console.log("pauseButton clicked rec.recording=", rec.recording);
    isPaused = !isPaused;
    if (rec.recording) {
        //pause
        rec.stop();
        recordButton.innerHTML = "PAUSED";
        pauseButton.innerHTML = "<ion-icon class='mic-icon' name='mic-off'></ion-icon>";
    } else {
        //resume
        rec.record()
        recordButton.innerHTML = "REC...";
        pauseButton.innerHTML = "<ion-icon class='mic-icon' name='mic'></ion-icon>";

    }
}

function stopRecording() {
    console.log("stopButton clicked");

    //disable the stop button, enable the record too allow for new recordings
    stopButton.disabled = true;
    pauseButton.disabled = true;

    //tell the recorder to stop the recording
    rec.stop();

    //stop microphone access
    gumStream.getAudioTracks()[0].stop();

    //create the wav blob and pass it on to createDownloadLink
    rec.exportWAV(createDownloadLink);
}

function incrementDuration() {
    let secondsLabel = '0';
    let minutesLabel = '0';
    if (!isPaused) {
        totalSeconds++;
        if (totalSeconds % 60 < 10) {
            secondsLabel = `0${totalSeconds % 60}`;
        } else {
            secondsLabel = `${totalSeconds % 60}`
        }
        if (Math.floor(totalSeconds / 60) < 10) {
            minutesLabel = `0${Math.floor(totalSeconds / 60) / 10}`;
        } else {
            minutesLabel = `${Math.floor(totalSeconds / 60) < 10}`
        }
        durationLabel.innerHTML = `<h2>${minutesLabel}:${secondsLabel}</h2>`;
    }
}

function createDownloadLink(blob) {

    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    var link = document.createElement('a');

    //name of .wav file to use during upload and download (without extendion)
    var filename = "iPresent_YourRecording";

    //add controls to the <audio> element
    au.controls = true;
    au.src = url;

    //save to disk link
    link.href = url;
    link.download = filename + ".wav"; //download forces the browser to download the file using the filename
    link.click();

    var next = document.createElement('a');
    next.href = "/upload";
    next.click();
}