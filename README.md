# iPresent

![alt text](https://github.com/danielhariyanto/iPresent/blob/main/figures/main_page.PNG)

## Pitch Link
https://www.youtube.com/watch?v=QQAM0No8Ou4&feature=youtu.be

## About Us
* Daniel Hariyanto
* Meshach Adoe
* Sidney Jaury  
* Ekin Tiu | CS / AI @ Stanford University | https://www.linkedin.com/in/ekin-tiu-0aa467200/

## About iPresent
### Inspiration: 
As university students, presentations have become a hallmark of life. We spend countless hours perfecting our final speeches, anxiously awaiting the reactions and critiques of our audience. In that time, we practice speaking over and over again…to ourselves. 

Our predicament with remote learning doesn’t help. We scramble to pull up our PowerPoint decks in isolation, dreading the moment when it’s our turn to say, “Can you all see my screen?” 

At iPresent we aspire to *solve these problems.* 

Instead of practicing your speech in isolation, iPresent offers an anxiety-free alternative: *present to an AI software.* Not as intimidating now, right? Without having to present to your best friends, parents, or even 5-year old siblings, you can obtain high-quality, data-driven feedback to help you analyze your habits and improve. 

Our mission is to help our users, be it anxious college students or professional speakers, gain useful AI-powered insights about their presentation patterns, and ultimately provide suggestions on how to improve, based on real psychological research-backed metrics. 

### What it does: 
We wanted to design a platform that would lessen the ‘intimidation factor’ of public speaking, by providing **both** a real-time presentation simulator **and** a suite of algorithms to critically evaluate your presentation content (transcript), audio, and even facial expressions. 

You can use iPresent when you want to practice speaking in front of an audience before the main event, or you can use it to obtain a diverse range of presentation metrics and feedback. 

### How we built it: 
One major component of our application is the set of **algorithms,** both classical and machine learning, that use either the *audio, transcript, or video (next step)* as training modalities to compute useful presentation metrics. 

![alt text](https://github.com/danielhariyanto/iPresent/blob/main/figures/pipeline_new.PNG)

Our approach was to systematically research metrics that are indicative of high-quality presenters, and design a means to compute each of them algorithmically. The metrics can be broken down into the following categories: rating classification, passion, brevity, cadence, diction, diversity of language, and engagement. Below, we break down how we calculate each metric. 

#### **Rating Classification: TED Talk Model**

For most of us, when we think of speeches, TED Talks are our ‘go-to’. 

Our approach utilizes the voluminous, information rich TED Talk Dataset from Kaggle containing transcripts from past speeches in order to implement a **multi-label classification algorithm** to process transcripts and output ratings such as 'Beautiful', 'Confusing', 'Courageous', 'Funny', 'Informative', 'Ingenious', 'Inspiring', 'Longwinded', 'Unconvincing', 'Fascinating', 'Jaw-dropping', 'Persuasive', 'OK', 'Obnoxious'. 

![alt text](https://github.com/danielhariyanto/iPresent/blob/main/figures/TED.PNG)

The pipeline is defined as such: 
* For each transcript, use the Word2Vec algorithm (pre-trained model from gensim), to convert each word in the transcript to a **300-feature vector** . 
* We average these vectors in the axis of the number of words in order to get one 300-vector to represent the entire transcript. 
* We parse the dataset to obtain clean rating labels for each of our transcripts. We set the top-4 ratings as one, and the rest are zero. 
* Last but not least, we train the model using this engineered dataset, and evaluate on test data to obtain a top-k categorical accuracy of ~0.85. 

#### **Passion/Urgency:** Are you passionate about your words, or are you putting your audience to sleep?  

For this category, we develop algorithms to *analyze each of the data modalities.* 

We use Google Cloud’s Sentiment Analysis API to determine a sentiment score and magnitude for the entire transcript. 

We also train a **convolutional neural network** to classify snippets of audio as either neutral, or passionate/expressive. We used the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) dataset from Kaggle to classify the emotional intensity, or the lack thereof (monotony) of audio files by transforming audio wav files into spectrograms, which are then used to train two CNNs. 

![alt text](https://github.com/danielhariyanto/iPresent/blob/main/figures/audio.PNG)

Lastly, we used the facial expression dataset on Kaggle to train a CNN to classify neutral or expressive facial expressions. The algorithm is applied to the user’s mp4 file to help them improve their expression of enthusiasm for their speeches. 

![alt text](https://github.com/danielhariyanto/iPresent/blob/main/figures/facial.PNG)

### **Brevity:**

The question we try to answer here is, “How many unnecessary filler words or phrases are you using?” 

We compute a metric for brevity by creating two master lists containing sets of phrases or words that are commonly known in the english language as filler words. Using these lists, we iterate through the AI-generated **transcript,** and count the number of usages of each phrase. 

#### **Cadence: How fast are you speaking?**

We compute cadence by measuring the words / minute. Using the outputs from the Google Cloud Speech-to-Text API (abv. As GCST), we calculate both the number of words, and manually compute the duration of the audio file using GCST’s timestamps. 

We also calculate the number of pauses throughout the speech based on the per-word time outputs from the GCST. On average, we speak one word in 0.5 seconds. We compute the amount of time it takes for the user to say one word. If it’s greater than our predefined threshold, we classify it as a pause. 

#### **Diction: How comprehensible is your speech?**

We simply use GCST’s confidence metric for quantifying the interpretability of your speech. 

Focus/Engagement: Are you maintaining eye contact, even with your virtual audience? 
Our team uses the following GitHub repository (https://github.com/antoinelame/GazeTracking) as a starting point for our Gaze Tracking algorithm. After cloning this repo, we add more gaze classifications such as “looking up” and “looking down” and tweak some thresholds. 

We split the mp4 file into sizable chunks, each chunk an image that is fed into the GazeTracker. Thus, we obtain a gaze classification at each second in the presentation, so that the user can see how often their eyes drift away from the screen. 

### Challenges I ran into: 

On the algorithmic side, there were several challenges that our team faced. Our challenges can be categorized into the following: 

**Inability to detect filler words with Google Speech-to-Text API.**
Our team initially wanted to calculate the number of ‘umms’ or ‘hmms’ that the speaker used during the presentation, however to our dismay, we found that GCST filtered those phrases out automatically!

We had to discover an efficient solution, fast. After one of our team members uploaded an mp3 with multiple ‘umms’ or ‘hmms’, we discovered that these were factored in to the other words. Thus, our solution was to calculate the number of seconds spoken per each word, and if that time was greater than a certain threshold, we could classify that as a pause. 

**Finding the right datasets.**
We spent a lot of time scouring Kaggle and the internet to find the perfect datasets for each of our intended tasks. 

Luckily, we discovered a TED Talk dataset and noticed that it had a ratings feature which could be used as a label for our model. We had a particularly difficult time deciding upon an audio dataset, since audio is much rarer to find online than text-based datasets. 

**Unfavorable preliminary results for audio analysis.**
Before experimenting with spectrograms for audio decomposition, our team used the librosa library to extract a set of six features from each wav file. 

We ran this through a Dense neural network, but achieved low performance (around ~0.50). 

We then decided to analyze audio with a CNN, based on previous audio processing literature. This increased our performance substantially to an AUROC of ~0.79 on the binary classification task. 

### Accomplishments that we're proud of: 
* Trained three machine learning models (essentially three separate mini-projects) on three completely different modalities in such a short period of time, while having to familiarize ourselves with three different datasets. 
* Our models performed very well, overall. 
* Created a diverse suite of research-backed metrics that can be used to help our users analyze and improve their presentation skills
* Implemented these models and algorithms into a meticulously designed web-based platform. 
* Developed a visually appealing, interactive, and intuitive user interface to make the presentation simulations easy to use. 

### What we learned: 
* After this Hackathon, our team feels very comfortable dealing with different types of data, ranging from audio files to text data. During the hackathon, we had the opportunity to practice feature engineering, normalization, train-test-split, and other common data preprocessing techniques. 
* We learned about how audio files can be transformed into interpretable representations, and how certain features can be extracted from audio wav files. We then learned that we could use these representations to train a Convolutional Neural Network to perform our intended binary classification task of neutral vs. expressive. 
* We learned more about the Word2Vec algorithm, and how Natural Language Processing techniques can be used to represent large chunks of text such as TED Talks into compressed feature vector representations used to train a model with high performance. 
 
### What's next for iPresent: 

From a technical standpoint, we aim to make the following improvements to our algorithms and product: 

* Incorporate the video recording features to the stack, and run our video algorithms to compute the relevant metrics. 
* Apply a contrastive learning approach to both the audio and video machine learning algorithms in order to leverage unlabeled data to create pre-trained models before fine-tuning. (Our team actually downloaded an unlabeled dataset of faces to do this, but alas, since it’s a hackathon, we didn’t have the time to experiment).
( Use a temporal model, such as an LSTM, in addition to the CNN to process audio at different points in time. 
* Build an AI-based ‘umm’ classifier, or use another Speech-to-Text API to supplement GCST.
* Add more metrics over time. 
