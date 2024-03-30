from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask import Flask, render_template, request, redirect, url_for
import string

app = Flask(__name__)

# Initialize variables to store sentiment analysis results
neg_percent = None
pos_percent = None
neu_percent = None

@app.route('/', methods=['GET', 'POST'])
def home():
    global neg_percent, pos_percent, neu_percent
    if request.method == 'POST':
        text = request.form.get('inputtext')
        lower_case = text.lower()
        cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
        tokenized_words = word_tokenize(cleaned_text)
        final_words = [word for word in tokenized_words if word not in stopwords.words('english')]
        
        # Join the words back into a single string before passing to the analyzer
        cleaned_text = ' '.join(final_words)
        
        # Analyze sentiment
        score = SentimentIntensityAnalyzer().polarity_scores(cleaned_text)
        neg = score['neg']
        pos = score['pos']
        neu = score['neu']

        # Calculate percentages
        total = neg + pos + neu
        if total == 0:
            neg_percent = 0
            pos_percent = 0
            neu_percent = 0
        else:
            neg_percent = (neg / total) * 100
            pos_percent = (pos / total) * 100
            neu_percent = (neu / total) * 100
        
        return render_template('index.html', neg_percent=neg_percent, pos_percent=pos_percent, neu_percent=neu_percent)

    # Clear results if requested
    elif request.method == 'GET' and request.args.get('clear'):
        neg_percent = None
        pos_percent = None
        neu_percent = None
        return redirect(url_for('home'))

    return render_template('index.html', neg_percent=neg_percent, pos_percent=pos_percent, neu_percent=neu_percent)

if __name__ == '__main__':
    app.run(debug=True)
