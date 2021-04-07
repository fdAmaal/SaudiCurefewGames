# SaudiCurefewGames
an academic research for social computing course in King Saud University fall 2020.

Code cintains lines to read a csv file that contains Arabic tweets related to Gaming words to preproccessing and analysing them.

Before using the code, we need to some installation.

First, on MacOS

#  install python3
As mentioned in (https://docs.python-guide.org/starting/install3/osx/)
We used this command in intall python3 and pip3.
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# Twint Installation
Since the following command doesn't work with me,
pip3 install Twint

I used the second way that mentioned in (https://pypi.org/project/twint/) by ising this Command,
pip3 install --user --upgrade -e git+https://github.com/twintproject/twint.git@origin/master#egg=twint

# Collect Tweets using Twint
By determining the keyword using -s ,
start date using --since ,
end date using --until,
also to save the results info a file called curefewGames as csv file using --csv -o 

twint -s "'العاب الحجر'" --since 2020-03-09 --until 2020-06-21 --csv -o curefewGames.csv

However, when I tried looking for many keywords in the same time such as the following one:
twint -s "'العاب الحجر'" -s "'ألعاب الحجر'" -s "'من يلعب معي'" -s "'العاب اون لاين'" -s "'ألعاب أون لاين'" -s"'لعبة حلوة'" -s "'لعبة رهيبة'" -s "'لعبت مع'" -s "'لعبنا'" -s "'نلعب'" -s "'بلعب'" -s "'لعبة احبها'" -s "'العاب عجبتني'" -s "'لعبت لحالي'" --since 2020-03-09 --until 2020-06-21 --csv -o gamesKeywords.csv
However, I got a file with only the last word which means each keyword neads to be in seperate line.

# install pyarabic
pip install pyarabic

# Mazajak

    import requests
    import json
    '''
    This function offers the ability to predict the sentiment of a single sentence 
    through the API, the sentiment is one of three classes (positive negative, neutral)
    Input: 
            sentence(str): the input sentence of which the sentiment is to be predicted
    Output:
            prediction(str): the sentiment of the given sentence 
    '''
    
    def predict(sentence):
        url = "http://mazajak.inf.ed.ac.uk:8000/api/predict"
        to_sent = {'data': sentence}
        data = json.dumps(to_sent)
        headers = {'content-type': 'application/json'}
        # sending get request and saving the response as response object
        response = requests.post(url=url, data=data, headers=headers)

        prediction = json.loads(response.content)['data']

        return prediction



    '''
    This is an example to use the functions
    '''


      prediction = json.loads(response.content)['data']

      return prediction
      positive = 0
      negative = 0
      neutral = 0
      for tweet in Filtered_tweets['tweet']:
        #print(predict(tweet))
        if 'negative' == predict(tweet):
          negative += 1
        elif 'positive' == predict(tweet):
          positive += 1
        elif 'neutral' == predict(tweet):
          neutral += 1
          
      print('negative tweets =', negative, 'out of =', len(Filtered_tweets))
      print('positive tweets =', positive, 'out of =', len(Filtered_tweets))
      print('neutral tweets =', neutral, 'out of =', len(Filtered_tweets))


# Arabic Sentiment Analysis in tweets
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import string
    import re
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger') 
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix,accuracy_score, classification_report

    data =  pd.read_csv(r"alone2.csv")
    print(data.head())
    print(data.sample(5))


    '''
    The first step is to subject the data to preprocessing.
    This involves removing both arabic and english punctuation
    Normalizing different letter variants with one common letter
    '''
        # first we define a list of arabic and english punctiations that we want to get rid of in our text

    punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation

    # Arabic stop words with nltk
    stop_words = stopwords.words()

    arabic_diacritics = re.compile("""
                                 ّ    | # Shadda
                                 َ    | # Fatha
                                 ً    | # Tanwin Fath
                                 ُ    | # Damma
                                 ٌ    | # Tanwin Damm
                                 ِ    | # Kasra
                                 ٍ    | # Tanwin Kasr
                                 ْ    | # Sukun
                                 ـ     # Tatwil/Kashida
                             """, re.VERBOSE)

    def preprocess(text):
    
    '''
    text is an arabic string input
    
    the preprocessed text is returned
    '''
    
    #remove punctuations
    translator = str.maketrans('', '', punctuations)
    text = text.translate(translator)
    
    # remove Tashkeel
    text = re.sub(arabic_diacritics, '', text)
    
    #remove longation
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)

    text = ' '.join(word for word in text.split() if word not in stop_words)

    return text
  
    data['tweet'] = data['tweet'].apply(preprocess)
    print(data['tweet'].head(5))

# Additional Codes
# حذف الحركات بما فيها الشدة
    import pyarabic.araby as araby
    import pyarabic.number as number
    from pyarabic.araby import strip_tashkeel

    text = u"الْعَرَبِيّةُ"
    for row in Filtered_tweets['tweet']:
      print(strip_tashkeel(row))

# tokenize لتفريق النص إلى كلمات
    text = u"العربية لغة جميلة."
    for row in Filtered_tweets['tweet']:
      tokens = araby.tokenize(row)
      print(u",".join(tokens))
      
# remove tashkeel and filter out non-Arabic words:

    from pyarabic.araby import tokenize, is_arabicrange, strip_tashkeel
    text = u"ِاسمٌ الكلبِ في اللغةِ الإنجليزية Dog واسمُ الحمارِ Donky"
    for row in Filtered_tweets['tweet']:
      token= tokenize(row, conditions=is_arabicrange, morphs=strip_tashkeel)
      print(token)
