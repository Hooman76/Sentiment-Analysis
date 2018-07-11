# Sentiment-Analysis
This project aims to analyze sentences and state them as positive or negative.

pre-requirements:
  1. nltk 3.33
  2. you need to download data for nltk, you can do this by: nltk.download('sentiwordnet'), nltk.download('stopwords'),
     nltk.download('wordnet')
  3. spacy
  4. matplotlib
  
To run:
  run sentiment_analyzer.py
  
Data used for labeled words:
  sentiwordnet
  
Data used for testing:
  1. movie reviews 
  2. Obama's Tweets during 2012-2017
  
sentiwordnet includes positive, negative and objective score and another column which indicates a word's state ('n' = noun,
'v' = verb, 'a' = adjective and 'r' = adverb)
For analyzing whether a tweet is positive or negative, we tokenize the sentence and check each word's role in the sentence
(noun, verb, ...) in order to check its score for that specific role in the sentence then we add the tokens' pos_score, neg_score
and obj_score and call and then divide them by the number of tokens included in that tweet or sentence. We obtain the overall_score 
by subtracing neg_score from pos_score, the result if greater than zero will be indicated as positive and if less than zero will
be called negative.

The plot shows the overall score of Obama's tweets in each year which means how much positive his tweets are in that specific year.

The algorithms precision and recall is obtained from the movie reviews dataset because Obama's Tweets didn't have any sentiment label
but the movie reviews include a sentiment for each review so that we can calculate precision and recall.

Model Performance metrics:
------------------------------
Accuracy: 0.597
Precision: 0.6699
Recall: 0.597
F1 Score: 0.5481

Prediction Confusion Matrix:
------------------------------
                 Predicted:         
                   positive negative
Actual: positive       6948      562
        negative       5483     2007

