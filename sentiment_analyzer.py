import pandas as pd
import numpy as np
import text_normalizer as tn
from nltk.corpus import sentiwordnet as swn
import matplotlib.pyplot as plt
import model_evaluation_utils as meu

data_set = pd.read_csv(r'obama_tweets.csv')
data_set_senti = pd.read_csv(r'movie_reviews.csv')

tweets = np.array(data_set['Text'])
dates = np.array(data_set['Date'])


reviews = np.array(data_set_senti['review'])
sentiments = np.array(data_set_senti['sentiment'])

test_reviews = reviews[35000:]
test_sentiments = sentiments[35000:]
sample_review_ids = [7626, 3533, 13010]
# Creating data for test
test_tweets = tweets[:]
test_dates = dates[:]
print(test_dates)
sample_ids = []
for i in range(len(test_tweets)):
    sample_ids.append(i)


def analyze_sentiment_sentiwordnet_lexicon(review, verbose=False):
    # tokenize and POS tag text tokens
    tagged_text = [(token.text, token.tag_) for token in tn.nlp(review)]
    pos_score = neg_score = token_count = obj_score = 0
    # get wordnet synsets based on POS tags
    # get sentiment scores if synsets are found
    for word, tag in tagged_text:
        ss_set = None
        if 'NN' in tag and list(swn.senti_synsets(word, 'n')):
            ss_set = list(swn.senti_synsets(word, 'n'))[0]
        elif 'VB' in tag and list(swn.senti_synsets(word, 'v')):
            ss_set = list(swn.senti_synsets(word, 'v'))[0]
        elif 'JJ' in tag and list(swn.senti_synsets(word, 'a')):
            ss_set = list(swn.senti_synsets(word, 'a'))[0]
        elif 'RB' in tag and list(swn.senti_synsets(word, 'r')):
            ss_set = list(swn.senti_synsets(word, 'r'))[0]
        # if senti-synset is found
        if ss_set:
            # add scores for all found synsets
            pos_score += ss_set.pos_score()
            neg_score += ss_set.neg_score()
            obj_score += ss_set.obj_score()
            token_count += 1

    # aggregate final scores
    final_score = pos_score - neg_score
    if token_count == 0:
        token_count = 1
    norm_final_score = round(float(final_score) / token_count, 2)
    final_sentiment = 'positive' if norm_final_score >= 0 else 'negative'
    result = [norm_final_score, final_sentiment]
    if verbose:
        norm_obj_score = round(float(obj_score) / token_count, 2)
        norm_pos_score = round(float(pos_score) / token_count, 2)
        norm_neg_score = round(float(neg_score) / token_count, 2)
        # to display results in a nice table
        sentiment_frame = pd.DataFrame([[final_sentiment, norm_obj_score, norm_pos_score,
                                         norm_neg_score, norm_final_score]],
                                       columns=pd.MultiIndex(levels=[['SENTIMENT STATS:'],
                                                                     ['Predicted Sentiment', 'Objectivity',
                                                                      'Positive', 'Negative', 'Overall']],
                                                             labels=[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]]))
        print(sentiment_frame)

    return result


overall_2012 = overall_2013 = overall_2014 = overall_2015 = overall_2016 = overall_2017 = 0
for tweet, date in zip(test_tweets[sample_ids], test_dates[sample_ids]):
    print('TWEET:', tweet)
    # print('Actual Sentiment:', sentiment)
    pred = analyze_sentiment_sentiwordnet_lexicon(tweet, verbose=True)
    if date[:4] == "2017":
        overall_2017 += pred[0]
    if date[:4] == "2016":
        overall_2016 += pred[0]
    if date[:4] == "2015":
        overall_2015 += pred[0]
    if date[:4] == "2014":
        overall_2014 += pred[0]
    if date[:4] == "2013":
        overall_2013 += pred[0]
    if date[:4] == "2012":
        overall_2012 += pred[0]
    print('-'*60)

predicted_sentiments = [analyze_sentiment_sentiwordnet_lexicon(review, verbose=False)[1] for review in test_reviews]

meu.display_model_performance_metrics(true_labels=test_sentiments, predicted_labels=predicted_sentiments,
                                  classes=['positive', 'negative'])


# Plotting using the calculated overalls
years = ('2012', '2013', '2014', '2015', '2016', '2017')
x_pos = np.arange(len(years))
overall = [overall_2012, overall_2013, overall_2014, overall_2015, overall_2016, overall_2017]
plt.bar(x_pos, overall, align='center', color='b')
plt.xticks(x_pos, years)
plt.ylabel('Overall')
plt.title("Obama's Tweets during 2012-2017")

plt.show()

# normalize dataset
norm_test_reviews = tn.normalize_corpus(test_tweets)
