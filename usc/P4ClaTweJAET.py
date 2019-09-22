# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 12:42:58 2015

@author: adane_000
"""
import tweepy
import os.path
import pickle
import time
import re
import sys

"""Definir los argumentos que acepta el programa y valores predeterminados."""
import argparse, textwrap
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
         description=textwrap.dedent('''\
...  Clasificador de tweets
...  -----------------------------------------------------------------------------
...  Este es un script python que extrae contenidos de dos perfiles en Twitter, 
...  se procesan los tweets extraidos y se realiza experimentos básicos de 
...  clasificación y análisis de sentimentos.
...  Ejemplos de ejecución.    
...   1:Extraer 2200  tweets de dos figuras en camapaña politica en USA para 
...     las elecciones 2016, genera y compara modelos de clasificación también
...     compara la polaridad de tweets de ambos politicos.
...        P4ClaTweJAET.py -t1 hillaryclinton -t2 realdonaldtrump -t 2200
...   2:Extraer 2800 tweets de dos figuras politicas en USA, una en camapaña y 
...      la segunda ejerciendo su cargo. Se compara el comportamiento.
...        P4ClaTweJAET.py -t1 BarackObama -t2 realdonaldtrump -t 2800
...   Valores por defecto:
...        -t1 BarackObama -t2 realdonaldtrump -t 2800
...   El programa acepta los siguientes parametros de configuración.
...         '''))
parser.add_argument('-t1','--target1', help='Perfil 1 de twitter',required=False, default='hillaryclinton')
parser.add_argument('-t2','--target2', help='Perfil 2 de twitter',required=False, default='realdonaldtrump')
parser.add_argument('-t','--tweets',help='Número de tweets a extraer', required=False, default='2200')
args = parser.parse_args()

WINDOW_RATE_MIN = 16
SLEEP_TIME_SEC = 60
TWEETS_EXTRACT = int(args.tweets)
targets = {}
targets['t1'] = {}
targets['t1']["id"] = args.target1
targets['t1']["nameFile"] = targets['t1']["id"] + str(TWEETS_EXTRACT) + '.pickle'
#targets['t1']["nameFile"] = targets['t1']["id"] + '_40.pickle'
targets['t2'] = {}
targets['t2']["id"] =args.target2
targets['t2']["nameFile"] = targets['t2']["id"] + str(TWEETS_EXTRACT) + '.pickle'
#targets['t2']["nameFile"] = targets['t2']["id"] + '_40.pickle'
import authTweepy as at
auth = tweepy.OAuthHandler(at.consumer_key, at.consumer_secret)
auth.set_access_token(at.access_token, at.access_token_secret)
# Get the User object for twitter...
api = tweepy.API(auth)
def getDateString(dt):
    """Obtener el string de un datetime para la fecha de creacion del tweet"""
    stringDateTime = str(dt.date()) + ' ' + str(dt.time())
    return stringDateTime

def extractTweetsfromTarget(t):
    """Extraer tweets de la persona de interés"""
    dtweet = {}
    i = 0
    tweetsPerRequest = 200
    MaxIdTweet = tweepy.Cursor(api.user_timeline, id=t["id"],include_rts=False,exclude_replies=True).items(1).next().id
    rateLimit = False
    while TWEETS_EXTRACT > len(dtweet):
        for tweet in tweepy.Cursor(api.user_timeline, id=t["id"],max_id=MaxIdTweet, include_rts=False,exclude_replies=True).items(tweetsPerRequest):
            dtweet[i] = {}
            dtweet[i]['ID'] = tweet.id
            dtweet[i]['text'] = tweet.text #.encode('utf-8')
            dtweet[i]['created_at'] = getDateString(tweet.created_at)
            i = i + 1
        MaxIdTweet = dtweet[i-1]['ID']
        MaxIdTweet = MaxIdTweet - 1
        print("   Sleep: ", str(len(dtweet)), " / ", str(TWEETS_EXTRACT))
        if(len(dtweet) >= 2600 and rateLimit == False ):
            print("   Sleep 15 min:")
            rateLimit = True
            minute = 0
            while minute < WINDOW_RATE_MIN:
                sys.stdout.write('.')
                time.sleep(60)
                minute = minute + 1
        else:
            time.sleep(SLEEP_TIME_SEC)
    print("   Wake up...")
    #guardar la estructura de datos en binario
    dicStore = {k: dtweet[k] for k in range(0,TWEETS_EXTRACT)}
    pickle.dump(dicStore, open(t['nameFile'], 'wb'))
    print("   Se ha almacenado un archivo binario con los tweets de ", t["id"])

"""*******Cargar o extraer tweets de cada figura**********************"""
keys_tar = ['t1','t2']
first = True
for key in keys_tar:
    target = targets[key]
    if os.path.isfile(target["nameFile"]):
        print("Reutilizar: " + target["nameFile"])
    else:
        print("Nueva extracción: " + target["nameFile"])
        extractTweetsfromTarget(target)
        minute = 0
        if first == True:
            first = False
            while minute < WINDOW_RATE_MIN:
                sys.stdout.write('.')
                time.sleep(60)
                minute = minute + 1
    target['tweetsDic'] = pickle.load(open(target["nameFile"], 'rb'))
    target['tweetsText'] = [target['tweetsDic'].get(k).get('text') for k in target['tweetsDic'].keys()]
    #**********Preprocesamiento Eliminar URL's *******************
    target['tweetsText'] = [re.sub(r"http\S+", "", tweet) for tweet in target['tweetsText']]



"""*******Separar los tweets de entrenamiento y de test********************"""
from sklearn import cross_validation
import numpy as np
mergedTweets = targets['t1']['tweetsText'] + targets['t2']['tweetsText']
y = np.zeros(TWEETS_EXTRACT * 2, dtype=np.int64)
y[0:TWEETS_EXTRACT] = 1
y[-TWEETS_EXTRACT:] = 2
mergedTweetsTrain, mergedTweetsTest, yTrain, yTest = cross_validation.train_test_split(mergedTweets, y, test_size=0.2, random_state=0)
print ("yTrain", yTrain.shape, "  yTest", yTest.shape)

"""*******stopwords********************************************************"""
import codecs
def getStopWords():
    fStopWords = 'stopwords-en.txt'
    f = codecs.open(fStopWords, "r", "utf-8")
    lsStopWords = [line.rstrip('\n') for line in f]
    f.close()
    return lsStopWords
lsStopWords = getStopWords()

"""*******Twitter-aware tokenizer ******************************************"""
#Requiere NLTK http://www.nltk.org/install.html
#The Natural Language Toolkit (NLTK) is a Python package for natural language processing
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

"""**********CountVectorizer {bag of words} ********************************"""
from sklearn.feature_extraction.text import CountVectorizer
analisis = {}
vectorizer = CountVectorizer(min_df=10,stop_words=lsStopWords, tokenizer=tknzr.tokenize,
                             decode_error='ignore',ngram_range=(1,1))
X = vectorizer.fit_transform(mergedTweetsTrain)
mxTrainTweetsTokens = X.toarray()
analisis['palabrasCorpus'] = np.sum(mxTrainTweetsTokens)
features = vectorizer.get_feature_names()
analisis['palabrasDistintasCorpus'] = len(features)
#Actualizar los parámetros de vectorizador dado que los tweets de test son menos
vectorizer.set_params(vocabulary = features, min_df=2)
"""********TfIdf Train******************************************************"""
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False, sublinear_tf=False, use_idf=True)
tfidf = transformer.fit_transform(mxTrainTweetsTokens)
tfidfTrain = tfidf.toarray()
"""********CountVectorizer & TfIdf Test*************************************"""
#A partir del vectorizer generado con los 
#CountVectorizer Test
mxTestTweetsTokens = vectorizer.fit_transform(mergedTweetsTest).toarray()
#tweets de entrenamiento generar X_test
tfidfTest = transformer.fit_transform(mxTestTweetsTokens).toarray()


"""******** Clasificación***************************************************"""
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
""" Amplio rango de valores de el parametro de regularización C de SVCs"""
Cstep1 = np.arange(0.001,0.01,0.004)
Cstep2 = np.arange(0.01,0.1,0.04)
Cstep3 = np.arange(0.1,1.01,0.4)
Cs = np.concatenate((Cstep1, Cstep2, Cstep3), axis=0)
hyperParams = {'C': Cs}

"""Entrenar con validación 10 para elegir el mejor C para cada kernel"""
from sklearn.grid_search import GridSearchCV
from sklearn import svm
import zipfile

#En caso de que el archivo con los mejores modelos este en zip
bestModelsFileZip = "bestModels" + str(TWEETS_EXTRACT) + ".zip"
if os.path.isfile(bestModelsFileZip):
    print("Descomprimir: " + bestModelsFileZip)
    with zipfile.ZipFile(bestModelsFileZip, "r") as z:
        z.extractall()
models = {}
bestModelsFile = "bestModels" + str(TWEETS_EXTRACT) + ".pickle"
if os.path.isfile(bestModelsFile):
    print("Reutilizar: " + bestModelsFile)
    models = pickle.load(open(bestModelsFile, 'rb'))
    mCV_linear = models["mCV_linear"]
    mCV_rbf = models["mCV_rbf"]
    mCV_poly = models["mCV_poly"]
    tfidfTrain = models["tfidfTrain"]
    yTrain = models["yTrain"]
    tfidfTest = models["tfidfTest"]
    yTest = models["yTest"]
    targets = models["targets"]
else:
    mCV_linear = GridSearchCV(svm.SVC(kernel='linear'), 
                           hyperParams, cv=10, scoring='accuracy')
    mCV_linear.fit(tfidfTrain, yTrain)
    mCV_rbf = GridSearchCV(svm.SVC(kernel='rbf', gamma=0.7), 
                           hyperParams, cv=10, scoring='accuracy')
    mCV_rbf.fit(tfidfTrain, yTrain)
    mCV_poly = GridSearchCV(svm.SVC(kernel='poly', degree=3), 
                           hyperParams, cv=10, scoring='accuracy')
    mCV_poly.fit(tfidfTrain, yTrain)
    models["mCV_linear"] = mCV_linear
    models["mCV_rbf"] = mCV_rbf
    models["mCV_poly"] = mCV_poly
    models["tfidfTrain"] = tfidfTrain
    models["yTrain"] = yTrain
    models["tfidfTest"] = tfidfTest
    models["yTest"] = yTest
    models["targets"] = targets
    pickle.dump(models, open(bestModelsFile, 'wb'))
    print("   Se ha almacenado un archivo binario ", bestModelsFile)

print ("Best hyperparameters svm.SVC(kernel='linear'): ", mCV_linear.best_params_)
print ("Best hyperparameters svm.SVC(kernel='rbf', gamma=0.7): ", mCV_rbf.best_params_)
print ("Best hyperparameters svm.SVC(kernel='poly',degree=3): ", mCV_poly.best_params_)

"""******** Matriz de Confusión*********************************************"""
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    global targets
    targets['t1']["id"]
    target_names = [targets['t1']["id"],targets['t2']["id"]]    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
def clasificacion(modelo, StrKernel):
    global tfidfTest, yTest
    print ("************------",StrKernel,"------************")
    y_pred = modelo.predict(tfidfTest)
    print (  "Precisión: ", accuracy_score(yTest, y_pred))
    # Compute confusion matrix
    cm = confusion_matrix(yTest, y_pred)
    np.set_printoptions(precision=2)
    print('Matriz de confusión sin normalización')
    print(cm)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    titulo = 'Matriz de confusión normalizada '
    print(titulo)
    print(cm_normalized)
    titulo = titulo + StrKernel
    plt.figure()
    plot_confusion_matrix(cm_normalized, title=titulo)
    plt.show()
clasificacion(mCV_linear, StrKernel = 'SVC linear')
clasificacion(mCV_rbf, StrKernel = 'SVC rbf')
clasificacion(mCV_poly, StrKernel = 'SVC poly')



 #probar con distintos valores de C
 #y  hacer el analisis básico de sentimientos
# https://www.quora.com/I-have-a-list-of-positive-and-negative-words-How-do-I-proceed-to-do-a-sentiment-analysis-of-Tweets-on-Python-using-the-said-list


"""******** Análisis básico de sentimientos*********************************"""
from collections import Counter
def readwords( filename ):
    f = open(filename)
    words = [ line.rstrip() for line in f.readlines()]
    return words
positive = readwords('positive-words.txt')
negative = readwords('negative-words.txt')
def polarity(tweet):
    tokens = tknzr.tokenize(tweet)
    count = Counter(tokens)
    pos = 0
    neg = 0
    points = 0
    for key, val in count.items():
        key = key.rstrip('.,?!\n') # removing possible punctuation signs
        if key in positive:
            pos += val
        if key in negative:
            neg += val
    points = pos - neg
    if points > 0:
        return '+'
    elif points < 0:
        return '-'
    return '0'

keys_tar = ['t1','t2']
first = True
for key in keys_tar:
    target = targets[key]
    target['polarity'] = []
    for tweet in target['tweetsText']:
        target['polarity'].append(polarity(tweet))
#Tabla de frecuencias por persona
hT1 = Counter(targets['t1']['polarity'])
#Tabla de frecuencias por persona
hT2 = Counter(targets['t2']['polarity'])
positives = [hT1['+'],hT2['+']]
negatives = [hT1['-'],hT2['-']]
neutrals = [hT1['0'],hT2['0']]
N = 2
ind = np.arange(N)  # the x locations for the groups
width = 0.25       # the width of the bars
fig, ax = plt.subplots()

rects1 = ax.bar(ind, negatives, width, color='r')
rects2 = ax.bar(ind + width, neutrals, width, color='b')
rects3 = ax.bar(ind + width + width, positives, width, color='g')

ax.set_ylabel('Count')
ax.set_title('Basic Sentiment Analysis')
ax.set_xticks(ind + width)
ax.set_xticklabels((targets['t1']["id"], targets['t2']["id"]))

ax.legend((rects1[0], rects2[0],rects3[0]), ('Negatives', 'Neutrals','Positives'),bbox_to_anchor=(1.3,0.9))
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
plt.show()

