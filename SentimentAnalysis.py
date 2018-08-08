import nltk
import numpy as np
import pandas as pd
import sklearn


from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


pathtrainingdata = r"Desktop\all\train.tsv"
pathtestingdata = r"Desktop\all\test.tsv"

rawtrain = pd.read_csv(pathtrainingdata, header =0, delimiter="\t", quoting=3)
rawtest = pd.read_csv(pathtestingdata, header =0, delimiter="\t", quoting=3)


traindata = rawtrain.fillna(0)
testdata = rawtest.fillna(0)



phrases = []
for x in range(0,len(traindata['Phrase'])):
    phrases.append(traindata['Phrase'][x])

classifier = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('dt', DecisionTreeClassifier())])
classifier = classifier.fit(phrases, traindata['Sentiment'])

phraseTest = []
for x in range(0,len(testdata['Phrase'])):
    phraseTest.append(testdata['Phrase'][x])

predictedData = classifier.predict(phraseTest)

with open(r"Desktop\all\sampleSubmission.csv", "w") as subm:
    subm.write("PhraseID, Sentiment\n")
    for x in range(0, len(testdata['Phrase'])):
        subm.write(str(testdata['PhraseId'][x]) + "," + str(predictedData[x])+"\n")
        
