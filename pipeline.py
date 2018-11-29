import numpy
import sklearn

### in this basic implementation, restricting number of entries to 500
### to speed processing times.

"""
create array of training text entries
"""

# open the training_text file and read its contents to a string
with open('training_text.txt', 'r') as trainingFile:
    trainingText = trainingFile.read()

# split the string on the pipe character
trainingText = trainingText.split('||')

# strip whitespace characters
trainingText = [x.strip() for x in trainingText]

# limit to the first 200 entries
trainingData = trainingText

"""
create 2d array of sample, features
"""

# create a CountVectorizer object

from sklearn.feature_extraction.text import CountVectorizer

countVectorizer = CountVectorizer(stop_words='english')

# use countVectorizer to create 2d array of text features

trainingX = countVectorizer.fit_transform(trainingData)

"""
correct for text length and word frequency
"""

from sklearn.feature_extraction.text import TfidfTransformer
tfidfTransformer = TfidfTransformer()

trainingXCorrected = tfidfTransformer.fit_transform(trainingX)

"""
create array of targets
"""

# open training variant file and read lines
with open('training_variants.txt', 'r') as targetFile:
    targetText = targetFile.readlines()

# append correct column of each line to target array

targets = []

for line in targetText:
    splitLine = line.strip().split(',')
    targets.append(splitLine[3])

sampleTargets = targets

"""
train classifier on sample,features v. targets
"""

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB().fit(trainingXCorrected,sampleTargets)

"""
test the classifier's ability to...classify
"""

# open test data
with open('test_text.txt', 'r') as f:
    testData = f.read()

# vectorize text data
testData = testData.split("||")
testData = [x.strip() for x in testData]
testData = testData[1:] # get rid of header line

testXCounts = countVectorizer.transform(testData)
testXCorrected = tfidfTransformer.transform(testXCounts)

# open test target data
with open('training_variants.txt','r') as f:
    testTargets = f.readlines()

testTargets = [line.strip().split(',')[3] for line in testTargets]

# predicted test values
predicted = classifier.predict(testXCorrected)

# grade performance
score = 0

for i in range(len(predicted)):
    if predicted[i] == testTargets[i]:
        score += 1

# report results
print("score is %d of %d for %f correct" % (score,len(predicted),float(score) / float(len(predicted))))
print("random would have achieved %d of %d correct" % (len(predicted) * (float(1) / float(9)), len(predicted)))
