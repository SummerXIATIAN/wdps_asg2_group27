import csv
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import time


def read_data(path):
    content_list = []
    score_list = []
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f)
        for data in reader:
            content = data[2]
            score = data[3]
            content_list.append(content)
            score_list.append(score)
    return content_list[1:], score_list[1:]


cont_list, sco_list = read_data('Data.csv')
for i in range(len(sco_list)):
    sco_list[i] = int(sco_list[i])

X_train, X_test, y_train, y_test = train_test_split(cont_list, sco_list, test_size=0.2, random_state=22)

vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(X_train, y_train)
t1 = time.time()
prediction_linear = classifier_linear.predict(X_test)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))

report = classification_report(y_test, prediction_linear, output_dict=True)

print('score 5: ', report['5'])
print('score 4: ', report['4'])
print('score 3: ', report['3'])
print('score 2: ', report['2'])
print('score 1: ', report['1'])