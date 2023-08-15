import os
import glob
import pickle
from MyTraining import preprocessing, removeStopWords

# this function is basically a script for reading the emails from the test folder one by one, and performs preprocessing task
# it will return the email list after performing the preprocessing task
def getData(folder_name):
    email_collection = []  # store list of emails to be tested
    fileList = os.listdir(folder_name + os.sep)  # store list of files
    numberOfFiles = len(fileList)  # get length
    for i in range(1, numberOfFiles + 1):  # read files serially
        for file in glob.glob(folder_name + os.sep + "email" + str(i) + ".txt"):  # get that particular file path
            fileObject = open(file, "r+", encoding='utf-8', errors='ignore')  # open the file
            email = fileObject.read()  # read the file
            email = preprocessing(email)
            email = removeStopWords(email)  # remove stop words
            email_collection.append(email)  # add a email string to the list

    return email_collection

# this is the function for performing the prediction on test data
def prediction():
    x_test = getData('Test')  # get list of emails in the test folder
    filename_model = 'model.pkl'
    filename_count_vect = 'count_vectorizer.pkl'
    filename_tfidf = 'tfidf_transformer.pkl'

    if os.path.isfile(filename_model) and os.path.isfile(filename_count_vect) and os.path.isfile(filename_tfidf):
        # loading the models saved at the time of training
        svm_model = pickle.load(open(filename_model, 'rb'))  # load the SVM classifier
        count_vect_model = pickle.load(open(filename_count_vect, 'rb'))
        tfidf_model = pickle.load(open(filename_tfidf, 'rb'))  # load the feature extractor

        count_vec_transform = count_vect_model.transform(x_test)
        test = tfidf_model.transform(count_vec_transform)

        print(svm_model.predict(test))  # calling the prediction function

        # writing the predictions on the output file
        with open("output.txt", "w") as output:  # the output to a file
            output.write(str(svm_model.predict(test)))
    else:
        print('Some models are not saved properly !!')

prediction()
