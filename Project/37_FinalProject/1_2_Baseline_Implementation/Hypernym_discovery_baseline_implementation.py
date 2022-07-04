# !/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Embedding, Bidirectional, LSTM, TimeDistributed, Flatten, GRU
from keras.models import load_model
from keras.models import Model, Sequential
import pickle


def hypernym_discovery_baseline(task, model="gru"):
    print("Entering parent method")
    if task == "1A":
        dataset = "English"
        english_embeddings_file = open("embeddings/medical.txt", "r")
        embeddings = english_embeddings_file.read().splitlines()
        training_set = "dir-data/training/data/1A.english.training.data.txt"
        training_hypernym_set = "dir-data/training/gold/1A.english.training.gold.txt"
        testing_set = "dir-data/test/data/1A.english.test.data.txt"
        testing_hypernym_set = "dir-data/test/gold/1A.english.test.gold.txt"
        validation_set = "dir-data/trial/data/1A.english.trial.data.txt"
        validation_hypernym_set = "dir-data/trial/gold/1A.english.trial.gold.txt"
        if model == "gru":
            hypernyms_to_be_saved_for_trainset_data = "/home2/nayan.anand/nlp_project/train_predicted_hypernym_gru_english.txt"
            hypernyms_to_be_saved_for_testset_data = "/home2/nayan.anand/nlp_project/test_predicted_hypernym_gru_english.txt"
        else:
            hypernyms_to_be_saved_for_trainset_data = "/home2/nayan.anand/nlp_project/train_predicted_hypernym_lstm_english.txt"
            hypernyms_to_be_saved_for_testset_data = "/home2/nayan.anand/nlp_project/test_predicted_hypernym_lstm_english.txt"

    elif task == "2A":
        dataset = "Medical"
        medical_embeddings_file = open("embeddings/medical.txt", "r")
        embeddings = medical_embeddings_file.read().splitlines()
        training_set = "dir-data/training/data/2A.medical.training.data.txt"
        training_hypernym_set = "dir-data/training/gold/2A.medical.training.gold.txt"
        testing_set = "dir-data/test/data/2A.medical.test.data.txt"
        testing_hypernym_set = "dir-data/test/gold/2A.medical.test.gold.txt"
        validation_set = "dir-data/trial/data/2A.medical.trial.data.txt"
        validation_hypernym_set = "dir-data/trial/gold/2A.medical.trial.gold.txt"
        if model == "gru":
            hypernyms_to_be_saved_for_trainset_data = "/home2/nayan.anand/nlp_project/train_predicted_hypernym_gru_medical.txt"
            hypernyms_to_be_saved_for_testset_data = "/home2/nayan.anand/nlp_project/test_predicted_hypernym_gru_medical.txt"
        else:
            hypernyms_to_be_saved_for_trainset_data = "/home2/nayan.anand/nlp_project/train_predicted_hypernym_lstm_medical.txt"
            hypernyms_to_be_saved_for_testset_data = "/home2/nayan.anand/nlp_project/test_predicted_hypernym_lstm_medical.txt"


    elif task == "2B":
        print("Entering subtask 1c")
        dataset = "Music"
        music_embeddings_file = open("embeddings/music.txt", "r")
        embeddings = music_embeddings_file.read().splitlines()
        training_set = "dir-data/training/data/2B.music.training.data.txt"
        training_hypernym_set = "dir-data/training/gold/2B.music.training.gold.txt"
        testing_set = "dir-data/test/data/2B.music.test.data.txt"
        testing_hypernym_set = "dir-data/test/gold/2B.music.test.gold.txt"
        validation_set = "dir-data/trial/data/2B.music.trial.data.txt"
        validation_hypernym_set = "dir-data/trial/gold/2B.music.trial.gold.txt"
        if model == "gru":
            hypernyms_to_be_saved_for_trainset_data = "/home2/nayan.anand/nlp_project/train_predicted_hypernym_gru_music.txt"
            hypernyms_to_be_saved_for_testset_data = "/home2/nayan.anand/nlp_project/test_predicted_hypernym_gru_music.txt"
        else:
            hypernyms_to_be_saved_for_trainset_data = "/home2/nayan.anand/nlp_project/train_predicted_hypernym_lstm_music.txt"
            hypernyms_to_be_saved_for_testset_data = "/home2/nayan.anand/nlp_project/test_predicted_hypernym_lstm_music.txt"

    else:
        print("Invalid choice")
        return 0
    if model != "gru" and model != "lstm":
        print("Invalid model choice")
        return 0
    vocab_size = len(embeddings)
    # creating embedding_matrix
    print("creating embedding matrix")
    embedding_matrix = np.zeros((vocab_size, 300))  # as the size of embedding vector is 300
    counter = 1
    word_vocab = []
    # inserting the unk tag as the first word of vocab
    word_vocab.append("UNK")
    # inserting a random array of length 300 as embedding for 'UNK' tag
    embedding_matrix[0] = np.random.random(300)

    for word_embed in embeddings[1:]:
        # word = word_embed.split
        word_vocab.append(word_embed.split()[0])  # getting the word
        temp_embedding = word_embed.strip().split(' ')[1:]  # here all the numbers are strings
        temp_array = np.zeros(shape=(
        1, 300))  # intitalizing a temproary array that shall contain embedding corresponding to a single word

        for i in range(len(temp_embedding)):  # converting the string array to float
            temp_array[0, i] = np.float(temp_embedding[i])
        embedding_matrix[counter] = temp_array[0]
        counter = counter + 1
    # getting the queryset data
    print("Getting the queryset")
    queryset_file = open(training_set, "r")
    queryset = queryset_file.readlines()

    queryset_test_file = open(testing_set, "r")
    queryset_test = queryset_test_file.readlines()

    queryset_validation_file = open(validation_set, "r")
    queryset_validation = queryset_validation_file.readlines()

    # getting rid of the unnecessary information for medicalcorpus
    print("cleaning queryset")
    for i in range(len(queryset)):  # as The embeddings were made by joining the space separate words
        queryset[i] = "_".join(queryset[i].split("\t")[0].split())

    for i in range(len(queryset_test)):
        queryset_test[i] = "_".join(queryset_test[i].split("\t")[0].split())

    for i in range(len(queryset_validation)):
        queryset_validation[i] = "_".join(queryset_validation[i].split("\t")[0].split())

    # reading the hypernyms in a list
    print("reading hypernyms")
    training_hypernym_file = open(training_hypernym_set, "r")
    training_hypernyms = training_hypernym_file.read().splitlines() 

    testing_hypernym_file = open(testing_hypernym_set, "r")
    testing_hypernyms = testing_hypernym_file.read().splitlines() 

    validation_hypernym_file = open(validation_hypernym_set, "r")
    validation_hypernyms = validation_hypernym_file.read().splitlines() 

    # generating the train, test and validationset
    print("Preparing dataset for model training")
    training_query_hypernym_pair, y_train = dataset_preparation(queryset, word_vocab, training_hypernyms,
                                                                embedding_matrix)
    testing_query_hypernym_pair, y_test = dataset_preparation(queryset_test, word_vocab, testing_hypernyms,
                                                              embedding_matrix)
    validation_query_hypernym_pair, y_validation = dataset_preparation(queryset_validation, word_vocab,
                                                                       validation_hypernyms, embedding_matrix)

    # training the model and receiving the trained model object
    Trained_model = model_training(training_query_hypernym_pair, y_train, testing_query_hypernym_pair, y_test,
                                   validation_query_hypernym_pair, y_validation, dataset, model)

    # predict_hypernyms
    final_total_hypernyms_predicted_trainset = predict_hypernyms(queryset, word_vocab, embedding_matrix, Trained_model,
                                                                 Trained_model)
    final_total_hypernyms_predicted_testset = predict_hypernyms(queryset_test, word_vocab, embedding_matrix,
                                                                Trained_model, Trained_model)
    # writing the hypernyms in a text file

    write_hypernyms(hypernyms_to_be_saved_for_trainset_data, final_total_hypernyms_predicted_trainset)
    write_hypernyms(hypernyms_to_be_saved_for_testset_data, final_total_hypernyms_predicted_testset)



def model_training(training_query_hypernym_pair, y_train, testing_query_hypernym_pair, y_test,
                   validation_query_hypernym_pair, y_validation, dataset, model_type):
    loss_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=False,
        label_smoothing=0.0,
        axis=-1,
        name='binary_crossentropy')

    adam_optimiser = tf.keras.optimizers.Adam(
        learning_rate=0.003,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name='Adam',
    )

    if model_type == "gru":
        model, history, evaluation_test_set_score = create_gru_model(loss_fn, adam_optimiser,
                                                                     training_query_hypernym_pair, y_train,
                                                                     testing_query_hypernym_pair, y_test,
                                                                     validation_query_hypernym_pair, y_validation)
    else:
        model, history, evaluation_test_set_score = create_lstm_model(loss_fn, adam_optimiser,
                                                                      training_query_hypernym_pair, y_train,
                                                                      testing_query_hypernym_pair, y_test,
                                                                      validation_query_hypernym_pair, y_validation)

    trainHistoryDict = model_type + "_" + dataset + "_train_history.txt"  # name of the file that contains model history
    model_evaluation_score = model_type + "_" + dataset + "_evaluation_score_on_testset.txt"
    with open(trainHistoryDict, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    with open(model_evaluation_score, 'wb') as file_evl:
        pickle.dump(evaluation_test_set_score, file_evl)

    model_saved_file_name = model_type + "_" + dataset + "_trained_model"
    model.save(model_saved_file_name)  # saving the trained model
    return model  # returning the model to the parent function



def write_hypernyms(hypernyms_to_be_saved, total_hypernyms_predicted):
    print("within write_hypernyms method")
    hyp_counter = 0
    for hyp in total_hypernyms_predicted:
        if len(hyp) > 15:
            counter_limit = 15  # as We want to select atmost top 15 hypernyms
        else:
            counter_limit = len(hyp)

        while counter_limit > 0:  # writing 15 hypernyms atmost in a single looping
            with open(hypernyms_to_be_saved, 'a') as f:
                f.writelines("%s\t" % hyp[hyp_counter][0])
                print(hyp[hyp_counter][0])
            f.close()
            hyp_counter = hyp_counter + 1
            counter_limit = counter_limit - 1
        with open(hypernyms_to_be_saved, 'a') as f:
            print("------------------")
            hyp_counter = 0
            f.writelines("\n")
        f.close()



# for the predition set I shall give the test_queryset * entire vocab (as Input) and I get the binary crossentopy scores as output
# if binary_cross entropy score > 0.5 I treat them as +ve cases else -ve cases
def predict_hypernyms(queryset, given_vocab, embedding_matrix, model, model_type="gru"):
    print("within predict_hypernyms method")
    vocab_size = len(given_vocab)
    prediction_array = np.zeros(
        (vocab_size - 1, 2, 300))  # as "UNK" token in not a potential hypernym to any word so medial_vocab_size -1
    total_hypernyms_predicted = []
    shortlisted_hypernyms = []
    for i in range(len(queryset)):
        print(i)
        if queryset[i] in given_vocab:
            index = given_vocab.index(queryset[i])  # getting the index of each queryset
            query_embedd = embedding_matrix[index]
        else:
            query_embedd = embedding_matrix[0]  # assigning the embedding of UNK

        for j in range(vocab_size - 1):
            prediction_array[j][0] = query_embedd
            prediction_array[j][1] = embedding_matrix[
                j + 1]  # as we are traversing vocab from index 1 : end ( so [j+1] )

        predicted_hypernyms = model.predict(prediction_array)

        for z in range(len(predicted_hypernyms)):
            if predicted_hypernyms[z][0] < 0.5:
                continue
            else:
                shortlisted_hypernyms.append([given_vocab[z], predicted_hypernyms[z][0]])
                # associate the UNK tag to all the word in the first place
        total_hypernyms_predicted.append(shortlisted_hypernyms)
        shortlisted_hypernyms = []
        prediction_array = np.zeros((vocab_size - 1, 2, 300))
    # sorting the total_hypernyms_predicted in descending order of binary crossentropy score to pick top "X" hypernyms of each word
    for i in range(len(total_hypernyms_predicted)):
        total_hypernyms_predicted[i] = sorted(total_hypernyms_predicted[i], key=lambda x: x[1], reverse=True)
    return total_hypernyms_predicted  # returning the sorted predicted hypernyms


def create_gru_model(loss_fn, adam_optimiser, training_query_hypernym_pair, y_train, testing_query_hypernym_pair,
                     y_test, validation_query_hypernym_pair, y_validation):
    print("within create_gru_model method")
    model = Sequential()
    model.add(Input(shape=(2, 300)))
    model.add(GRU(300, return_sequences=True, dropout=0.3))
    model.add(GRU(300, dropout=0.2))
    model.add(Flatten())
    model.add(Dense(300, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss_fn, optimizer=adam_optimiser, metrics=['accuracy'])
    history = model.fit(training_query_hypernym_pair, y_train, batch_size=32, epochs=10, verbose=2,
                        validation_data=(validation_query_hypernym_pair, y_validation))
    evaluation_test_set_score = model.evaluate(testing_query_hypernym_pair, y_test, batch_size=32)

    return model, history, evaluation_test_set_score



def create_lstm_model(loss_fn, adam_optimiser, training_query_hypernym_pair, y_train, testing_query_hypernym_pair,
                      y_test, validation_query_hypernym_pair, y_validation):
    # creating a LSTM model
    print("within create_lstm_model method")
    model = Sequential()
    model.add(Input(shape=(2, 300)))
    model.add(LSTM(300, return_sequences=True, dropout=0.3))
    model.add(LSTM(300, dropout=0.2))
    model.add(Flatten())
    model.add(Dense(300, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss_fn, optimizer=adam_optimiser, metrics=['accuracy'])
    history = model.fit(training_query_hypernym_pair, y_train, batch_size=32, epochs=10, verbose=2,
                        validation_data=(validation_query_hypernym_pair, y_validation))
    evaluation_test_set_score = model.evaluate(testing_query_hypernym_pair, y_test, batch_size=32)

    return model, history, evaluation_test_set_score



def dataset_preparation(queryset, word_vocab, hypernyms_tab_sep, embedding_matrix):
    print("within dataset preparation stage")
    # generating a list of query-hypernym pairs for medical corpus using embedding
    query_hypernym_training_embedding = np.zeros((1, 2, 300))
    y_label = np.array([])  # no label to begin with for y_label
    # one for the actual pair and 5 -ve samples per word
    counter = 0
    for i in range(len(queryset)):
        if queryset[i] in word_vocab:  # checking if the quertyset is present in the medical vocab
            index = word_vocab.index(queryset[i])  # getting the index of each queryset
            query_embedd = embedding_matrix[index]
            for hypernyms_found in hypernyms_tab_sep[i].split("\t"):  # as al the hypernyms are separated by a "\t"
                hypernyms_found = "_".join(hypernyms_found.split())
                if hypernyms_found in word_vocab:
                    hypernym_index = word_vocab.index(hypernyms_found)

                    hypernym_embedd = embedding_matrix[
                        hypernym_index]  # making a query hypernym pair for each of the query - hypernyms
                    vec_temp = np.vstack((query_embedd, hypernym_embedd)).reshape(1, 2, -1)
                    query_hypernym_training_embedding = np.concatenate((query_hypernym_training_embedding, vec_temp),
                                                                       axis=0)  # Merging
                    y_label = np.append(y_label, 1)  # inserting a positive label for each +ve hyponym-hypernym pair
                    # generating 5 -ve samples per correct pair
                    negative_samples_counter = 0
                    while negative_samples_counter < 5:  # 5 negative samples per positive sample
                        random_index = random.randint(0, len(word_vocab) - 1)
                        if random_index != index:
                            hypernym_embedd = embedding_matrix[random_index]
                            vec_temp = np.vstack((query_embedd, hypernym_embedd)).reshape(1, 2, -1)
                            query_hypernym_training_embedding = np.concatenate(
                                (query_hypernym_training_embedding, vec_temp), axis=0)  # Merging
                            y_label = np.append(y_label,
                                                0)  # inserting a negative label for each -ve hyponym-hypernym pair
                            counter = counter + 1
                            negative_samples_counter = negative_samples_counter + 1

                        else:
                            continue
                else:
                    print(hypernyms_found)
                    query_embedd = embedding_matrix[index]
                    hypernym_index = 0
                    print(counter)
                    hypernym_embedd = embedding_matrix[hypernym_index]
                    vec_temp = np.vstack((query_embedd, hypernym_embedd)).reshape(1, 2, -1)
                    query_hypernym_training_embedding = np.concatenate((query_hypernym_training_embedding, vec_temp),
                                                                       axis=0)  # Merging
                    y_label = np.append(y_label, 1)  # inserting a positive label for each +ve hyponym-hypernym pair
                    # generating 5 -ve samples per correct pair
                    negative_samples_counter = 0
                    while negative_samples_counter < 5:  # 5 negative samples per positive sample
                        random_index = random.randint(0, len(word_vocab) - 1)
                        if random_index != index:
                            hypernym_embedd = embedding_matrix[random_index]
                            vec_temp = np.vstack((query_embedd, hypernym_embedd)).reshape(1, 2, -1)
                            query_hypernym_training_embedding = np.concatenate(
                                (query_hypernym_training_embedding, vec_temp), axis=0)  # Merging
                            y_label = np.append(y_label,
                                                0)  # inserting a negative label for each -ve hyponym-hypernym pair
                            counter = counter + 1
                            negative_samples_counter = negative_samples_counter + 1

                        else:
                            continue

    # deleting the unpopulated cells of the array i.e the ones that contained zeros
    query_hypernym_training_embedding = query_hypernym_training_embedding[1:, :, :]
    # shuffling the query_hypernym_pair and labels randomly before training
    list_for_shuffling = list(zip(query_hypernym_training_embedding, y_label))
    random.shuffle(list_for_shuffling)
    training_query_hypernym_pair, y_train = zip(*list_for_shuffling)
    return query_hypernym_training_embedding, y_label


if __name__ == "__main__":
    os.chdir("/home2/nayan.anand/nlp_project")

    #Calling the python functions

    hypernym_discovery_baseline("1A", "gru")
    hypernym_discovery_baseline("1A", "lstm")
    hypernym_discovery_baseline("2A", "gru")
    hypernym_discovery_baseline("2A", "lstm")
    hypernym_discovery_baseline("2B", "gru")
    hypernym_discovery_baseline("2B", "lstm")
