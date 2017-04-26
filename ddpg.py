import keras.preprocessing.text as kt
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json
import datamodule
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

OU = OU()       #Ornstein-Uhlenbeck Process

log_file = open('training_log.txt', 'w+')

def playGame(train_indicator=1):    #1 means Train, 0 means simply Run
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    GAMMA = 0.99
    TAU = 0.001     #Target Network HyperParameters
    LRA = 0.0001    #Learning rate for Actor
    LRC = 0.001     #Lerning rate for Critic

    action_dim = 1  # anomaly
    state_dim = 41  #input dimension

    np.random.seed(1337)

    EXPLORE = 100000.
    episode_count = 2000
    max_steps = 100
    reward = 0
    done = False
    step = 0
    epsilon = 1
    indicator = 0

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRA)
    critic = CriticNetwork(sess, state_dim, action_dim, BATCH_SIZE, TAU, LRC)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    # load NSL-KDD dataset
    DATA_PATH = '/home/tkdrlf9202/PycharmProjects/DDPG-Keras-AD'
    data_train, data_test = datamodule.loader_all(DATA_PATH)
    idx_discrete = [1, 2, 3]
    idx_continuous = range(len(data_train[0]))
    idx_continuous = np.delete(idx_continuous, idx_discrete).tolist()

    data_train_cont, data_train_disc, data_train_class = datamodule.preprocess(data_train, idx_continuous, idx_discrete)
    # generate tokenizer, one per discrete column
    print('tokenizing discrete features...')
    tokenizers = [kt.Tokenizer(num_words=1000) for i in xrange(len(idx_discrete))]
    # tokenize discrete columns
    data_train_tokenized = datamodule.generate_tokens(tokenizers, data_train_disc)
    vocab_size = [min(len(tokenizers[i].word_index), tokenizers[i].num_words) for i in xrange(len(idx_discrete))]
    print('vocab size of discrete features from training set : ' + str(vocab_size))

    data_train_tokenized = np.array(data_train_tokenized, np.float32)

    # naive concat for tokinized values
    # needs embedding
    data_train_x = np.concatenate((data_train_cont, data_train_tokenized), axis=1)
    data_train_x = normalize(data_train_x, axis=0)

    data_train_y = []
    for elem in data_train_class:
        if elem == 'normal':
            data_train_y.append(0.)
        else:
            data_train_y.append(1.)

    # split the train data for validation
    data_train_x, data_valid_x, data_train_y, data_valid_y = train_test_split(data_train_x,
                                                                              data_train_y,
                                                                              test_size=0.2)


    #Now load the weight
    print("Now we load the weight")
    try:
        actor.model.load_weights("actormodel.h5")
        critic.model.load_weights("criticmodel.h5")
        actor.target_model.load_weights("actormodel.h5")
        critic.target_model.load_weights("criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("Experiment Start.")
    for i in range(episode_count):
        # shuffle the dataset
        feed = list(zip(data_train_x, data_train_y))
        random.shuffle(feed)
        data_train_x, data_train_y = zip(*feed)
        data_train_x = np.array(data_train_x)
        data_train_y = np.array(data_train_y)

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))
        # a feature vector for the current state
        s_t = data_train_x[0]
        total_reward = 0.
        preds = []

        for j in range(max_steps):
            loss = 0 
            epsilon -= 1.0 / EXPLORE
            a_t = np.zeros([1, action_dim])
            noise_t = np.zeros([1, action_dim])

            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0]))

            # store the prediction for accuracy measure
            preds.append(a_t_original[0][0])

            # add OU noise for DDPG training
            noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.5, 0.3)
            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]

            # define reward as the distance btw anomality and ground truth
            r_t = 1 - abs(a_t[0][0]-data_train_y[j])

            # currently the next state is independent from the current action
            # wrong assumption for RL, but for initial debugging purpose
            s_t1 = data_train_x[j+1]

            # if the full training set is used, add the done flag
            if (j + 1 == data_train_x.shape[0]):
                done = 1
            else:
                done = 0

            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            
            # Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])  
           
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]
       
            if train_indicator:
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

            total_reward += r_t
            s_t = s_t1

            if step % 100 == 99:
                # training accuracy
                preds_ = np.where(np.array(preds)>=0.5, 1, 0)
                accuracy = accuracy_score(data_train_y[:j+1], preds_)
                print("Episode", i, "Step", step, "Action_original", a_t_original, "Action_OU", a_t, "Reward", r_t, "Loss", loss, "Accuracy", accuracy)

            step += 1
            if done:
                break

        # model validation per episode
        preds_val = []
        for j in range(data_valid_x.shape[0]):
            preds_val.append(actor.model.predict(data_valid_x[j].reshape(1, data_valid_x[j].shape[0]))[0][0])
        preds_val_ = np.where(np.array(preds_val)>=0.5, 1, 0)
        accuracy_val = accuracy_score(data_valid_y, preds_val_)
        print("validation accuracy", accuracy_val)

        # episode / total reward / total loss / training accuracy / validation accuracy
        log_string = str(i) + '\t' + \
                     str(total_reward) + '\t' +\
                     str(loss) + '\t' +\
                     str(accuracy) + '\t' +\
                     str(accuracy_val) + '\n'
        log_file.write(log_string)

        if np.mod(i, 10) == 0:
            if train_indicator:
                print("Now we save model")
                actor.model.save_weights("actormodel.h5", overwrite=True)
                with open("actormodel.json", "w") as outfile:
                    json.dump(actor.model.to_json(), outfile)

                critic.model.save_weights("criticmodel.h5", overwrite=True)
                with open("criticmodel.json", "w") as outfile:
                    json.dump(critic.model.to_json(), outfile)

        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    print("Finish.")

if __name__ == "__main__":
    playGame()
