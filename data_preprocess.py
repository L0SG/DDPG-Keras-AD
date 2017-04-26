import datamodule

DATA_PATH = '/home/tkdrlf9202/PycharmProjects/DDPG-Keras-AD'

data_train, data_valid, data_test = datamodule.loader_all(DATA_PATH)