# Import of libraries and modules
import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
import pandas as pd
from tensorflow.keras import regularizers
import numpy as np
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Data pre-processing (setting random seed, uploading datasets and splitting into train and test)
tf.random.set_seed(42)
np.random.seed(42)

df1 = pd.read_csv(r"data\feats_mesloc1_voltage_phaseA_withRE.csv", sep=",").dropna()
df2 = pd.read_csv(r"data\feats_mesloc2_voltage_phaseA_withRE.csv", sep=",").dropna()
df3 = pd.read_csv(r"data\feats_mesloc3_voltage_phaseA_withRE.csv", sep=",").dropna()
df4 = pd.read_csv(r"data\feats_mesloc4_voltage_phaseA_withRE.csv", sep=",").dropna()

df = [df1, df2, df3, df4]

training_set_temp = []
test_set_temp = []

for dataframe in df:
    train, test = sklearn.model_selection.train_test_split(
        dataframe, test_size=0.2, shuffle=True, stratify=dataframe['faultLabel']
    )
    training_set_temp.append(train)
    test_set_temp.append(test)

training_sets = []
test_sets = []

for i in range(len(df)):
    # Shaping the data
    data = training_set_temp[i].drop(labels=['locLabel', 'faultLabel'], axis=1)
    locations = training_set_temp[i]['locLabel'].astype(str)
    faults = training_set_temp[i]['faultLabel'].astype(str)

    targets = (locations + faults).values

    data_test = test_set_temp[i].drop(labels=['locLabel', 'faultLabel'], axis=1)
    locations_test = test_set_temp[i]['locLabel'].astype(str)
    faults_test = test_set_temp[i]['faultLabel'].astype(str)

    targets_test = (locations_test + faults_test).values

    # Reshaping Ys
    targets = targets.reshape(-1, 1)
    targets_test = targets_test.reshape(-1, 1)

    # Xs Normalization
    norm = MinMaxScaler()
    data = norm.fit_transform(data)
    data_test = norm.transform(data_test)

    # Targets encoding
    targ_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    targ_enc = targ_enc.fit(targets)
    targets = targ_enc.transform(targets)
    targets_test = targ_enc.transform(targets_test)

    # Location Y encoding
    # loc_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # loc_enc = loc_enc.fit(locations)
    # locations = loc_enc.transform(locations)
    # locations_test = loc_enc.transform(locations_test)

    # Fault Y encoding
    # fault_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    # fault_enc = fault_enc.fit(faults)
    # faults = fault_enc.transform(faults)
    # faults_test = fault_enc.transform(faults_test)

    training_sets.append((data, targets))
    test_sets.append((data_test, targets_test))

# CLASSIFICATION

i0 = Input(shape=(training_sets[0][0].shape[1],), name="device0")
x0 = Dense(32, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l2=1e-4))(i0)
x0 = Dense(64, activation='relu')(x0)
# x0 = Dropout(0.4)(x0)

x0_2 = Dense(78, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l2=1e-4))(
    x0)
x0_2 = Dense(32, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l2=1e-4))(
    x0_2)

i1 = Input(shape=(training_sets[0][0].shape[1],), name="device1")
x1 = Dense(32, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l2=1e-4))(i1)
x1 = Dense(84, activation='relu')(x1)
# x1 = Dropout(0.4)(x1)

x1_2 = Dense(128, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l2=1e-4))(
    x1)
x1_2 = Dense(32, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l2=1e-4))(
    x1_2)

i2 = Input(shape=(training_sets[0][0].shape[1],), name="device2")
x2 = Dense(32, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l2=1e-4))(i2)
x2 = Dense(128, activation='relu')(x2)
# x2 = Dropout(0.4)(x2)

x2_2 = Dense(64, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l2=1e-4))(
    x2)
x2_2 = Dense(32, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l2=1e-4))(
    x2_2)

i3 = Input(shape=(training_sets[0][0].shape[1],), name="device3")
x3 = Dense(32, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l2=1e-4))(i3)
x3 = Dense(64, activation='relu')(x3)
# x3 = Dropout(0.4)(x3)

x3_2 = Dense(64, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l2=1e-4))(
    x3)
x3_2 = Dense(32, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(l2=1e-4))(
    x3_2)

x_2 = tf.keras.layers.concatenate([x0_2, x1_2, x2_2, x3_2])

x_2 = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l2=1e-4))(x_2)
x_2 = Dropout(0.3)(x_2)
x_2 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='glorot_uniform',
                            kernel_regularizer=regularizers.l2(l2=1e-4))(x_2)
x_2 = Dropout(0.3)(x_2)
# x_2 = tf.keras.layers.Dense(16, activation='relu', kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(l2=1e-4))(x_2)


target_pred = tf.keras.layers.Dense(44, activation='softmax', name='fault_pred')(x_2)

model = tf.keras.Model(inputs=[i0, i1, i2, i3], outputs=target_pred)

# keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.CategoricalAccuracy()]
)

model.summary()

callback = tf.keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=10)

model.fit(
    {"device0": training_sets[0][0], "device1": training_sets[1][0], "device2": training_sets[2][0],
     "device3": training_sets[3][0]},
    training_sets[0][1], epochs=1000, batch_size=8,
    callbacks=[callback])

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(
    {"device0": test_sets[0][0], "device1": test_sets[1][0], "device2": test_sets[2][0], "device3": test_sets[3][0]},
    test_sets[0][1])
print("test loss, test acc:", results)

# QUESTO NON è CENTRALIZZATO è SEMPRE UNA SPLIT NN PERO STO VALUTANDO IN MANIERA CONGIUNTA FAULT E LOCATION
# STO PROVANDO A FARE UNA PREDIZIONE CONGIUNTA

