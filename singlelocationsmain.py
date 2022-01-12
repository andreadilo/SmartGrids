import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import GlorotUniform

df1 = pd.read_csv(r"data\feats_mesloc1_voltage_phaseA_withRE.csv", sep=",").dropna()
df2 = pd.read_csv(r"data\feats_mesloc2_voltage_phaseA_withRE.csv", sep=",").dropna()
df3 = pd.read_csv(r"data\feats_mesloc3_voltage_phaseA_withRE.csv", sep=",").dropna()
df4 = pd.read_csv(r"data\feats_mesloc4_voltage_phaseA_withRE.csv", sep=",").dropna()
df1_healty = pd.read_csv(r"data\feats_mesloc1_voltage_phaseA_healty_withRE.csv").dropna()
df2_healty = pd.read_csv(r"data\feats_mesloc2_voltage_phaseA_healty_withRE.csv").dropna()
df3_healty = pd.read_csv(r"data\feats_mesloc3_voltage_phaseA_healty_withRE.csv").dropna()
df4_healty = pd.read_csv(r"data\feats_mesloc4_voltage_phaseA_healty_withRE.csv").dropna()

df1 = df1.drop(columns=['resistance'])
df2 = df2.drop(columns=['resistance'])
df3 = df3.drop(columns=['resistance'])
df4 = df4.drop(columns=['resistance'])

df1_healty = df1_healty.drop(columns=['lineLength'])
df2_healty = df2_healty.drop(columns=['lineLength'])
df3_healty = df3_healty.drop(columns=['lineLength'])
df4_healty = df4_healty.drop(columns=['lineLength'])

df1_healty['locLabel'] = ['0'] * len(df1_healty)
df2_healty['locLabel'] = ['0'] * len(df2_healty)
df3_healty['locLabel'] = ['0'] * len(df3_healty)
df4_healty['locLabel'] = ['0'] * len(df4_healty)


df1_healty['faultLabel'] = ['0'] * len(df1_healty)
df2_healty['faultLabel'] = ['0'] * len(df2_healty)
df3_healty['faultLabel'] = ['0'] * len(df3_healty)
df4_healty['faultLabel'] = ['0'] * len(df4_healty)


df1 = pd.concat([df1, df1_healty])
df2 = pd.concat([df2, df2_healty])
df3 = pd.concat([df3, df3_healty])
df4 = pd.concat([df4, df4_healty])

df = [df1, df2, df3, df4]


def build_model(data, y):
    lmodel = keras.models.Sequential()

    lmodel.add(Dense((len(data[1]) / 2), input_dim=len(data[1]), activation='relu',
                     kernel_initializer=GlorotUniform(),
                     kernel_regularizer=regularizers.l2(l2=1e-3)))

    lmodel.add(Dense(len(y[1]), activation='softmax'))

    lmodel.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=[keras.losses.CategoricalCrossentropy(from_logits=False), ],
        metrics=[keras.metrics.CategoricalAccuracy()]
    )

    return lmodel


locations_results = list()
faults_results = list()

for dataframe in df:

    # Shaping the data
    data = dataframe.drop(labels=['locLabel', 'faultLabel'], axis=1)
    locations = dataframe['locLabel'].values.astype(str)
    faults = dataframe['faultLabel'].values.astype(str)

    # Reshaping Ys
    locations = locations.reshape(-1, 1)
    faults = faults.reshape(-1, 1)

    # Xs Normalization
    norm = MinMaxScaler()
    data = norm.fit_transform(data)

    # Location Y encoding
    loc_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    loc_enc = loc_enc.fit(locations)
    locations = loc_enc.transform(locations)

    # Fault Y encoding
    fault_enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    fault_enc = fault_enc.fit(faults)
    faults = fault_enc.transform(faults)

    # Early stop
    # callback = tf.keras.callbacks.EarlyStopping(monitor='categorical_accuracy', patience=10)

    # Location Model training
    X_train, X_test, y_train, y_test = train_test_split(data, locations, test_size=0.3, shuffle=True)
    model = build_model(data, locations)
    history = model.fit(X_train, y_train, epochs=10, batch_size=16)

    locations_results.append(model.evaluate(X_test, y_test)[1])

    # Fault Model training
    X_train, X_test, y_train, y_test = train_test_split(data, faults, test_size=0.2, shuffle=True)
    model = build_model(data, faults)
    history = model.fit(X_train, y_train, epochs=10, batch_size=8)

    faults_results.append(model.evaluate(X_test, y_test)[1])

indexes = ['Location 1', 'Location 2', 'Location 3', 'Location 4']
names = ['Location', 'Fault']

results = pd.DataFrame(list(zip(locations_results, faults_results)), index=indexes, columns=names)

print(results)

# BASELINE PRIVATA PER OGNI LOCATION MISURA
# OGNI LOCATION HA IL SUO MODELLO

# MODELLO CENTRALIZZATO (SECONDA BASELINE) DATI TUTTI SU UNICA LOCATION
# FL CON SPLIT NEURAL NETWORK