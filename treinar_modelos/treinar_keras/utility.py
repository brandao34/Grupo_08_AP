import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

## Funções de anteriores de treinar_DNN
def load_dataset(dataset_version):
    base_path = f'../../datasets/{dataset_version}/'
    train_file = f'Dataset{dataset_version}_train_clean.csv'
    test_file = f'Dataset{dataset_version}_test_clean.csv'
    validation_file = f'Dataset{dataset_version}_validation_clean.csv'
    
    train = pd.read_csv(os.path.join(base_path, train_file), sep=',')
    test = pd.read_csv(os.path.join(base_path, test_file), sep=',')
    
    # Verifica se o arquivo de validação existe
    validation_path = os.path.join(base_path, validation_file)
    if os.path.exists(validation_path):
        validation = pd.read_csv(validation_path, sep=',')
    else:
        validation = None  # Define como None se não existir
    
    return train, test, validation

def tfidf_preprocessing(train, test, validation):
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train = tfidf.fit_transform(train['text']).toarray()
    X_test = tfidf.transform(test['text']).toarray()
    
    y_train = train['Label'].map({'AI':1, 'Human':0}).values
    y_test = test['Label'].map({'AI':1, 'Human':0}).values
    
    # Processa validação apenas se existir
    if validation is not None:
        X_val = tfidf.transform(validation['text']).toarray()
        y_val = validation['Label'].map({'AI':1, 'Human':0}).values
    else:
        X_val, y_val = None, None  # Retorna None se não houver validação
    
    return X_train, y_train, X_test, y_test, tfidf, X_val, y_val


import tensorflow as tf

def keras_text_preprocessing(train,test,val, text_vectorization,batch_size, do_print = True):
    train_ds = tf.data.Dataset.from_tensor_slices((train["text"].values,
                                               train["Label"].map({'AI':1, 'Human':0}).values)).batch(batch_size)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val["text"].values,
                                               val["Label"].map({'AI':1, 'Human':0}).values)).batch(batch_size)
    
    test_ds = tf.data.Dataset.from_tensor_slices((test["text"].values,
                                               test["Label"].map({'AI':1, 'Human':0}).values)).batch(batch_size)
 

    text_only_train_ds = train_ds.map(lambda x, y: x)
    text_vectorization.adapt(text_only_train_ds)

    binary_ngram_train_ds = train_ds.map(  lambda x, y: (text_vectorization(x), y),
                                            num_parallel_calls=tf.data.AUTOTUNE)
    
    binary_ngram_test_ds = test_ds.map( lambda x, y: (text_vectorization(x), y),
                                num_parallel_calls=tf.data.AUTOTUNE)

    binary_ngram_val_ds = val_ds.map( lambda x, y: (text_vectorization(x), y),
                                            num_parallel_calls=tf.data.AUTOTUNE)

    if do_print:
        print(text_vectorization.get_vocabulary())
        for inputs, targets in binary_ngram_train_ds:
            print("inputs.shape:", inputs.shape)
            print("inputs.dtype:", inputs.dtype)
            print("targets.shape:", targets.shape)
            print("targets.dtype:", targets.dtype)
            #print("inputs[0]:", inputs[0])
            #print("targets[0]:", targets[0])
            break

    return binary_ngram_train_ds, binary_ngram_test_ds, binary_ngram_val_ds


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop




def get_dnn_model(max_tokens=20000, hidden_dims=[64,32], dropout = 0.2, learning_rate = 0.0001):
    inputs = keras.Input(shape=(max_tokens,))
    x = inputs
    for hidden_dim in hidden_dims:
        x = layers.Dense(hidden_dim, activation="relu")(x)
        x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    Optimizer = RMSprop(learning_rate=learning_rate)  
    model.compile(optimizer=Optimizer,
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def train_model(model,model_name,tf_train,tf_val):
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",  # Stop if validation loss stops improving
        patience=5,  # Number of epochs to wait before stopping
        restore_best_weights=True  # Restore the best model weights after stopping
    )

    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        model_name,
        save_best_only=True,
        monitor="val_loss",
        mode="min"
    )
    callbacks = [ early_stopping ,
        checkpoint_callback
    ]

    history = model.fit(tf_train.cache(),
            validation_data=tf_val.cache(),
            epochs=5000,
            callbacks=callbacks)
    
    model = keras.models.load_model(model_name)

    return history, {"train_loss": history.history['loss'][-1],
                     "train_accuracy": history.history['accuracy'][-1],
                     "val_loss": history.history["val_loss"][-1], 
                     "val_accuracy":history.history["val_accuracy"][-1]  }


import matplotlib.pyplot as plt

def plot_losses(history):
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss", linestyle="dashed")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.show()

from sklearn.metrics import classification_report, confusion_matrix

def test_model(model,test):
    metrics = model.evaluate(test)

    y_test = np.concatenate([y.numpy() for _, y in test]) 
    y_pred_probs = model.predict(test)
    y_pred = (y_pred_probs > 0.5).astype(int) 
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    return {"test_loss": metrics[0], "test_accuracy": metrics[1]}