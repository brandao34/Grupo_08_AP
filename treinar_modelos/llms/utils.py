import numpy as np
import pandas as pd
import os

def load_dataset(dataset_version ):
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
        df = pd.concat([train,test,validation])
    else:
        df = pd.concat([train,test])

    return df