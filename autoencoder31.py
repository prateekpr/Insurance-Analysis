import pickle
import pandas as pd
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from tensorflow import keras
import tensorflow as tf
from gen_report import generate_html_table
from tensorflow.keras.models import load_model

def cat_dim(dataset):
    
    l=['commit_sha']
    dataset.drop(l, axis=1, inplace=True)
    # print(dataset)

    # Select the column containing categorical data
    categorical_column = ['author','files_modified','Test Name']  # Replace with your actual column name
    # Apply one-hot encoding
    dummy_df = pd.get_dummies(dataset[categorical_column], prefix=categorical_column)
    dataset = pd.concat([dataset, dummy_df], axis=1)
    dataset.drop(categorical_column, axis=1, inplace=True)
    print(dataset)
    # X_train=dataset

# Build the autoencoder
    global input_dim 
    input_dim = dataset.shape[1]
    return dataset,input_dim


def autoencoder():
    dataset = pd.read_csv("100.csv")
    X_train,input_dim=cat_dim(dataset)
    '''
    l=['commit_sha']
    dataset.drop(l, axis=1, inplace=True)
    # print(dataset)

    # Select the column containing categorical data
    categorical_column = ['author','files_modified','Test Name']  # Replace with your actual column name
    # Apply one-hot encoding
    dummy_df = pd.get_dummies(dataset[categorical_column], prefix=categorical_column)
    dataset = pd.concat([dataset, dummy_df], axis=1)
    dataset.drop(categorical_column, axis=1, inplace=True)
    print(dataset)
    X_train=dataset

# Build the autoencoder
    # global input_dim 
    input_dim = X_train.shape[1]
    '''
    print(input_dim)
    
    hidden_dim = 10


    input_layer = Input(shape=(input_dim,))
    hidden_layer = Dense(hidden_dim, activation='relu')(input_layer)
    output_layer = Dense(input_dim, activation='linear')(hidden_layer)


    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')


# Train the autoencoder
    autoencoder.fit(X_train, X_train, epochs=400, batch_size=32, validation_split=0.2)

# Example code for saving a Keras autoencoder model


# Train your autoencoder model here
# ...

# Save the model
    autoencoder.save("autoencoder_model.h5")
    # csv_file = 'Output_3_chance_%.csv' 
dataset=pd.read_csv('100.csv')
_,desired_shape = cat_dim(dataset)
def evaluatedata():

    # Load the trained autoencoder model
    autoencoder = load_model('autoencoder_model.h5')  # Replace 'autoencoder_model.h5' with the path to your trained model
    dataset = pd.read_csv("50.csv")
    
    l=['commit_sha']
    dataset.drop(l, axis=1, inplace=True)

    # Assuming your model starts with a Dense layer as the input layer
        # categorical_column = ['files_modified','Test Name'] 
    categorical_column = ['author','files_modified','Test Name']   # Replace with your actual column name
        # Apply one-hot encoding
    dummy_df = pd.get_dummies(dataset[categorical_column], prefix=categorical_column)
    dataset = pd.concat([dataset, dummy_df], axis=1)
    dataset.drop(categorical_column, axis=1, inplace=True)

    # desired_shape = 208
    # desired_shape = input_dim

    
    print(desired_shape)

    # Method 1: Adding columns with zero values
    extra_columns = pd.DataFrame(np.zeros((dataset.shape[0], desired_shape - dataset.shape[1])))
    X_test = pd.concat([dataset, extra_columns], axis=1)

    # Method 2: Reshaping existing columns
    X_test = X_test.values.reshape(X_test.shape[0], desired_shape)

    # Display the resulting DataFrame shape
    # print(df.shape)
    # file_path = 'tt1.csv'
    X_test=pd.DataFrame(X_test)
    # # Save the NumPy array to a CSV file
    # np.savetxt(file_path,dataset, delimiter=',')

    X_test = X_test.astype(np.float32)


    reconstruction_error = autoencoder.evaluate(X_test, X_test, verbose=0)
        # print(f"Reconstruction error: {reconstruction_error:.4f}")
    print(f"Reconstruction error: {reconstruction_error}")

        # Detect flaky test cases
    reconstructed = autoencoder.predict(X_test)
    test=pd.read_csv('test_new.csv')

    chance=list()
    for i in X_test.index:
            j=i+1
            reconstruction_error = autoencoder.evaluate(X_test[i:j], X_test[i:j], verbose=0)
            # value1=reconstruction_error*100
            # value1=value1+40
            chance.append(round(reconstruction_error*100,2))

    df1=pd.read_csv('test_new.csv')
        # print(df1)
    df2 = pd.DataFrame(chance, columns=['Flakiness %'])
        # print(df2)
    df3 = df1.join(df2)
        # df3 = pd.merge(df1, df2)
    print(df3)
    df3.to_csv('Output_3_chance_%.csv',index=False)
    # gen_report 

    import csv
    import html

    html_table = '<table>\n'
    csv_file = 'Output_3_chance_%.csv'   
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for i,row in enumerate(reader):
            html_table += '  <tr>\n'
            if i==0:
                for cell in row:
                    html_table += f'    <th>{cell}</th>\n'
            else:
                for cell in row:
                    value=float(row[1])
                    value=int(value)
                    if value>50:
                        html_table += f'    <td class="fail">{cell}</td>\n'
                    elif value<50:
                        html_table += f'    <td class="pass">{cell}</td>\n'
                        html_table += '  </tr>\n'
                    else:
                        pass    
        
    html_table += '</table>'
    # Replace with the path to your CSV file
    html_table = generate_html_table()

    html_template = f'''
    <!DOCTYPE html>
    <html>
    <head>
    <title>Flakiness_detection_Report</title>
    <link rel="stylesheet" href="style.css">

    </head>
    <body>
    <h1>Test Report</h1>
    
    <div class="section">
        <div class="section-title">Summary</div>
        <div class="section-content">
        <p>Flaky tests are tests that fail intermittently, making it difficult to determine whether they are actually failing or not. This can be a major problem, as it can lead to false positives and false negatives.
        <p>If flaky percentage is higher it indicates inconsistent and unreliable test results and leads to be a flaky failure.    </div>
    </div>

    <div class="section">
        <div class="section-title">Test Case Results</div>
        <div class="section-content">
        
        {html_table}

        </div>
    </div>
    '''
    with open('index.html', 'w') as file:
        file.write(html_template)