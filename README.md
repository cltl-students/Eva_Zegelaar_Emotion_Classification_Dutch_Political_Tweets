# READ ME

This forms part of my master thesis for the Text Mining Linguistics Master at the VU. 

Here, a CNN-BiLSTM and a simple SVM is used to classify Dutch tweets written by members of the Dutch parliament into emotion categories (6 labels), binary purpose categories (2 labels) and basic polarity categories (3 labels). 

The baseline SVM resulted in a highest f1-score of 0.59 with polarity labels (positive, negative, neutral). The CNN-BiLSTM performed very low in all other categories as it needs more training data. More information on the system performance can be found in the written thesis (pdf file). 

# Thesis Report
Eva_Zegelaar_Thesis_Report.pdf


# Data
In the 'Data' folder you can find the excel file 'gold.xlsx' with the training and test data containing all the gold annotations. The data can also be found in the folders containing the the systems. 

# Systems
2 Folders named: 'Main_CNN_BiLSTM'  & 'Baseline_SVM'. 

In each folder you can find one jupyter notebook containing the code and the data for system input. 

Folder 'Pipeline-for-Internship-Red-Data', containins the final pipeline for Red Data in a .py file. This .py file contains the two 'best' performing systems (SVM polarity labels and SVM proactivity labels) and outputs a JSON file containing the tweets with their two labels. 

# Resources
Word Embeddings:
Due to the large size, the open-source pre-trained Dutch word embeddings were not uploaded here. However, to be able to run the CNN-BiLSTM, you need those word embeddings. These acan be downloaded from the following github page: https://github.com/coosto/dutch-word-embeddings.


# References

A. Nieuwenhuisje. Open-source dutch word embeddings, Jul 2018a. URL https://www.linkedin.com/pulse/open-source-dutch-word-embeddings-alexander-nieuwenhuijse/.

A. Nieuwenhuisje. dutch-word-embeddings. Jul 2018b. URL https://github.com/ coosto/dutch-word-embeddings.

S. Dandge. saitejdandge/sentimentalanalysislstmconv1d,Feb2019. https://github.com/saitejdandge/Sentimental_Analysis_LSTM_Conv1D

Kaggle. Lstm with word2vec embeddings, Apr 2017. URL https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings.

Keras. Keras documentation: Bidirectional lstm on imdb, May 2020. URL https://keras.io/examples/nlp/bidirectional_lstm_imdb/.

Z.-X. Liu, D.-G. Zhang, G.-Z. Luo, M. Lian, and B. Liu. A new method of emotional analysis based on cnn–bilstm hybrid neural network. Cluster Computing, Mar 2020. 10.1007/s10586-020-03055-9.
