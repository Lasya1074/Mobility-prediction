The current project is about trajectory prediction the dataset we use is 'MICROSOFT Geolife GPS trajectory dataset' it is a public  available dataset in kaggle.
"https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/"
We created two bilstm models for the prediction of trajectory 
the frist we preprocess the data and derive other features like speed, acceleration,pitch, bearing which are useful for trajectory prediction
WE created a BI-LSTM Attention Model and Sequential BI_LSTM Model
The we run is for only 1 lakh datapoints the preprocessed dataset contain nearly 5 million data entries.
