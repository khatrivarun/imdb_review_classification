# iMDb Movie Review Sentiment Classification
Using Scikit Learn modules to classify movie reviews into positive review and negative reviews!
My first project to get into the domain of machine learning and NLP in general and this serves me a way to understand the process and tasks to carry out in order to process human language.

## Data Used

I used the infamous [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) and prepared a custom pipeline to download the data, extract and read the data for further processing

## Data Processing Strategy

1. Removed all HTML tags and also numbers and special characters to keep only words in the reviews by writing a custom sklearn transformer.
2. Tokenizing and filtering of stopwords which builds a dictionary of features and transforms documents to feature vectors using `CountVectorizer`.
3. Divide the number of occurrences of each word in a document by the total number of words in the document to generate term frequencies using `TfidfTransformer`.
4. Using various classical machine learning models to train and test on the training and testing data generated when reading in the data.
5. Tuning the model's performance by tuning their hyperparameters using `RandomizedSearchCV` and also providing a parameter search space.

## Issues and Pull Requests

Any suggestions on how to improve the model's metrics are welcomed through issues posted on the issues tabs or pull requests!
