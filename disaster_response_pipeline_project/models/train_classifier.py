import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from time import time
from sklearn.metrics import precision_recall_fscore_support as score

def load_data(database_filepath):
    # This function loads the database.db into a dataframe   
    engine = create_engine('sqlite:///DisasterReponse.db')
    df = pd.read_sql('DisasterResponse', con= 'sqlite:///DisasterReponse.db')
    
    # Now the feature X are the messages and the Y targets are all the categories
    # In our case we will build a Multi output classifier
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = df.columns[4:]
    print ('The loaded category names are', category_names)
    
    return X, Y, category_names


def tokenize(text):
    #This function tokenizes and lemmitizes the text by words, in addition to 
    # removing stopwords and special characters
    tokenized_text_list = word_tokenize(text, language='english')

    # Next we remove stopwords of this normalized and tokenized words.
    text_tokenized_no_stop_words = [w for w in tokenized_text_list
                                    if w not in stopwords.words('english')]
    
    # Next we use Lemmitization to get this text ready for feature extraction 
    lemmed_text = [WordNetLemmatizer().lemmatize(w).lower().strip() 
                   for w in text_tokenized_no_stop_words]
    
    # We remove all special characters that are not letters in the alphabet or numbers
    cleaned_tokens = list(filter(lambda x:
                                 x, map(lambda x:re.sub(r'[^a-zA-Z0-9]', '', x)
                                        , lemmed_text)))
    
    return cleaned_tokens
    
def build_model():
    # This function builds the model in a pipeline  and then the hyper parameters 
    # defined in parameter grid will then be optimized over Gridsearch. 
    # Consequently a model will be returned which can then be fitted to the 
    # train and test splits of the features and labels to evaluate the 
    # performance of the model.
    
    # When building the ML NLP Pipeline as done in standard ways, for feature
    # extraction we vectoize the tokenized text, then we transform this
    # before finally applying a multioutput classifier 
    pipeline = Pipeline([ ('vect', CountVectorizer(tokenizer=tokenize)), 
                         ('tfidf', TfidfTransformer()), 
                         ('clf', MultiOutputClassifier(RandomForestClassifier())), ])
    
    # We define below the hyperparameters over which we would like to optimise using
    # GridSearch
    param_grid = {'clf__estimator__min_samples_split': [2, 3]}
    
    # Here we use Grid search to optimize over the above mentioned hyperparameters
    # defined in param_grid
    grid = GridSearchCV(pipeline, param_grid)
    
    return grid


def evaluate_model(model, X_test, Y_test, category_names):
    # We will use the metric precision_recall_fscore_support which is imported as 
    # score to measure the performance of the model. After trying in the notebook
    # the algorithm chosen for this project is the Randon Forest Classifier
    # in order to show the capability of building the pipeline and GridSearch
    
    # We first build teh dataframe which will be output as the perfo of the algorithm
    results = pd.DataFrame(columns= ['category_names', 'precision', 'recall', 'f1-score',
                                      'support'])
    
    # We update the category_names column
    results['category_names'] = category_names
    
    # We predict on the test set what the messages would predict in terms of categories.
    Y_pred = model.predict(X_test)
    
    # Nowe we apply the score method in order to update the results dataframe
    # We need to go through each column of both the predicted Y and the test Y 
    # column by column due to the multi ouput character of the target
    
    for i in range(len(category_names)):
        precision, recall, fscore, support = score(Y_test.iloc[:, i],
                                                   Y_pred[:, i],average='macro')
        results.loc[i,'precision'] = precision
        results.loc[i,'recall'] = recall
        results.loc[i,'f1-score'] = fscore
        results.loc[i,'support'] = support
        
    #For this project we output an array of the mean of the precision, recall and fscore
    print ('The scores of the algorithm is', 
           results[['precision','recall', 'f1-score']].mean())
    return results[['precision','recall', 'f1-score']].mean()
    
def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        t0 = time()
        model.fit(X_train, Y_train)
        print ("training time: ", round(time()-t0, 3), "s")
            
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()