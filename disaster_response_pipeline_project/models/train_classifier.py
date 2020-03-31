import sys
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


def load_data(database_filepath):
    # This function loads the database.db into a dataframe   
    engine = create_engine('sqlite:///DisasterReponse.db')
    df = pd.read_sql('DisasterResponse', con= 'sqlite:///DisasterReponse.db')
    
    # Now the feature X are the messages and the Y targets are all the categories
    # In our case we will build a Multi output classifier
    X = df['messages']
    Y = df[4:]
    category_names = df.columns
    
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
    cleaned_tokens = list(filter(lambda x:x, map(lambda x:re.sub(r'[^a-zA-Z0-9]', '', x), lemmed_text)))
    
    return cleaned_tokens
    
def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


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
        model.fit(X_train, Y_train)
        
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