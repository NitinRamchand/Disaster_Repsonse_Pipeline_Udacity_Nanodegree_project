import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # This functions loads the messages and the categories datasets which are in csv format 
    # into pandas dataframes and the finally returns a merged dataframe on the 'id' column
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='outer', on='id')
    
    return df

def clean_data(df):
    # This function takes the merged dataframe as input and cleans it up before saving it
    # into a database.
    
    # categories will be a dataframe with all the values expanded of the categories column
    categories = df.categories.str.split(';', expand=True)
    
    # From the expanded Dataframe defined above we will extract the list with all the 
    # column names
    new_catergory_columns = []
    
    # This is just the first row of the dataframe
    row = categories.loc[0]
    
    # Then to extract the column names we take the first part of the string when we split it at the -
    
    for cat in row.str.split('-'):
        new_catergory_columns.append(cat[0])
        
    # Now we update the column names of the categories dataframe
    categories.columns = new_catergory_columns
    
    for cat in new_catergory_columns:
        # For each category we change te value in the dataframe into the last character of the string
        # which indicated whether its a 1 or a 0
        categories[cat] = categories[cat].str[-1]
        
        # We also convert this string into an integer
        categories[cat] = pd.to_numeric(categories[cat])
        
    # Now we add the 'id' row in the categories Dataframe in order to then merge it with df
    categories['id'] = df['id']
    
    # The following step is to drop the categories column in the df
    df.drop(columns='categories', inplace=True)
    
    # Now we merge df with the categories dataframe
    df = df.merge(categories, how='inner', on='id')
    
    #We deal with duplicated rows and we delete them
    print ('number of duplicated items before deleting duplicates are', df.duplicated().sum())
    print('shape of dataframe is', df.shape)
    df.drop_duplicates(inplace=True)
    print ('number of duplicated items after deleting duplicates are', df.duplicated().sum())
    print('shape of dataframe is', df.shape)
    
    return df

def save_data(df, database_filename):
    # This function loads the dataframe into a SQLlite Database 
    
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()