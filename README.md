# Disaster Response Pipeline Project

### File Structure of project
- DisasterResponse.db   # database to save clean data to
- app
    - template
        - master.html  # main page of web app
        - go.html  # classification result page of web app
    - run.py  # Flask file that runs app

- data
    - disaster_categories.csv  # data to process 
    - disaster_messages.csv  # data to process
    - process_data.py

- models
    - train_classifier.py
    - classifier.pkl  # saved model 

- README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

If the ResponseDisaster.db file is already there, then there is no need to run the ETL pipeline and same for the classifier.pkl file in and the ML pipeline command.

Once the Web app is running, two vidualizations of the data are shown as below:

![Image of Data Visualization](https://github.com/NitinRamchand/Disaster_Repsonse_Pipeline_Udacity_Nanodegree_project/blob/master/Graphs.png)

And there is also an example of what the classification algorthm does with the following message: "Hurricane #Maria made landfall near Yabucoa, Puerto Rico, around 6:15am AST with maximum sustained winds of 155 mph (250 km/h)":

![Image of Classification](https://github.com/NitinRamchand/Disaster_Repsonse_Pipeline_Udacity_Nanodegree_project/blob/master/Predicted_cat_message.png)
