# Yelp Dataset Analysis
## Project Objectives
Goal is to predict the propability for bisinesses closure based on the attributes in the yelp dataset with a logistic regression model.
## Yelp Dataset
The yelp dataset contrains informations about businesses, yelp users und reviews form yelp review website.


**Business Attributes:**
* business_id: ID of the business
* name: name of the business
* address: address of the business
* city: city of the business
* state: state of the business
* postal_code: postal code of the business
* latitude: latitude of the business
* longitude: longitude of the business
* stars: average rating of the business
* review_count: number of reviews received
* is_open: 1 if the business is open, 0 otherwise
* categories: multiple categories of the business

**Review Attributes:**
* review_id: ID of the review
* user_id: ID of the user
* business_id: ID of the business
* stars: ratings of the business
* date: review date
* text: review
* useful: number of users who vote a review as usefull
* funny: number of users who vote a review as funny
* cool: number of users who vote a review as cool

## Setup
Download the Dataset on https://www.yelp.com/dataset and setup die following folder structure.
```
├── mount                                       
│   ├── analyseBusinessClosed.py                # script for analyzing extracted features
│   ├── extractFeatures.py                      # script for extracting features from data
│   ├── readData.py                             # script for data reading
│   ├── filterStr.py                            # script for filtering data based on substings
│   ├── start.sh                                # docker entry pont
│   ├── yelp_academic_dataset_business.json     # business data from yelp dataset
│   └── yelp_academic_dataset_review.json       # review data from yelp dataset
├──Dockerfile                                   # file for setting up the docker image
├──win_start.bat                                # windows starting script
├──ux_start.sh                                  # linux starting script
```
### Run With Docker
The docker image only setsup the environment and links the the starting script inside a mounted folder. The starting script, python files and the datasets are inside a mounted folder.
1. download yelp dataset
2. setup folder structure
3. install docker
4. create docker image from the Dockerfile with the following command
    ```
    docker build -t python-yelpanalysis .
    ```
5. run the docker container with the following command
    Windows:
    ```
    docker run --mount type=bind,source="%cd%"\mount,target=/app/mount python-yelpanalysis
    ```
    Linux (not checked if it works): 
    ```
    docker run --mount type=bind,source="$(pwd)"/mount,target=/app/mount python-yelpanalysis
    ```
### Run Without Docker
1. download yelp dataset
2. setup folder structure
3. install docker
4. run the right starting script for your operating system in the base folder

## Scripts
Example starting script for analyzing businesses of the categories Nightlife, Bars, Pubs, SportsBars and BeerBar:
```
python ./mount/readData.py  --filename yelp_academic_dataset_business  \
                            --outputname yelp_academic_dataset_businesss_raw  \
                            --columns business_id,state,stars,review_count,is_open,hours,categories,latitude,longitude,city
python ./mount/readData.py  --filename yelp_academic_dataset_review   \
                            --outputname yelp_academic_dataset_review_raw   \
                            --columns review_id,user_id,business_id,stars,date

python ./mount/filterStr.py --strInc Nightlife,Bars,Pubs,SportsBars,BeerBar    \
                            --inputname yelp_academic_dataset_businesss_raw    \
                            --columnName categories    \
                            --outputname yelp_academic_dataset_businesss_filtered   \

python ./mount/extractFeatures.py --inputnameBusinesses yelp_academic_dataset_businesss_filtered    \
                                  --inputnameReviews yelp_academic_dataset_review_raw
```
### filterStr.py
This script reads in a dataframe and filters the rows based on substrings in a specific column and saves the filtered dataframe.
Only include rows witch contain at least one string of the strInc's as a substring in a specific column with the name columnName.
Exclude strings witch contain at least one string of the strExc's as a substring in a specific column with the name columnName.
Filter some chars.

#### Starting Parameters
- **inputname:** Name of the input dataframe
-  **strInc:**    
Only include rows witch contain at least one string of the strInc's as a substring in a specific column with the name columnName (seperator = ,)
-  **columnName:**  Filter based on the strings in this column
-  **outputname:**  Dataframe is saved under this name
-  **strExc:**          Exclude strings witch contain at least one string of the strExc's as a substring in a specific column with the name columnName (seperator = ,)
-  **folder:**      Relative path of folder to operate in
### readData.py
This script reads in .json files and saves the data in a pandas dataframe.
#### Starting Parameters
- **filename:**     Name of the json file to read
- **outputname:**   Dataframe is saved under this name
- **columns:**  Read only some columns (speperator = ,)
- **query:**    filter data with query
- **chunksize:**    Chunksize for reading the data
- **folder:**   Relative path of folder to operate in
### extractFeatures.py
This script reads in a dataframe containing the yelp business data and a dataframe containing the yelp review data and extracts features describing the busnesses.

#### Columns of input data
Businesses: 
business_id, state, stars, review_count, is_open, hours, categories, latitude, longitude, city

Reviews: 
review_id, user_id, business_id, stars,date
#### Extracted Features

Extracted features:
stars, review_count, latitude, longitude, EveryDay_h, total_h, Monday_h, Tuesday_h, "Wednesday_h, Thursday_h, Friday_h, Saturday_h, Sunday_h, ReviewsPerDay, ratingTrend, Age, RewFreqTrend, BusPerCity, "BusPerState

#### Starting Parameters
- **inputnameBusinesses:**    Name of the input business dataframe
- **inputnameReviews:**    Name of the input review datafram
- **outputname:**    Dataframe containing the features and labels is saved under this name
- **folder:**    Relative path of folder to operate in

### analyseBusinessClosed.py
This script reads trains and evaluates a logistic regression model prediction the probability for business closure.
the the prediction is made based of all attributes and on subsets of attributes containing only most important attributes.
plot the ROC curve, feature importance and a report file containing some performance metrics for all of the models

#### Starting Parameters
- **inputname:**    Name of data file
- **NameLable:**    name of label
- **folder:**    Relative path of folder to operate in