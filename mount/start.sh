#!/bin/sh

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

python ./mount/analyseBusinessClosed.py --inputname FeaturesWithLabels