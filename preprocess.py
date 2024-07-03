import numpy as np
import pandas as pd


#TODO: trip_id -> int (id)
#TODO: part -> int (id)
#TODO: trip_id_unique_station -> int (id)
#TODO: trip_id_unique -> int (id)
#TODO: line_id id -> int (id)
# IS LINE ID RELEVANT (DEST 9 vs 15 rehavia)
#TODO: alternative -> number of alternatives?
#TODO: cluster -> int(id)
#TODO: station_name -> needed? id and index
#TODO: arrival_time -> hh only? peak time only {0,1}
# DO WE NEED TO ADD time^2
#TODO: door_closing_time - arrival_time ? more passenj.
#TODO: passengers_continue -> threshold? pas_up?

#TODO PASSENGERS:
#   -> no floats (clean_half_pepole())
def clean_half_persons(X: pd.DataFrame):
    
    pass


#TODO station :
#   -> door_closing_time - arrival_time < 0 ? 
def clean_time_in_station(X: pd.DataFrame):
    X = X[(X['door_closing_time'] - X['arrival_time']) >= 0]
    return X



#TODO ALL: 
def delete_null(X: pd.DataFrame, y = None):
    
    df = X.copy()
    df["lable"] = y
    
    df = df.dropna() #.drop_duplicates() 
    df = df[df["lable"] < 0] # no negative number of pass
    
    y = df["lable"]
    X = df.drop(columns = ["lable"])
    
    return X, y


def delete_outliers(X: pd.DataFrame):
    #WHAT ARE THE OUTLIERS?
    for lier in OUTLIERS_KEYS:
        OUTLIERS_FUNC[lier](X)
    return X






OUTLIERS_KEYS = [
    "clean_half_persons",
    "clean_time_in_station"
]


OUTLIERS_FUNC = {
    "clean_time_in_station" : clean_time_in_station
    "clean_half_persons" : 
}


PREP_FUNC = {
    "delete_null" : delete_null,
    "delete_outliers" : delete_outliers
    }