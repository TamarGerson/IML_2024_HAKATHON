import numpy as np
import pandas as pd
PASSENGER_PRE_PRO_COLUMNS = ["passengers_up" #LABLES
                             ,"passengers_continue"]



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






#TODO PASSENGERS---------------------------------------------------------------:
#TODO: passengers_continue -> threshold? pas_up?
def clean_negative_passengers(X: pd.DataFrame):
    for passeng_fitch in PASSENGER_PRE_PRO_COLUMNS:
        X = X[X[passeng_fitch] >= 0]
    return X



#   -> no floats (clean_half_pepole())
def clean_half_persons(X: pd.DataFrame):
    for passeng_fitch in PASSENGER_PRE_PRO_COLUMNS:
        X[passeng_fitch] = pd.to_numeric(X[passeng_fitch], errors='coerce')
    X.dropna(subset=PASSENGER_PRE_PRO_COLUMNS)
    return X
#PASSENGERS---------------------------------------------------------------:



#TODO station---------------------------------------------------------------::
def clean_time_in_station(X: pd.DataFrame):
    X = X[(X['door_closing_time'] - X['arrival_time']) >= 0]
    return X
#station---------------------------------------------------------------::



#TODO ALL---------------------------------------------------------------::: 
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
#ALL---------------------------------------------------------------::: 


OUTLIERS_KEYS = [
    "clean_half_persons",
    "clean_time_in_station",
    "clean_negative_passengers"
]



OUTLIERS_FUNC = {
    "clean_time_in_station" : clean_time_in_station
    ,"clean_half_persons" : clean_half_persons
    ,"clean_negative_passengers": clean_negative_passengers
}



PREP_FUNC = {
    "delete_null" : delete_null
    ,"delete_outliers" : delete_outliers
    }