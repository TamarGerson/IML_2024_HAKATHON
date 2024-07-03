

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



#TODO ALL: 
#   -> clean null
def delete_null_and_outliers(X, y = None):
    
    # X = delete_outliers(X) #WHAT ARE THE OUTLIERS?
    df = X.copy()
    df["lable"] = y
    
    df = df.dropna() #.drop_duplicates() 
    df = df[df["lable"] < 0] # no negative number of pass
    
    y = df["lable"]
    X = df.drop(columns = ["lable"])
    
    return X, y


def delete_outliers(X):
    #WHAT ARE THE OUTLIERS?
    return X
#   -> 



#TODO PASSENGERS:
#   -> no floats (clean_half_pepole())


#TODO station :
#   -> door_closing_time - arrival_time < 0 ? 