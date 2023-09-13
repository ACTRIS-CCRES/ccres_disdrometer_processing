import pandas as pd
from create_input_files_quicklooks import get_data_event

path_events = (
    "/home/ygrit/Documents/disdro_processing/ccres_disdrometer_processing/"
    "bdd_rain_events/palaiseau/rain_events_palaiseau_length180_event60.csv"
)
events = pd.read_csv(path_events)

events["Start_time"] = pd.to_datetime(events["Start_time"])
events["End_time"] = pd.to_datetime(events["End_time"])

weather, dcr, disdro = get_data_event(
    events["Start_time"][0],
    events["End_time"][0],
)
print(weather)
