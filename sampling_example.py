import pm4py

from PNC_SYN.postprocessing import clean_xes_file
from PNC_SYN.synthesizer import PNCEventLogSynthesizer

# Load Model
palsyn_model = PNCEventLogSynthesizer()
palsyn_model.load("experiments/models/LSTM_Road_Fines_Short_Event_Log_u=16_ep=5")

# Sample
event_log = palsyn_model.sample(sample_size=5600, batch_size=100)
event_log_xes = pm4py.convert_to_event_log(event_log)

# Save as XES File
xes_filename = "road_fines_e=inf.xes"
pm4py.write_xes(event_log_xes, xes_filename)
clean_xes_file(xes_filename, xes_filename)

# Save as XSLX File for quick inspection
df = pm4py.convert_to_dataframe(event_log_xes)
df["time:timestamp"] = df["time:timestamp"].astype(str)
df.to_excel("road_fines_e=inf.xlsx", index=False)
