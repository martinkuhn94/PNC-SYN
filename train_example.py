import pm4py

from PNC_SYN.synthesizer import PNCEventLogSynthesizer

# Read Event Log Road_Traffic_Fine_Management_Process.xes
xes_file_path = "example_logs/Road_Fines_Short_Event_Log.xes"
event_log = pm4py.read_xes(xes_file_path)

# Initialize Model
palsyn_model = PNCEventLogSynthesizer(
    embedding_output_dims=128,
    epochs=5,
    batch_size=128,
    dropout=0.3,
    trace_quantile=0.9,
    method="LSTM",
    units_per_layer=[16],
)

# Train Model
palsyn_model.fit(event_log)
palsyn_model.save_model("experiments/models/LSTM_Road_Fines_Short_Event_Log_u=16_ep=5")
