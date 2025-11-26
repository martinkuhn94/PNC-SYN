# PNC-SYN (Petri Net Conditional Synthesizer)

## Overview

PNC-SYN is a toolkit for generating process-oriented synthetic event logs that satisfy
user-defined conditions derived from Petri nets. The focus is on what-if analysis:
you can probe how a process would behave under specific structural constraints,
such as enforcing a particular routing, steering cases toward rare variants, or
capturing post-drift behavior.

This has created a growing interest in synthetic event log generation techniques, and in particular in conditional generation methods that can produce event data under specific user-defined constraints. The ability to generate realistic event data under user-specified conditions, for example, to study statistical properties of post-drift behavior, particular outcome classes, or rare process variants, is therefore increasingly important.

## Features

- **Petri-Net-Guided Conditioning:** Discover Petri nets and reuse them as constraints for sampling specific routing alternatives.
- **Scenario / What-If Analysis:** Generate counterfactual traces to analyze drifts, rare outcomes, or targeted case classes.
- **Multi-Perspective Modeling:** Jointly learns activity labels and inter-event times to keep control-flow and temporal realism aligned.

## Installation
Choose the workflow that best matches your setup.

### 1. Minimal Runtime (requirements.txt)
```bash
git clone https://github.com/martinkuhn94/PNC-SYN.git
cd PNC-SYN
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Editable Install (`pip install .`)
```bash
git clone https://github.com/martinkuhn94/PNC-SYN.git
cd PNC-SYN
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install .
```

### 3. Development Environment (`pip install -e .[dev]`)
```bash
git clone https://github.com/martinkuhn94/PNC-SYN.git
cd PNC-SYN
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .[dev]
```
This third option installs tooling such as Ruff, mypy, pytest, coverage, and type stubs that are referenced in `pyproject.toml`.

## Usage

### Training the Model
The snippet below matches `train_example.py`. It trains an LSTM with 128-dimensional embeddings, a single 16-unit recurrent layer, dropout, and a 90th-percentile trace filter before saving the weights to `experiments/models`.

```python
import pm4py
from PNC_SYN.synthesizer import PNCEventLogSynthesizer

xes_file_path = "example_logs/Road_Fines_Short_Event_Log.xes"
event_log = pm4py.read_xes(xes_file_path)

palsyn_model = PNCEventLogSynthesizer(
    embedding_output_dims=128,
    epochs=5,
    batch_size=128,
    dropout=0.3,
    trace_quantile=0.9,
    method="LSTM",
    units_per_layer=[16],
)

palsyn_model.fit(event_log)
palsyn_model.save_model("experiments/models/LSTM_Road_Fines_Short_Event_Log_u=16_ep=5")
```

### Sampling Event Logs
After training or loading a saved model, sample synthetic traces and export them to XES/Excel as shown in `sampling_example.py`. Generated models, synthetic logs, and evaluation outputs all live under the `experiments/` directory by default.

```python
import pm4py

from PNC_SYN.postprocessing import clean_xes_file
from PNC_SYN.synthesizer import PNCEventLogSynthesizer

palsyn_model = PNCEventLogSynthesizer()
palsyn_model.load("experiments/models/LSTM_Road_Fines_Short_Event_Log_u=16_ep=5")

event_log = palsyn_model.sample(sample_size=5600, batch_size=100)
event_log_xes = pm4py.convert_to_event_log(event_log)

xes_filename = "road_fines_e=inf.xes"
pm4py.write_xes(event_log_xes, xes_filename)
clean_xes_file(xes_filename, xes_filename)

df = pm4py.convert_to_dataframe(event_log_xes)
df["time:timestamp"] = df["time:timestamp"].astype(str)
df.to_excel("road_fines_e=inf.xlsx", index=False)
```

## Future Work
Future work will focus on enhancing the algorithm and making it available on PyPI.

## Contribution

We welcome contributions from the community. If you have any suggestions or issues, please create a GitHub issue or a pull request. 


## License
This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details. 



## Funding 
This research is funded by the German Federal Ministry of Education and Research (BMBF) and NextGenerationEU (European Union) in the project KI-AIM under the funding code 16KISA115K.

