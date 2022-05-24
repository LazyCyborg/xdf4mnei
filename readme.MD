# Load XDF files into MNE

MNE cannot currently load xdf files, and the tool that allows for it (MNE-python) does not automatize this step. You MUST use the GUI, which is undesirable
This tool allows for loading any xdf files containing EEG into MNE, with markers converted as annotations.

It is based (copied and tweaked) from the implementation of MNELab

Developpers of MNE and MNELab brushed away my suggestion then my pull request and leave me no choice than making this crappy repository

## Installation

via pip `pip install xdf4mne`

via github `pip install .`

## Usage

```python
from xdf4mne.xdf import read_raw_xdf

# load the file
eeg_file = "./sample.xdf"
raw = read_raw_xdf(fname=eeg_file)
# display signal
#raw.plot(scalings="auto")

raw
```

