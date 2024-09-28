from pyxdf import load_xdf
from mne.utils import verbose, logger, warn
from mne.io import RawArray
from mne import create_info, Annotations
import numpy as np

def read_raw_xdf(fname,
                 name_stream_eeg: str = None,
                 name_stream_markers: str = None,
                 data_type: str = 'EEG',
                 data_type_markers: str = 'Markers',
                 *args, **kwargs):
    """Read XDF file.
    Either specify the name or the stream id.

    Note that it does not recognize different data types in the same stream (e.g., EEG + MISC).

    Parameters
    ----------
    fname : str
        Name of the XDF file.
    name_stream_eeg : str
        Name of the data stream to load (optional).
    name_stream_markers : str
        Name of a specific marker stream to load (optional), otherwise inserts all marker streams.
    data_type : str
        Type of the data stream to load (default: 'EEG').
    data_type_markers : str
        Type of the marker stream to load (default: 'Markers').

    Returns
    -------
    raw : mne.io.Raw
        Raw object from XDF file data.
    """

    # Load the XDF file
    streams, header = load_xdf(fname)

    # Search for the EEG stream
    eeg_stream = None
    for stream in streams:
        info = stream['info']
        if name_stream_eeg:
            if info['name'][0] == name_stream_eeg:
                eeg_stream = stream
                break
        else:
            if info['type'][0] == data_type:
                eeg_stream = stream
                break

    if eeg_stream is None:
        raise ValueError('No EEG stream found')

    # Load EEG data information to compose the info
    n_chans = int(eeg_stream["info"]["channel_count"][0])
    fs = float(eeg_stream["info"]["nominal_srate"][0])
    labels = []
    try:
        for ch in eeg_stream["info"]["desc"][0]["channels"][0]["channel"]:
            labels.append(str(ch["label"][0]))
    except (TypeError, IndexError, KeyError):  # No channel labels found
        pass
    if not labels:
        labels = [f'EEG {n}' for n in range(n_chans)]

    info = create_info(ch_names=labels, sfreq=fs, ch_types=['eeg'] * n_chans)
    data = np.array(eeg_stream["time_series"]).T
    # Create the Raw object
    raw = RawArray(data, info)
    # Manually define the _filenames attribute
    raw._filenames = [fname]

    # Keep the first sample timestamp to align markers
    first_samp = eeg_stream["time_stamps"][0]

    # Find the marker streams
    markers_streams = []
    for stream in streams:
        info = stream['info']
        if name_stream_markers:
            if info['name'][0] == name_stream_markers:
                markers_streams.append(stream)
                break
        else:
            if info['type'][0] == data_type_markers:
                markers_streams.append(stream)

    # Iterate over marker streams
    for marker_stream in markers_streams:
        # Realign the first timestamp to the data
        onsets = marker_stream["time_stamps"] - first_samp
        # Extract description labels
        descriptions = [item[0] if isinstance(item, list) else item for item in marker_stream["time_series"]]
        # Add to the raw as annotations
        durations = [0] * len(onsets)
        annotations = Annotations(onset=onsets, duration=durations, description=descriptions)
        raw.set_annotations(raw.annotations + annotations)

    return raw
