'''
I updated some of the string comparisons to byte, but Im not sure exactly
which to change. Might work, if you get a unicode error or everything is
empty that might be the reason...~MJT
'''
from __future__ import absolute_import
import re
import datetime as dt
import numpy as np
import pandas as pd
from .core import load_events, load_fs, load_info


class FindEnd():

    def __init__(self):
        self.keep = True

    def check(self, code):
        if code in '()FfTt]N>#':
            self.keep = False
        return self.keep


def get_stim_start(stim_end_row, digmarks):
    '''
    Finds the digmark row corresponding to the beginning of a stimulus

    Parameters
    ------
    stim_end_row : pandas dataframe
        The row of the digmark dataframe corresponding to the end of a stimulus
    digmarks : pandas dataframe
        The digmark dataframe

    Returns
    ------
    this_trial : pandas dataframe
        Row containing the digmark corresponding to the start of the stimulus

    '''
    rec, ts = stim_end_row['recording'], stim_end_row['time_samples']
    mask = (
        (digmarks['recording'] == rec)
        & (digmarks['time_samples'] < ts)
        & ~digmarks['codes'].str.contains('[RCL]')
    )
    this_trial_mask = (
        digmarks[mask].iloc[::-1]['codes'].apply(FindEnd().check).iloc[::-1]
        & digmarks[mask]['codes'].str.contains('<')
    )
    this_trial = digmarks[mask][this_trial_mask]
    return this_trial.iloc[0]


def _is_not_floatable(arg):
    ''' returns True if arg cannot be converted to float
    '''
    try:
        float(arg)
        return False
    except ValueError:
        return True


def get_stim_info(trial_row, stimulus, fs):
    '''
    finds the stimulus info for a trial.

    Parameters
    -------
    trial_row
        row from a trial
    stimulus
        pandas dataframe of all stimulus events
    fs : float
        sampling rate of block

    Returns
    -------
    digmark row for the response event

    '''
    rec, samps = trial_row['recording'], trial_row['time_samples']
    stim_mask = (
        (stimulus['recording'] == rec)
        & (stimulus['time_samples'] > (samps - 1.0 * fs))
        & (stimulus['time_samples'] < (samps + fs))
    )

    if stim_mask.sum() > 0:
        return stimulus[stim_mask].iloc[0]
    else:
        return dict(codes=np.nan, time_samples=np.nan, recording=np.nan, text=np.nan)


def get_stim_end(trial_row, digmarks, fs, window=75.0):
    '''
    finds the end of the stimulus event for a trial.

    Parameters
    -------
    trial_row
        row from a trial
    digmarks
        pandas dataframe of all digmark events
    fs : float
        sampling rate of block
    window : float
        time window (in seconds) after the stimulus start in which to look for
        the stimulus end. default: 60.0

    Returns
    -------
    digmark row for the response event

    '''
    rec, samps = trial_row['recording'], trial_row['time_samples']
    # print((rec, samps))
    resp_mask = (
        (digmarks['recording'] == rec)
        & (digmarks['time_samples'] > samps)
        & (digmarks['time_samples'] < (samps + fs * window))
        & digmarks['codes'].str.contains('[>#]')
    )
    # print(digmarks[resp_mask].shape)
    assert digmarks[resp_mask].shape[0] > 0
    return digmarks[resp_mask].iloc[0]


def get_response(trial_row, digmarks, fs, window=5.0):
    '''
    finds the response event for a trial.

    Parameters
    -------
    trial_row
        row from a trial
    digmarks
        pandas dataframe of all digmark events
    fs : float
        sampling rate of block
    window : float
        time window (in seconds) after the stimulus end in which to look for
        the response. default: 5.0

    Returns
    -------
    digmark row for the response event

    '''
    rec, samps = trial_row['recording'], trial_row['time_samples']
    try:
        stim_dur = trial_row['stimulus_end'] - trial_row['time_samples']
    except KeyError:
        stim_dur = get_stim_end(rec, samps, fs)['time_samples'] - samps
    resp_mask = (
        (digmarks['recording'] == rec)
        & (digmarks['time_samples'] > (samps + stim_dur))
        & (digmarks['time_samples'] < (samps + stim_dur + fs * window))
        & digmarks['codes'].str.contains('[RLN]')
    )
    if digmarks[resp_mask].shape[0] > 0:
        return digmarks[resp_mask].iloc[0]
    else:
        return dict(codes=np.nan, time_samples=np.nan, recording=np.nan)


def get_consequence(trial_row, digmarks, fs, window=2.0):
    '''
    finds the consequence event for a trial.

    Parameters
    -------
    trial_row
        row from a trial
    digmarks
        pandas dataframe of all digmark events
    fs : float
        sampling rate of block
    window : float, optional
        time window (in seconds) after the reponse in which to look for the
        consequence. default: 2.0

    Returns
    -------
    digmark row for the consequence event

    '''
    rec, samps = trial_row['recording'], trial_row['time_samples']
    rt = trial_row['response_time']
    bds = rt, rt + fs * window
    resp_mask = (
        (digmarks['recording'] == rec)
        & (digmarks['time_samples'] > bds[0])
        & (digmarks['time_samples'] < bds[1])
        & digmarks['codes'].str.contains('[FfTt]')
    )
    if digmarks[resp_mask].shape[0] > 0:
        return digmarks[resp_mask].iloc[0]
    else:
        return dict(codes=np.nan, time_samples=np.nan, recording=np.nan)


def is_correct(consequence):
    '''
    Checks if the consequence indicates that the trial was correct.
    '''
    try:
        return consequence in 'Ff'
    except TypeError:
        return consequence


def calc_rec_datetime(file_origin, start_time):
    '''
    calculates the datetime of recording from the probe-the-broab mat export
    filename and the timestamp of where in the file the recording started

    Parameters
    -----
    file_origin : str
        the mat file (e.g. `SubB997Pen01Site04Epc02File01_10-25-15+12-46-08_B997_block.mat`)
    start_time : float
        the start time of the recording in seconds (e.g. 4.8e-05)

    Returns
    -----
    rec_datetime : datetime
        the datetime of the recording

    '''
    datetime_str = re.search('_([0-9\-\+]+)_', file_origin).groups()[0]
    datetime = dt.datetime.strptime(datetime_str, '%m-%d-%y+%H-%M-%S')
    return datetime + dt.timedelta(seconds=start_time)


class FindCorrectionTrials():

    def __init__(self):
        self.correction = False

    def check(self, row):
        if self.correction:
            if row['correct'] == True:
                self.correction = False
            return True
        else:
            if row['correct'] == False:
                self.correction = True
            return False


def load_trials(block_path):
    '''
    returns a pandas dataframe containing trial information for a given block_path

    Parameters
    -------
    block_path : str
        the path to the block

    Returns
    ------
    trials : pandas dataframe

    Columns
    ------
    time_samples : int
        Time in samples of the start of a stimulus (trial)
    stimulus : str
        Name of the stimulus
    stimulus_end : int
        Time in samples of the end of the stimulus
    response : str
        Response code of the animal
    response_time : int
        Time in samples of the response of the animal
    consequence : str
        Consequence code
    correct : bool
        Whether the trial was correct or not

    '''
    digmarks = load_events(block_path, 'DigMark')
    digmarks['codes'] = digmarks.apply(lambda row: row['codes'].decode(), axis=1)
    digmarks = digmarks[digmarks['codes'] != b'C']
    stimulus = load_events(block_path, 'Stimulus')
    stimulus['text'] = stimulus.apply(lambda row: row['text'].decode(), axis=1)

    stim_mask = (
        ~(stimulus['text'].astype(str).str.contains('date'))
        & (stimulus['text'].apply(_is_not_floatable))  # occlude floats
    )
    stimulus = stimulus[stim_mask]
    fs = load_fs(block_path)
    info = load_info(block_path)

    stim_end_mask = digmarks['codes'].isin(('>', '#'))
    trials = digmarks[stim_end_mask].apply(lambda row: get_stim_start(row, digmarks), axis=1)[:]
    trials.reset_index(inplace=True)
    del trials['index']
    del trials['codes']
    trials['stimulus'] = trials.apply(lambda row: get_stim_info(row, stimulus, fs)['text'], axis=1)
    trials['stimulus_end'] = trials.apply(
        lambda row: get_stim_end(row, digmarks, fs)['time_samples'], axis=1)
    trials['response'] = trials.apply(lambda row: get_response(row, digmarks, fs)['codes'], axis=1)
    trials['response_time'] = trials.apply(
        lambda row: get_response(row, digmarks, fs)['time_samples'], axis=1)
    trials['consequence'] = trials.apply(
        lambda row: get_consequence(row, digmarks, fs)['codes'], axis=1)
    trials['correct'] = trials['consequence'].apply(is_correct)

    def trial_time(row):
        file_origin = info['recordings'][row['recording']]['file_origin']
        start_time = info['recordings'][row['recording']]['start_time']
        return calc_rec_datetime(file_origin, start_time) + dt.timedelta(seconds=row['time_samples'] / fs)

    trials['datetime'] = trials.apply(trial_time, axis=1)
    trials.sort_values('datetime', inplace=True)
    trials['correction'] = trials.apply(FindCorrectionTrials().check, axis=1)
    return trials

TRIAL_CHANNEL = 0

def oe_load_trials(block_path):
    
    ttls = load_events(block_path, 'TTL')
    stimuli = load_events(block_path, 'Stimulus')

    trial_starts = ttls[(ttls.channel == TRIAL_CHANNEL) & (ttls.eventID==1)]['time_samples'].values
    trial_ends = ttls[(ttls.channel == TRIAL_CHANNEL) & (ttls.eventID==0)]['time_samples'].values
    if len(trial_starts) > len(trial_ends):
        trial_ends = np.append(trial_ends, trial_starts[-1]+181000)
    stims = [x.decode('utf8') for x in stimuli['text'].values]
    time_samples = stimuli['time_samples'].values
    stimulus_end = stimuli['stimulus_end'].values  
    mvl = np.amin([len(x) for x in (trial_starts, trial_ends, stims, time_samples, stimulus_end)])
    trials = pd.DataFrame({'trial_start': trial_starts[:mvl], 'trial_end': trial_ends[:mvl], 'time_samples': time_samples[:mvl], 'stimulus_end': stimulus_end[:mvl], 'stimulus': stims[:mvl]})
    return trials
    
