"""
model.labels
============
Canonical label <-> index mappings, shared between preprocessing and
training so nothing drifts out of sync.

MELD uses 7 emotion classes (matches Table 2 of the paper: Neu, Sur, Joy,
Sad, Ang, Fea, Dis). The raw MELD CSVs spell them out in lowercase English
words in the "Emotion" column - this maps those to the paper's class order.
"""

MELD_EMOTIONS = ["neutral", "surprise", "joy", "sadness", "anger", "fear", "disgust"]
MELD_LABEL2IDX = {label: i for i, label in enumerate(MELD_EMOTIONS)}
MELD_IDX2LABEL = {i: label for label, i in MELD_LABEL2IDX.items()}

# IEMOCAP's 4-class setup used in the paper (Neutral, Anger, Happy incl.
# Excited, Sad) - kept here for when you get IEMOCAP access.
IEMOCAP_EMOTIONS = ["neutral", "anger", "happy", "sad"]
IEMOCAP_LABEL2IDX = {label: i for i, label in enumerate(IEMOCAP_EMOTIONS)}
IEMOCAP_IDX2LABEL = {i: label for label, i in IEMOCAP_LABEL2IDX.items()}
