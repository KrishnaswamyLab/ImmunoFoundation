import numpy as np

BIOCHEMICAL_PROPERTIES = [
    "aliphathic_index", "boman", "charge", "hphobic", "hphobic2", "instability", "iso_epoint", "mw", "mz",
    "BLOSUM1", "BLOSUM2", "BLOSUM3", "BLOSUM4", "BLOSUM5", "BLOSUM6", "BLOSUM7", "BLOSUM8", "BLOSUM9", "BLOSUM10",
    "PP1", "PP2", "PP3",
    "F1", "F2", "F3", "F4", "F5", "F6",
    "KF1", "KF2", "KF3", "KF4", "KF5", "KF6", "KF7", "KF8", "KF9", "KF10",
    "MSWHIM1", "MSWHIM2", "MSWHIM3",
    "E1", "E2", "E3", "E4", "E5",
    "ProtFP1", "ProtFP2", "ProtFP3", "ProtFP4", "ProtFP5", "ProtFP6", "ProtFP7", "ProtFP8",
    "SV1", "SV2", "SV3", "SV4",
    "ST1", "ST2", "ST3", "ST4", "ST5", "ST6", "ST7", "ST8",
    "T1", "T2", "T3", "T4", "T5",
    "VHSE1", "VHSE2", "VHSE3", "VHSE4", "VHSE5", "VHSE6", "VHSE7", "VHSE8",
    "Z1", "Z2", "Z3", "Z4", "Z5",
    "Foreignness_Score", "Dissimilarity_Score", "quant_foreign", "smoothed_foreign",
    "master_property_score", "Mprop1", "Mprop2", "sasa_af", "smoothed_af_sasa"
]

def extract_biochemical_properties(csv_row):
    return csv_row[BIOCHEMICAL_PROPERTIES].values.astype(np.float32)
