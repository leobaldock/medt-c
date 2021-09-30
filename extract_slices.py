import random
import argparse
import threading
import queue
import os
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pathlib
from dotenv import load_dotenv

parser = argparse.ArgumentParser()
parser.add_argument('--slices', type=int, default=16, help="Number of slices to take from the scan. Default=16.")
parser.add_argument('--view', type=str, default='sagittal', help="The view to take. Default=sagittal.")
parser.add_argument('--singles', default=False, action='store_true', help="Whether to take only one scan per subject. Default=False.")
parser.add_argument('--balance', default=False, action='store_true', help="Whether to balance the data by class. Default=False.")
args = parser.parse_args()
load_dotenv()

########################
# Constants
########################
OFFSET = 0
NUM_SLICES = args.slices
SAVE_PATH = f"./data/adni_sliced_{args.view}"
########################

DIRECTORY = SAVE_PATH
try:
    os.mkdir(DIRECTORY)
except FileExistsError as FEE:
    print(f"Directory {DIRECTORY} already exists.")

data_path = os.getenv("ADNI_RDM_PATH")
metadata_file = open("./data/meta_data_with_clinical_data.json")
metadata = json.load(metadata_file)
metadata_file.close()

# Remove the bad image.
del metadata['905359']

# Firstly, filter out MCI samples, we just want CN and AD
subset = {}
for key in metadata:
    if metadata[key]['label'] != 1:
        # Valid CN or AD
        subset[key] = metadata[key]

# Then group each scan by subject and count scans per subject and their label.
totals = {0: 0, 2: 0}
subjects = {}
for key in subset:
    if subset[key]['subject'] not in subjects:
        subjects[subset[key]['subject']] = {
            'label': subset[key]['label'],
            'count': 1
        }
        totals[subset[key]['label']] += 1

    elif not args.singles:
        totals[subset[key]['label']] += 1
        subjects[subset[key]['subject']]['count'] += 1

print("Total CN:", totals[0], "Total AD:", totals[2])

# Split the data into training and test sets.
test_cn = totals[0] // 5
test_ad = totals[2] // 5
test_subjects = []

train_cn = totals[0] - test_cn
train_ad = totals[2] - test_ad
train_subjects = []
for subject, data in subjects.items():
    if data['label'] == 0:
        # CN
        if train_cn//10 > test_cn:
            test_subjects.append(subject)
            test_cn += data['count']
        else:
            train_subjects.append(subject)
            train_cn += data['count']

    if data['label'] == 2:
        # AD
        if train_ad//10 > test_ad:
            test_subjects.append(subject)
            test_ad += data['count']
        else:
            train_subjects.append(subject)
            train_ad += data['count']

print("Total Subjects:", len(subjects), "Test Subjects:", len(test_subjects), "Train Subjects", len(train_subjects))
print(train_cn, train_ad, test_cn, test_ad)
print(train_cn + train_ad + test_cn + test_ad)

# Keep track of which subjects we already have a scan for so we can skip them if singles is set.
sampled_subjects = set()

# Make slices and save to file.
for i, key in enumerate(tqdm(subset)):
    # If we want only one per subject and we have one, then skip.
    if args.singles and subset[key]['subject'] in sampled_subjects:
        continue

    # Find out whether this is a test or training subject.
    subpath = 'test' if subset[key]['subject'] in test_subjects else 'train'
    subpath = os.path.join(subpath, 'cn' if subset[key]['label'] == 0 else 'ad')
    path = os.path.join(SAVE_PATH, subpath)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
    nifti_image = nib.load(os.path.join(data_path, subset[key]['masked']))
    nii_data = nifti_image.get_fdata()
    nii_data = np.rot90(nii_data, 1, (1, 2))

    # Slice the image from the centre.
    midpoint = nii_data.shape[0] // 2
    slices = None
    if args.singles:
        slices = nii_data[midpoint]
        slices = slices[np.newaxis, ...]
    else:
        slices = nii_data[midpoint-NUM_SLICES//2:midpoint+NUM_SLICES//2,:,:]

    # Save the slices as separate images.
    for i, s in enumerate(slices):
        slice_image = nib.Nifti1Image(s, np.eye(4))
        filename = os.path.join(path, f"{key}_{i}")
        slice_image.to_filename(filename)

    # Record that we have sampled this subject already.
    sampled_subjects.add(subset[key]['subject'])