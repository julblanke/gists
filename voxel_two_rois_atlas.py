""" CodeSnippet to create DataFrame/csv containing voxel of two chosen ROIs of chosen atlas

parameters to set:
    atlas
    input_dir
    ROI1 and ROI2 (in neuro_branch)
    output_dir
    output_filename

returns:
    saves .csv in output_dir, containing nifti path and corresponding voxels of rois as 1-dim data
"""

import os
import pandas as pd
from photonai_neuro import NeuroBranch
from photonai.base import PipelineElement
from photonai_neuro.brain_atlas import AtlasLibrary


def voxel_two_rois_atlas():

    # print available atlases
    print("Available atlases: \n {}\n".format(AtlasLibrary.ATLAS_DICTIONARY.keys()))

    # set atlas and print available rois of atlas
    atlas = 'AAL'
    print("Available ROIs of atlas {}: \n {}\n".format(atlas, AtlasLibrary().list_rois(atlas=atlas)))

    # set input directory
    input_dir = 'path_to_input'

    # get full path of niftis in input directory
    nifti_filenames = os.listdir(input_dir)
    nifti_full_paths = [os.path.join(input_dir, nifti) for nifti in nifti_filenames]

    # Add neuro elements
    neuro_branch = NeuroBranch('Neuro', nr_of_processes=1)
    neuro_branch += PipelineElement('BrainAtlas',
                                    hyperparameters={},
                                    atlas_name=atlas,
                                    rois=['Hippocampus_L', "Hippocampus_R"],
                                    extract_mode='vec',
                                    batch_size=50)

    # transform niftis and get data
    X, _, _ = neuro_branch.transform(nifti_full_paths)

    # create dataframe
    df = pd.DataFrame(X)
    df.insert(0, "nifti_path", nifti_full_paths)

    # write dataframe to csv
    output_dir = 'path_to_output'
    output_filename = 'output.csv'
    df.to_csv(os.path.join(output_dir, output_filename), index=False)

if __name__ == '__main__':
    voxel_two_rois_atlas()
