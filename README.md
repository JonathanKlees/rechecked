# From Label Error Detection to Correction: A Modular Framework and Benchmark for Object Detection Datasets

This repository contains validated pedestrian annotations for a subset of the KITTI object detection benchmark as well as the code for reproducing the results of the paper. The codebase assumes that the label error detection methods have been applied and the resulting scores are present for the predictions of the object detectors Cascade R-CNN and YOLOX. To do so, train the object detectors on the KITTI dataset as described in our publication and apply the label error detection methods as described in the corresponding publications.

## Comparing original and validated Annotations
To determine the number of label errors, we compare the original annotations with the validated ones. Here, different data configurations are considered regarding object size, consideration of don't care regions as well as the soft label probability threshold for the validated annotations.
Call the script compare_annotations.py, it will prompt you for three inputs -- the soft label probability threshold, a minimal pixel height of objects to consider and whether to consider objects within don't care regions. To recreate the results of our paper, first make sure to run the process_kitty.py script which requires the path to the locally stored KITTI annotations and converts them to a .csv file containing only pedestrian annotations for the validation split. Then, run the script compile_table.py, which calls the compare_annotations.py script with the parameter combinations we evaluated and prints the resulting table.

Images of the identified label errors can be generated with python plot_label_errors.py. Crops around missing or inaccurate annotations are then stored under a newly created directory label_error_imgs/. To save space, we only generated a subsample of label errors, the size of this sample can be changed in the script, default is 12.

## Benchmarking label error detection methods
Evaluate the performance of label error detection methods on our benchmark using the script benchmark_evaluation.py. Make sure to add the predictions of your method as a csv file with the columns [filename, xmin, ymin, xmax, ymax, score] where the filename column corresponds to the KITTI image filenames (e.g. 000123.png) and the score column should contain the scores of your method that indicate the likelihood of this prediction representing a label error (e.g. the probability for the class pedestrian).
Make sure to change the path to your csv file in l.210 the benchmark_evaluation.py script. Also note that predictions need only be made for the validation subset specified by the train_val_split.json in the data directory.
The same evaluation procedure is done in the corresponding benchmark evaluation platform which can be found under https://www.codabench.org/competitions/10054/

## Cost analysis
The main evaluation of our work is done in cost_analysis.py. 
We provide the necessary files for this procedure in data/predictions/all_predictions.pkl. This file contains the predictions of the object detectors together with the scores assigned by the label error detection methods.
The script will run the evaluation procedure for the label error detection methods considered as well as for the different annotation strategies. Figures will be stored under plots/.
