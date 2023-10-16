#!/bin/bash

Anode_output_dir=../data/extracted_data/230412_Anode
Cathode_output_dir=../data/extracted_data/230330_Cathode
Anode_files=../data/230412_Anode/*.csv
Cathode_files=../data/230330_Cathode/*.csv

for file in $Anode_files
do
    python extract_rule_based_features.py --filepath $file --output_dir $Anode_output_dir
done

for file in $Cathode_files
do
    python extract_rule_based_features.py --filepath $file --output_dir $Cathode_output_dir
done