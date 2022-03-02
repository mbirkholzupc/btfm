# Script to create all datasets
python create_dataset.py -s train
mv dataset.json dataset_train.json
python create_dataset.py -s val
mv dataset.json dataset_val.json
# MPII and COCO have no test annotations (technically LSPET doesn't either)
python create_dataset.py -l -e -p -f -b -s test
mv dataset.json dataset_test.json
