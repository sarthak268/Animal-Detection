# Animal-Detection

### Commands for evaluating the model on heldout test set.

- mkdir Test_Images

All the images will be placed in this folder, along with the annotations file
- mkdir Test_Images_Final 
This folder will contain all the images after data cleaning and converting the annotations.json to annotations_test.csv.

Keep the annotations.json file in the same directory as the final_test_script.py.


- Saved weights for baseline : baseline.pt

#### Command for testing the baseline model: 
```
python visualize.py --dataset csv --csv_classes classname2id.csv  --csv_val annotations_test.csv --model basline.pt
```
- Saved weights for significant change 1: weighted.pt

#### Command for testing first significant change: 
```
python visualize.py --dataset csv --csv_classes classname2id.csv  --csv_val annotationtest.csv --model weighted.pt
```
- Saved weights for significant change 2: 
#### Command for testing second significant change:






