# Animal-Detection

### Commands for evaluating the model on heldout test set.

- mkdir Test_Images

All the images will be placed in this folder.
- mkdir Test_Images_Final 
This folder will contain all the images after data cleaning and converting the annotations.json to annotations_test.csv.

Keep the annotations.json file in the same directory as the final_test_script.py.


- Saved weights for baseline : baseline.pt

#### Command for visualising the Baseline Model: 
```
python visualize.py --dataset csv --csv_classes classname2id.csv  --csv_val annotations_test.csv --model basline.pt
```

- Saved weights with weighted random sampling: weighted.pt

#### Command for visualising results with Weighted Random Sampling: 
```
python visualize.py --dataset csv --csv_classes classname2id.csv  --csv_val annotationtest.csv --model weighted.pt
```

- Saved weights with ensemble approach: 

#### Command for visualising results with Ensemble Approach:






