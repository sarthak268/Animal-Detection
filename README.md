# Animal-Detection
<a href="https://drive.google.com/drive/folders/1hAgbxelAX58oHRAXGizAd02cxPuWISCN?usp=sharing">Link for saved weights</a>

- Commands for evaluating the model on heldout test set.

```
mkdir Test_Images
```
All the images will be placed in this folder
```
mkdir Test_Images_Final 
```
This folder will contain all the images after data cleaning and converting the annotations.json to annotations_test.csv.

Keep the annotations.json file in the same directory as the clean_data.py.



- For testing, first clean the test data:
```
python clean_data.py
```

- Command for testing Baseline Model:
```
python test_baseline.py --test_anno_file annotations_test.csv --type baseline
```

- Command for testing Baseline Model with Ensemble Approach:
```
python test_baseline.py --test_anno_file annotations_test.csv --type baseline_ensemble
```

- Command for testing Baseline Model with Weighted Random Sampling:
```
python test_improved.py --test_anno_file annotations_test.csv --type improved
```

- Command for testing Baseline Model with Weighted Random Sampling and Ensemble Approach: 
```
python test_improved.py --test_anno_file annotations_test.csv --type improved_ensemble
```

- Command for visualising the Baseline Model: 
```
python visualize.py --dataset csv --csv_classes classname2id.csv --csv_val annotations_test.csv --model <saved_model_pth>
```

- Command for visualising results with Weighted Random Sampling: 
```
python visualize.py --dataset csv --csv_classes classname2id.csv --csv_val annotations_test.csv --model <saved_model_pth>
```

- Command for visualising results with Ensemble Approach:
```
python visualize.py --dataset csv --csv_classes classname2id.csv --csv_val annotatoins_test.csv --model <saved_model_pth>
```



### Team 
- Anish Madan
- Sarthak Bhagat
- Shagun Uppal

