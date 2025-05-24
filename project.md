#### Responsive Bayesian to train:
```
python train_responsive_bayesian.py --csv_path round_frame_data1.csv --output_dir ./saved_models/responsive_bayesian
```
#### Responsive Bayesian to visualize:

```
python responsive_bayesian_viz.py --match_id 03e1f233-579c-462d-ac0e-1635d4718ef8.json --round_idx 2
```

### To see arguments:
```
python train_responsive_bayesian.py --help
```
```
python responsive_bayesian_viz.py --help
```

### To train normal responsive:
```
python train_responsive.py --csv_path round_frame_data1.csv --output_dir ./saved_models/responsive
```

### To visualize normal responsive:
```
python visualize_raw.py --match_id 03e1f233-579c-462d-ac0e-1635d4718ef8.json --round_idx 2 --model_path ./saved_models/responsive/best_model.pt --output_dir ./saved_models/responsive
```