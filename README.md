# Hackerearth_ML6
Hackerearth machine learning challenge - 6 (https://www.hackerearth.com/challenge/competitive/machine-learning-challenge-6-1/), 
3rd rank solution (https://www.hackerearth.com/challenge/competitive/machine-learning-challenge-6-1/leaderboard/)

Performance of models:

|   Model          |   Score   |   calibrated using sklearn |
| ---------------- | --------- | -------------------------- | 
| catboost         |   0.7488  |                            |
| lgbm             |   0.76723 |   0.7674                   |
| etree            |   0.6202  |                            |
| knn              |   0.5514  |                            |
| lgb_without_munc |   0.76663 |   0.76686                  |
| lgb_without_ward |   0.7662  |                            |
| catb_without_munc|   0.76201 |                            |
| catb_averaged    |   0.76558 |                            |
| nnet             |   0.7355  |                            |
| catb_less_feat   |   0.76821 |   0.76302                  |


Best_score=prob_lgb**0.3+prob_lgb_munc**0.2+prob_lgb_ward**0.1+prob_catb_less_feat**0.4 

0.76978
