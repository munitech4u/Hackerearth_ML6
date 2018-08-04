folds = KFold(n_splits= 5, shuffle=True, random_state=1001)

train_df['label']=train_df['damage_grade'].map(d)
sub_preds = np.zeros([test_df.shape[0],5])
#feature_importance_df = pd.DataFrame()
i=0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[v1], train_df['label'])):
    train_x, train_y = train_df[v1].iloc[train_idx], train_df['label'].iloc[train_idx]
    valid_x, valid_y = train_df[v1].iloc[valid_idx], train_df['label'].iloc[valid_idx]
    i+=1
    print ('iteration-%s'%i) 
    clf = LGBMClassifier(
    boosting_type= 'gbdt',
    objective= 'multiclass',
    num_class=5,
    metric= 'multi_logloss',
    learning_rate= 0.05,
    max_depth= 7,
    num_leaves= 60,
    feature_fraction= 0.7,
    bagging_fraction= 1,
    bagging_freq= 20,
#    min_data_in_leaf=100,
    nthread=4,
    n_estimators=5000)

    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            verbose= 100, early_stopping_rounds= 100)

    sub_preds += clf.predict_proba(test_df[v1], num_iteration=clf.best_iteration_)/ folds.n_splits
#    fold_importance_df = pd.DataFrame()
#    fold_importance_df["feature"] = v1
#    fold_importance_df["importance"] = clf.feature_importances_
#    fold_importance_df["fold"] = n_fold + 1
#    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



    
with open('lgb_prob.pkl', 'wb') as output:
    pickle.dump(sub_preds, output)
    
avg_prob=0.8*sub_preds+0.2*prob_catb

predictions = []

for x in sub_preds:
    predictions.append(np.argmax(x))
    
sub=pd.DataFrame({'building_id':test_df['building_id'],'grade':predictions})
d1={0:'Grade 1',1:'Grade 2',2:'Grade 3',3:'Grade 4',4:'Grade 5'}

sub['damage_grade']=sub['grade'].map(d1)
sub[['building_id','damage_grade']].to_csv("sub.csv",index=False)






