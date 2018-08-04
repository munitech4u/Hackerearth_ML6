# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:29:42 2018

@author: vvootl010
"""

df_pred_lvl1_train=pd.DataFrame(np.hstack((lgb_all_train,lgb_munc_train,lgb_ward_train)))
df_pred_lvl1_test=pd.DataFrame(np.hstack((lgb_all_test,lgb_munc_test,lgb_ward_test)))

train_df3=train_df2.join(df_pred_lvl1_train)
test_df1=test_df.reset_index().join(df_pred_lvl1_test)

folds = KFold(n_splits= 5, shuffle=True, random_state=1001)

sub_preds = np.zeros([test_df1.shape[0],5])
v1=list(set(train_df3.columns)-set(['building_id','damage_grade','index','label']))

i=0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df3[v1], train_df3['label'])):
    train_x, train_y = train_df3[v1].iloc[train_idx], train_df3['label'].iloc[train_idx]
    valid_x, valid_y = train_df3[v1].iloc[valid_idx], train_df3['label'].iloc[valid_idx]
    i+=1
    print ('iteration-%s'%i) 
    clf = clf = LGBMClassifier(
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

    sub_preds += clf.predict_proba(test_df1[v1], num_iteration=clf.best_iteration_)/ folds.n_splits



