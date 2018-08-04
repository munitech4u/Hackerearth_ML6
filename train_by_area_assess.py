folds = KFold(n_splits= 3, shuffle=True, random_state=1001)

train_df['label']=train_df['damage_grade'].map(d)
ls=list(train_df['area_assesed'].unique())
ls_pred={}

d1={0:'Grade 1',1:'Grade 2',2:'Grade 3',3:'Grade 4',4:'Grade 5'}

sub1=pd.DataFrame()

ls=['Building removed','Exterior','Not able to inspect','Interior']

for i in range(len(ls)):
    train_df_tmp=train_df[train_df.area_assesed==ls[i]]
    test_df_tmp=test_df[test_df.area_assesed==ls[i]]
    ls_pred[i] = np.zeros([test_df_tmp.shape[0],5])

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df_tmp[v1], train_df_tmp['label'])):
        train_x, train_y = train_df[v1].iloc[train_idx], train_df['label'].iloc[train_idx]
        valid_x, valid_y = train_df[v1].iloc[valid_idx], train_df['label'].iloc[valid_idx]
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
    
        ls_pred[i] += clf.predict_proba(test_df_tmp[v1], num_iteration=clf.best_iteration_)/ folds.n_splits
      
    predictions = []

    for x in ls_pred[i]:
        predictions.append(np.argmax(x))
    
    sub=pd.DataFrame({'building_id':test_df_tmp['building_id'],'grade':predictions})
    sub['damage_grade']=sub['grade'].map(d1)
    sub1=sub1.append(sub)    

sub1[['building_id','damage_grade']].to_csv("sub.csv",index=False)
    