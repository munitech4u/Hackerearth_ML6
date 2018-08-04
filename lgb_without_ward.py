folds = KFold(n_splits= 5, shuffle=True, random_state=1001)

sub_preds_new = np.zeros([test_df.shape[0],5])

v2=list(set(v1)-set(['ward_id','g1','g2','g3','g4','g5']))
i=0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[v2], train_df['label'])):
    train_x, train_y = train_df[v2].iloc[train_idx], train_df['label'].iloc[train_idx]
    valid_x, valid_y = train_df[v2].iloc[valid_idx], train_df['label'].iloc[valid_idx]
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

    sub_preds_new += clf.predict_proba(test_df[v2], num_iteration=clf.best_iteration_)/ folds.n_splits
    
with open('lgb_prob_without_ward.pkl', 'wb') as output:
    pickle.dump(sub_preds_new, output)

with open(r"catboost_prob.pkl", "rb") as input_file:
           prob_catb = pickle.load(input_file)
    
avg_prob=0.5*sub_preds+0.20*sub_preds_new+0.30*prob_catb

predictions = []

for x in avg_prob:
    predictions.append(np.argmax(x))
    
sub=pd.DataFrame({'building_id':test_df['building_id'],'grade':predictions})
d1={0:'Grade 1',1:'Grade 2',2:'Grade 3',3:'Grade 4',4:'Grade 5'}

sub['damage_grade']=sub['grade'].map(d1)
sub[['building_id','damage_grade']].to_csv("sub.csv",index=False)