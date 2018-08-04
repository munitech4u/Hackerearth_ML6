comb=pd.read_csv("comb.csv")

flots=[col for col in comb.columns if comb[col].dtype == 'float64']
ints=[col for col in comb.columns if comb[col].dtype == 'int64']

for i in range(len(flots)):
    comb[flots[i]]=comb[flots[i]].astype('float32')
    
for i in range(len(ints)):
    comb[ints[i]]=comb[ints[i]].astype('int32')
    
cats = [col for col in comb.columns if comb[col].dtype == 'object' and col not in['building_id','damage_grade']]
cats=cats+['district_id','vdcmun_id','ward_id','has_repair_started']

for i in range(len(cats)):
    comb[cats[i]]=comb[cats[i]].astype('category')
    
train_df = comb.iloc[:i_len]
test_df = comb.iloc[i_len:]
del comb
d={'Grade 1':0,'Grade 2':1,'Grade 3':2,'Grade 4':3,'Grade 5':4}
label=train_df['damage_grade'].map(d)
gc.collect()
train_df['label']=train_df['damage_grade'].map(d)

v1=list(set(train_df.columns)-set(['building_id','damage_grade','index','label']))

train_df1=train_df[:round(i_len*0.7)]
train_df2=train_df[round(i_len*0.7):]
del train_df
gc.collect()

folds = KFold(n_splits= 5, shuffle=True, random_state=1001)

pred_train = np.zeros([train_df2.shape[0],5])
pred_test = np.zeros([test_df.shape[0],5])

i=0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df1[v1], train_df1['label'])):
    train_x, train_y = train_df1[v1].iloc[train_idx], train_df1['label'].iloc[train_idx]
    valid_x, valid_y = train_df1[v1].iloc[valid_idx], train_df1['label'].iloc[valid_idx]
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

    pred_train += clf.predict_proba(train_df2[v1], num_iteration=clf.best_iteration_)/ folds.n_splits
    pred_test += clf.predict_proba(test_df[v1], num_iteration=clf.best_iteration_)/ folds.n_splits


with open('lgb_prob_all_lvl2_train.pkl', 'wb') as output:
    pickle.dump(pred_train, output)
    
with open('lgb_prob_all_lvl2_test.pkl', 'wb') as output:
    pickle.dump(pred_test, output)
 