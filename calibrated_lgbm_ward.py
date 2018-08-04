from sklearn.calibration import CalibratedClassifierCV

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
    comb[cats[i]],x=pd.factorize(comb[cats[i]])
    comb[cats[i]]=comb[cats[i]].astype('category')
i_len=631761
    
train_df = comb.iloc[:i_len]
test_df = comb.iloc[i_len:]
del comb
d={'Grade 1':0,'Grade 2':1,'Grade 3':2,'Grade 4':3,'Grade 5':4}
label=train_df['damage_grade'].map(d)
gc.collect()
train_df['label']=train_df['damage_grade'].map(d)

v1=list(set(train_df.columns)-set(['building_id','damage_grade','index','label','ward_id']))

folds = KFold(n_splits= 5, shuffle=True, random_state=1001)

train_df['label']=train_df['damage_grade'].map(d)
sub_preds = np.zeros([test_df.shape[0],5])

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
    nthread=4,
    n_estimators=5000)
    
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            verbose= 100, early_stopping_rounds= 100)
    calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
    calibrated_clf.fit(valid_x, valid_y)
    sub_preds += calibrated_clf.predict_proba(test_df[v1])/ folds.n_splits


    
with open('lgb_prob_calibrated_ward.pkl', 'wb') as output:
    pickle.dump(sub_preds, output)
    
predictions = []

for x in sub_preds:
    predictions.append(np.argmax(x))
    
sub=pd.DataFrame({'building_id':test_df['building_id'],'grade':predictions})
d1={0:'Grade 1',1:'Grade 2',2:'Grade 3',3:'Grade 4',4:'Grade 5'}

sub['damage_grade']=sub['grade'].map(d1)
sub[['building_id','damage_grade']].to_csv("sub.csv",index=False)






