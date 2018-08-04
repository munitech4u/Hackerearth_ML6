#with open(r"model_catb.pkl", "rb") as input_file:
#           catb_full = pickle.load(input_file)
#
comb=pd.read_csv("comb.csv")

i_len=631761

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

train_df1=train_df[:round(i_len*0.7)]
train_df2=train_df[round(i_len*0.7):].reset_index()
del train_df
v1=list(set(train_df1.columns)-set(['building_id','damage_grade','index']))
         
catb_prob_full_lvl2_train = catb_full.predict_proba(train_df2[v1])
catb_prob_full_lvl2_test = catb_full.predict_proba(test_df[v1])

with open(r"lgb_prob_all_lvl2_train.pkl", "rb") as input_file:
           lgb_all_train = pickle.load(input_file)
with open(r"lgb_prob_all_lvl2_test.pkl", "rb") as input_file:
           lgb_all_test = pickle.load(input_file)
           

with open(r"lgb_prob_munc_lvl2_train.pkl", "rb") as input_file:
           lgb_munc_train = pickle.load(input_file)
with open(r"lgb_prob_munc_lvl2_test.pkl", "rb") as input_file:
           lgb_munc_test = pickle.load(input_file)
           
with open(r"lgb_prob_ward_lvl2_train.pkl", "rb") as input_file:
           lgb_ward_train = pickle.load(input_file)
with open(r"lgb_prob_ward_lvl2_test.pkl", "rb") as input_file:
           lgb_ward_test = pickle.load(input_file)
           


X=np.hstack((lgb_all_train,lgb_munc_train,lgb_ward_train))

X_train=X[:120000]
X_vald=X[120000:]

train_y=train_df2['label'][:120000]
valid_y=train_df2['label'][120000:]

X_test=np.hstack((lgb_all_test,lgb_munc_test,lgb_ward_test))

lgb_lvl2 = LGBMClassifier(
       boosting_type= 'gbdt',
    objective= 'multiclass',
    num_class=5,
    metric= 'multi_logloss',
    learning_rate= 0.1,
    max_depth= 4,
    num_leaves= 20,
    feature_fraction= 1,
    bagging_fraction= 0.8,
    bagging_freq= 20,
#   min_data_in_leaf=100,
    nthread=4,
    n_estimators=1000)

clf.fit(X_train, train_y, eval_set=[(X_train, train_y), (X_vald, valid_y)], 
        verbose= 100, early_stopping_rounds= 200)

preds = clf.predict(X_test)
sub=pd.DataFrame({'building_id':test_df['building_id'],'grade':preds})

d1={0:'Grade 1',1:'Grade 2',2:'Grade 3',3:'Grade 4',4:'Grade 5'}

sub['damage_grade']=sub['grade'].map(d1)

sub[['building_id','damage_grade']].to_csv("sub.csv",index=False)

from sklearn.linear_model import LogisticRegressionCV

logit=LogisticRegressionCV()

logit.fit(X,train_df2['label'])

preds = logit.predict(X_test)
sub=pd.DataFrame({'building_id':test_df['building_id'],'grade':preds})

d1={0:'Grade 1',1:'Grade 2',2:'Grade 3',3:'Grade 4',4:'Grade 5'}

sub['damage_grade']=sub['grade'].map(d1)

sub[['building_id','damage_grade']].to_csv("sub.csv",index=False)
           
        




