from scipy.sparse import csr_matrix

cats = [col for col in comb.columns if comb[col].dtype == 'object' and col not in['building_id','damage_grade']]
cats=cats+['district_id','vdcmun_id','has_repair_started']

comb = pd.get_dummies(comb, columns= cats, dummy_na=True)

train_df = comb.iloc[:i_len]
test_df = comb.iloc[i_len:]

v1=list(set(train_df.columns)-set(['building_id','damage_grade','index','ward_id']))

d={'Grade 1':0,'Grade 2':1,'Grade 3':2,'Grade 4':3,'Grade 5':4}
label=train_df['damage_grade'].map(d)
gc.collect()

X_train=train_df[v1].fillna(0)
X_test=test_df[v1].fillna(0)

from sklearn.ensemble import ExtraTreesClassifier
model_et=ExtraTreesClassifier(verbose=10,max_depth=7,n_jobs=4, min_samples_split=50, min_samples_leaf=20,random_state=123,n_estimators=1000)
model_et.fit(X_train,label)
prob_et=model_et.predict_proba(X_test)




from sklearn.ensemble import RandomForestClassifier
model_rfm= RandomForestClassifier(max_depth=8, random_state=1,min_samples_leaf=50,min_samples_split=20,n_estimators=1000)
model_rfm.fit(X_train, label)
prob_rfm=model_rfm.predict_proba(X_test)


#########################predictions
predictions = []
for x in prob_knn:
    predictions.append(np.argmax(x))
    
sub=pd.DataFrame({'building_id':test_df['building_id'],'grade':predictions})

d1={0:'Grade 1',1:'Grade 2',2:'Grade 3',3:'Grade 4',4:'Grade 5'}

sub['damage_grade']=sub['grade'].map(d1)

sub[['building_id','damage_grade']].to_csv("sub.csv",index=False)

#####################knn
comb=pd.read_csv("comb.csv")
cats = [col for col in comb.columns if comb[col].dtype=='object' and col not in['building_id','damage_grade','district_id','vdcmun_id','ward_id']]

comb1 = pd.get_dummies(comb, columns= cats, dummy_na=True)

v4=list(set(comb1.columns)-set(['index','district_id','vdcmun_id','ward_id','g1','g2','g3','g4','g5','label','damage_grade','building_id']))


from sklearn import preprocessing
X_scaled = preprocessing.scale(comb1[v4].fillna(0))

X_train = X_scaled[:i_len]
X_test= X_scaled[i_len:]

d={'Grade 1':0,'Grade 2':1,'Grade 3':2,'Grade 4':3,'Grade 5':4}
label=trn['damage_grade'].map(d)


from sklearn.neighbors import KNeighborsClassifier
model_knn=KNeighborsClassifier(n_neighbors=50)
model_knn.fit(X_train, label) 
prob_knn_test=model_knn.predict_proba(X_test)
prob_knn_train=model_knn.predict_proba(X_train)