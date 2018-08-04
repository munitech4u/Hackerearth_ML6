comb=pd.read_csv("comb.csv")
flots=[col for col in comb.columns if comb[col].dtype == 'float64']
ints=[col for col in comb.columns if comb[col].dtype == 'int64']

for i in range(len(flots)):
    comb[flots[i]]=comb[flots[i]].astype('float32')
    
for i in range(len(ints)):
    comb[ints[i]]=comb[ints[i]].astype('int32')

  
cats = [col for col in comb.columns if comb[col].dtype == 'object' and col not in['building_id','damage_grade']]
cats=cats+['district_id','has_repair_started']
i_len=631761

comb = pd.get_dummies(comb, columns= cats, dummy_na=True)

train_df = comb.iloc[:i_len]
test_df = comb.iloc[i_len:]
v1=list(set(train_df.columns)-set(['building_id','damage_grade','index','ward_id','vdcmun_id']))

d={'Grade 1':0,'Grade 2':1,'Grade 3':2,'Grade 4':3,'Grade 5':4}
label=train_df['damage_grade'].map(d)
gc.collect()

#X_train=train_df[v1].fillna(0)
#X_test=test_df[v1].fillna(0)

from sklearn import preprocessing
X_train = preprocessing.scale(train_df[v1].fillna(0))
test_x=preprocessing.scale(test_df[v1].fillna(0))

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split

seed = 7
np.random.seed(seed)


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(label)
encoded_Y = encoder.transform(label)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(150, input_dim=len(v1), activation='relu'))
   #number of classes
	model.add(Dense(5, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=50, verbose=1)

#X_train, X_test, Y_train, Y_test = train_test_split(X_train, dummy_y, test_size=0.33, random_state=seed)
estimator.fit(X_train, dummy_y,validation_split=0.33)
predictions = estimator.predict(test_x)

keras_prob=estimator.predict_proba(test_x)
with open('keras_prob_all.pkl', 'wb') as output:
    pickle.dump(keras_prob, output)

sub=pd.DataFrame({'building_id':test_df['building_id'],'grade':predictions})

d1={0:'Grade 1',1:'Grade 2',2:'Grade 3',3:'Grade 4',4:'Grade 5'}

sub['damage_grade']=sub['grade'].map(d1)

sub[['building_id','damage_grade']].to_csv("sub.csv",index=False)




