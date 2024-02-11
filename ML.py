# importing train and test data into train_df and test_df dataframes

import pandas as pd
train = pd.read_csv('/kaggle/input/Titanic_Data_Source/train.csv')
test = pd.read_csv('/kaggle/input/Titanic_Data_Source/test.csv')



#In this section we upload the data of the train and test files. On this data, we will make the adjustments as far as possible to achieve the best possible accuracy



# printing train & test data information 
print(train.info())
print('-'*50) # Print a line between the data sets
print(test.info())


#In this section, we will print the data and see the following features that interest us such as: Null values and the type of values for each column in our data.
#By printing operation


# A peek at our data

train.head(n=10)


#In the above operation we can see the columns of the data belonging to the train data file. This is the right way to better understand what we are supposed to do.
#Naturally, because of the nature of our analysis project, we want as much as possible and where possible, to have numerical data to make the learning mechanism more efficient.
#After we see our data for the first time, we will convert the data as much as possible to numeric data. 
#We will start with the transition of the column called Sex to a Binary column. Female will be represented by the number zero, male will be represented by number one.
# convert sex column into number

sex = train['Sex'] # Defines the columns on which we will make the change in the context of the data set
sextest = test['Sex']
new = [] # Define a new list
newtest = []
for i in sex:# A loop that belongs to the train data set
    if(i == 'male'):
        new.append(1)
    else:
        new.append(0)
        
for i in sextest: # A loop that belongs to the test data set
    if(i == 'male'): # If the value is equal to the male
        newtest.append(1) # convert to zero Similar action was taken on the train data set
    else:
        newtest.append(0)
        
train['Sex'] = new # Define existing data for a new list
test['Sex'] = newtest


#In this terms we have defined here two variables called sex that related to the train data and a Sextest variable that related to our test data. In this action, we did direct action to our DATA FRAME in the project. We then defined list type variables called new and new test.
#We then used a standard, loop that would run on the same list and replace the male or female values with values that we want to be zero or one.
#After a successful run, we will show our data again and see what happened after doing that.

# A peek at our data
train.head(n=10)

#As we see the column of sex has changed to the values we wanted and it was performed successfully.
#Let's continue now.
#After looking through our data again we see that we still have a relatively large number of columns that are defined as a string or "object" in the output language here.
#Moving forward to change the column that is relatively easier to change compared to the rest, and called "Embarked", I point out that it holds a Repeating value sequence and will make it easier to classify each of the values into numeric values.


#We classify the values so that:

#Q will become zero S will become one C will become two
#This is how we convert the Embarked column to the number values, again so that the model can better learn our data.
#Here we used the same simple loop method to change the column as we did to the column last time.

# convert Embarked column into number
newEmbarked = [] # Define a new list
newEmbarkedtest = []
Embarked = train['Embarked'] # Defines the columns on which we will make the change in the context of the data set
Embarkedtest = test['Embarked']
for i in Embarked: # A loop that belongs to the train data set
    if(i == 'Q'):
        newEmbarked.append(0)
    elif(i == 'S'):
        newEmbarked.append(1)
    else:
        newEmbarked.append(2)
for i in Embarkedtest: # A loop that belongs to the test data set
    if(i == 'Q'):
        newEmbarkedtest.append(0)
    elif(i == 'S'):
        newEmbarkedtest.append(1)
    else:
        newEmbarkedtest.append(2)
        
train['Embarked'] = newEmbarked # Define existing data for a new list
test['Embarked'] = newEmbarkedtest


#After a successful run of our code, we will review the data again and see the result.

# A peek at our data
train.head(n=10)


#Great We see that the action we wanted was successful.
#Let's move on now

# A peek at our data
train.head(n=10)
After another peek we see that we have Null values, so we will use the following action to see the amount of these values by column.

# Our Null values in the data.
train.isnull().sum()


#We clearly see as the problematic columns in terms of missing values and the only ones, are the columns: Cabin and Age.
#We will use a function given to us by the course lecturer to fill in the Null values in the type of calculation we want, such as: median, mean or standard deviation.
#I'm Selected and left with mean.

# taking care of missing values
def missing(data): # Defines a Python language function with a variable named data
    d = data.copy(deep = True) # Make a copy in memory of the data frame
    for c in data:
        if (data[c].dtype =='int64') or (data[c].dtype =='float64') : 
            if data[c].isnull().values.any(): # If the value is Null, it will give it a value type listed above
                m = data[c].dropna().mean()
                d[c].fillna(m, inplace=True) # Null Value Fill command
        else:          
            if data[c].isnull().values.any():
                m = data[c].dropna().mode()[0]
                d[c].fillna(m, inplace=True)
    return d

trm = missing(train) # Define the data set with a new name
tsm = missing(test)


#After the successful run, we will take a look at our data again.
#We see that the function changed the database and now we will continue with the name trm instead of train and tsm instead of test

# A peek at our data
trm.head(n=10)


#We see that after running, the Null values are gone. We will check again by the same action as last time.
# look on NULL values in the data.

trm.isnull().sum()
Excellent We see that we do not have Null values in the data using the previous action

# printing training data information after missing values treatment
print(trm.info())
print('-'*50)
print(tsm.info())


# A peek at our data
trm.head(n=10)


#After looking at the data and other examples, a column that is a multiplier of other columns should be added for the accuracy of the model, 
#as I see in the data we can do this with age and status on the ship.

# age_class
trm['age_class'] = trm['Age'] * trm['Pclass']
tsm['age_class'] = tsm['Age'] * tsm['Pclass']
Here we define the new column which is a two-column multiplier. Age and Pclass

# A peek at our data
trm.head(n=10)


#After several runs, I see that the relationship of two Multiplied columns is very much reinforcing the accuracy, 
#so I will continue to add more columns of that kind and the next one will be passenger status on bord and the price they paid. Fare and Pclass
#Just great !

# Fare_class 
trm['Fare_class'] = trm['Fare'] * trm['Pclass']
tsm['Fare_class'] = tsm['Fare'] * tsm['Pclass']

#Here we define the new column which is a two-column multiplier. Fare and Pclass

# A peek at our data
trm.head(n=10)

# Age_Fare
trm['Age_Fare'] = trm['Fare'] * trm['Age']
tsm['Age_Fare'] = tsm['Fare'] * tsm['Age']

#Here we define the new column which is a two-column multiplier. Age and Fare

# A peek at our data
trm.head(n=10)

# new Family column for My data
trm['Family'] = trm['SibSp'] + trm['Parch'] 
tsm['Family'] = tsm['SibSp'] + tsm['Parch'] 

#Here we added a family column that we would multiplie with the family status column on board

# A peek at our data
trm.head(n=10)

#Great we see that a family column has been added that counts the amount of relatives on board.

# A family column and her status on board and how much she paid
trm['Family_Pclass_Fare'] = trm['Family'] * trm['Pclass'] * trm['Fare'] 
tsm['Family_Pclass_Fare'] = tsm['Family'] * tsm['Pclass'] * tsm['Fare'] 

#Here we define the new column which is a two-column multiplier. Family and Pclass and Fare

# A peek at our data
trm.head(n=10)


#Once we have added the columns and refined the data for study, we will give up float numbers and continue with double and do this for each column by the following action.
#I do this after realizing that this is what a Gaussian model should get
#After a few runs with double numbers the accuracy just drops so I will go back to int numbers

# Convert from float numbers to int
trm.age_class = trm.age_class.astype('int')
trm.Age = trm.Age.astype('int')
trm.Fare = trm.Fare.astype('int')
trm.Fare_class = trm.Fare_class.astype('int')
trm.Age_Fare = trm.Age_Fare.astype('int')
trm.Family_Pclass_Fare = trm.Family_Pclass_Fare.astype('int')
# A peek at our data
trm.head(n=10)
# preparing training data for submission
cols = ['Pclass','Age','SibSp','Parch','Fare', 'Sex', 'Embarked', 'age_class','Fare_class','Age_Fare','Family','Family_Pclass_Fare']
x_train = trm[cols]
y = trm['Survived']
x_test = tsm[cols]

#Here we add the columns we want our model to learn.

# defining GaussianNB model for My Notebook
from sklearn.naive_bayes import GaussianNB # It is especially used when attributes have continuous values.
m = GaussianNB(priors=None, var_smoothing=1e-09)

#Defining our model for submission.

# scoring GaussianNB model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(m, x_train, y, cv = 15)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# fitting model and building predictions
m.fit(x_train, y)
yy = m.predict(x_test) 


# preparing submission file
submission = pd.DataFrame( { 'PassengerId': test['PassengerId'] , 'Survived': yy } )
submission.to_csv('naive_bayes_GaussianNB_Moshe_Khorshidi.csv' , index = False )
