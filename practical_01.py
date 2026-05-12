# PRACTICAL NO - 01
# Aim: Perform the following operations using Python on any open source dataset (e.g., data.csv)

# import the pandas library
import pandas as pd

# ----------------------------------------
# Section 1: Creating Left and Right DataFrames
# ----------------------------------------

left = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
    'subject_id': ['sub1', 'sub2', 'sub4', 'sub6', 'sub5']
})

right = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
    'subject_id': ['sub2', 'sub4', 'sub3', 'sub6', 'sub5']
})

print("Left DataFrame:")
print(left)

print("\nRight DataFrame:")
print(right)

# ----------------------------------------
# Section 2: Spam Example
# ----------------------------------------

# Ordering Spam, egg, Spam, Spam, bacon and Spam (4 more servings of Spam)
spam_amount = 0
spam_amount = spam_amount + 4

if spam_amount > 0:
    print("\nBut I don't want ANY spam!")

viking_song = "Spam " * spam_amount
print(viking_song)

# ----------------------------------------
# Section 3: Grouping Data
# ----------------------------------------

# import the pandas library
import pandas as pd

ipl_data = {
    'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
             'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
    'Rank': [1, 2, 2, 3, 3, 4, 1, 1, 2, 4, 1, 2],
    'Year': [2014, 2015, 2014, 2015, 2014, 2015, 2016, 2017, 2016, 2014, 2015, 2017],
    'Points': [876, 789, 863, 673, 741, 812, 756, 788, 694, 701, 804, 690]
}

df = pd.DataFrame(ipl_data)
grouped = df.groupby('Year')
print("\nGrouped data for Year 2014:")
print(grouped.get_group(2014))

# ----------------------------------------
# Section 4: Concatenating DataFrames
# ----------------------------------------

import pandas as pd

one = pd.DataFrame({
    'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
    'subject_id': ['sub1', 'sub2', 'sub4', 'sub6', 'sub5'],
    'Marks_scored': [98, 90, 87, 69, 78]
}, index=[1, 2, 3, 4, 5])

two = pd.DataFrame({
    'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
    'subject_id': ['sub2', 'sub4', 'sub3', 'sub6', 'sub5'],
    'Marks_scored': [89, 80, 79, 97, 88]
}, index=[1, 2, 3, 4, 5])

print("\nConcatenated DataFrame:")
print(pd.concat([one, two]))
