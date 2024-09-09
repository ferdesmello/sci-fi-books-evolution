import pandas as pd
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------
# reading the data
df = pd.read_csv("sci-fi_books_filtered.csv", sep=";")

print(df.info())
print(df.head())



#----------------------------------------------------------------------------------
# General information of the sample of books
#'title', 'author', 'year', 'decade', 'rate', 'ratings', 'genres', 'synopsis', 'url'

book_per_decade = df['decade'].value_counts()
print(book_per_decade)

mean_per_decade = df.groupby('decade')[['rate', 'ratings']].mean()
print(mean_per_decade)

#----------------------------------------------------------------------------------



#----------------------------------------------------------------------------------



#----------------------------------------------------------------------------------
