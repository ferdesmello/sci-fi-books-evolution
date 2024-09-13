import pandas as pd
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------
# reading the data
df = pd.read_csv("./Data/sci-fi_books_FILTERED.csv", sep=";")
df_200 = pd.read_csv("./Data/top_sci-fi_books_200_PER_DECADE.csv", sep=";")

#print(df.info())
#print(df.head())

#----------------------------------------------------------------------------------
# General information of the FILTERED sample of books
#'title', 'author', 'year', 'decade', 'rate', 'ratings', 'genres', 'synopsis', 'review', 'url'
print("\nFILTERED books.")

book_per_decade = df['decade'].value_counts()
print(book_per_decade)

mean_per_decade = df.groupby('decade')[['rate', 'ratings']].mean()
print(mean_per_decade)

#----------------------------------------------------------------------------------
# General information of the 200 PER DECADE sample of books
#'title', 'author', 'year', 'decade', 'rate', 'ratings', 'genres', 'synopsis', 'review', 'url'
print("\n200 PER DECADE books.")

book_per_decade = df_200['decade'].value_counts()
print(book_per_decade)

mean_per_decade = df_200.groupby('decade')[['rate', 'ratings']].mean()
print(mean_per_decade)

#----------------------------------------------------------------------------------



#----------------------------------------------------------------------------------
