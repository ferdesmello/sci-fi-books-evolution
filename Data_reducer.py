import pandas as pd

#----------------------------------------------------------------------------------
df = pd.read_csv('sci-fi_books.csv', sep = ';')

print(df.info())

#----------------------------------------------------------------------------------
# Deleting parenthesis from titles
df["title"] = df["title"].str.replace(r' \(.*\)', '', regex = True)

# Sorting by year
df.sort_values(by = ['year'], axis = 0, ascending = True, inplace = True)

#----------------------------------------------------------------------------------
# Converting the genres column to actual lists
for index, row in df.iterrows():
    genres = row['genres']
    # Convert the string representation of lists into actual lists
    if isinstance(genres, str):
        df.at[index, 'genres'] = eval(genres)

#----------------------------------------------------------------------------------
# Grouping by decade

# Create a 'decade' column
# Floor division by 10, then multiply by 10 to get the start of the decade
df['decade'] = (df['year'] // 10) * 10 

# Group by decade and sort within each group by 'ratings' in descending order
grouped = df.sort_values(by = ['decade', 'ratings'], ascending = [True, False])

# Example: Filter for books from the 1960s and order by ratings
"""books_1960s = grouped[grouped['decade'] == 1960]
print("\nBooks from the 1960s ordered by ratings:")
print(books_1960s)"""

#----------------------------------------------------------------------------------
# Filtering genres
unwanted_genres = ['Graphic Novels',
                   'Comics',
                   'Graphic Novels Comics', 
                   'Comic Book', 
                   'Manga',
                   'Short Stories']
unwanted_genres = list(map(str.lower, unwanted_genres))
required_genre = 'Science Fiction'.lower()

# Initialize a list to keep track of indices of rows to keep
indices_to_keep = []

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    genres = row['genres']
    # Check if the row should be kept:
    # 1. The list of genres must not contain any unwanted genres
    # 2. The list must contain the required genre
    has_unwanted_genre = any(genre.lower() in unwanted_genres for genre in genres)
    has_required_genre = required_genre.lower() in [genre.lower() for genre in genres]

    # Add the index if the row does not have any unwanted genres and has the required genre
    if not has_unwanted_genre and has_required_genre:
        indices_to_keep.append(index)

# Create the filtered DataFrame using the indices of rows to keep
df_filtered = df.loc[indices_to_keep]

print(df_filtered.info())
#print(df_filtered.head())

# Save the filtered DataFrame back to a CSV
df_filtered.to_csv('sci-fi_books_filtered.csv', index=False, sep=';')

#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------