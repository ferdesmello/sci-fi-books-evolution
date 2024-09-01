import pandas as pd

#----------------------------------------------------------------------------------
df = pd.read_csv('sci-fi_books_extracted.csv', sep = ';')

df["title"] = df["title"].str.replace(r' \(.*\)', '', regex = True)

# Replacing some missing years
index_01 = df[df['title'] == "Blindsight"].index
df.loc[index_01, 'year'] = 2006

index_02 = df[df['title'] == "The Ringworld Throne"].index
df.loc[index_02, 'year'] = 1997

df.sort_values(by = ['year'], axis = 0, ascending = True, inplace = True)

print(df.info())
print(df.tail())

#----------------------------------------------------------------------------------


#----------------------------------------------------------------------------------
# Create a 'decade' column
df['decade'] = (df['year'] // 10) * 10  # Floor division by 10, then multiply by 10 to get the start of the decade

# Group by decade and sort within each group by 'ratings' in descending order
grouped = df.sort_values(by = ['decade', 'ratings'], ascending = [True, False])

# Display the grouped DataFrame
print(grouped)

# Example: Filter for books from the 1960s and order by ratings
books_2000s = grouped[grouped['decade'] == 1930]
print("\nBooks from the 2000s ordered by ratings:")
print(books_2000s)

#----------------------------------------------------------------------------------




#----------------------------------------------------------------------------------




#----------------------------------------------------------------------------------




#----------------------------------------------------------------------------------