import pandas as pd

#----------------------------------------------------------------------------------
df_filtered = pd.read_csv('./Data/sci-fi_books_FILTERED.csv', sep = ';', encoding="utf-8")

#----------------------------------------------------------------------------------
# Top 200 rating books for every decade

# Group by 'decade', sort by 'ratings' in descending order, and select the top 200 per group
df_top_books = (df_filtered.groupby('decade', group_keys=False)
                .apply((lambda x: x.sort_values('ratings', ascending=False).head(200)), 
                       include_groups=False))

#----------------------------------------------------------------------------------
# Decade has been lost in the grupby above, so reintroducing it below
decade_index = []

for index, _ in df_top_books.iterrows():
    decade_index.append(index)

df_top_books['decade'] = df_filtered['decade'].loc[decade_index]
df_top_books = df_top_books.reset_index(drop=True)

#----------------------------------------------------------------------------------
# Reordering columns

column_order = ['title', 
                'author', 
                'year',
                'decade', 
                #'pages', 
                'rate', 
                'ratings', 
                'series', 
                'genres', 
                'synopsis',
                'review',
                'url']

df_top_books = df_top_books.reindex(columns=column_order)

#----------------------------------------------------------------------------------

df_top_books.to_csv('./Data/top_sci-fi_books_200_PER_DECADE.csv', index=False, sep=';', encoding='utf-8-sig')

print("\nTOP 200 BOOKS PER DECADE Dataframe")
print(df_top_books.info())

#----------------------------------------------------------------------------------
# Sample of the total for testing

test_books = [
    "Dune",
    "The Hitchhiker’s Guide to the Galaxy",
    "1984",
    "Brave New World",
    "The Forever War",
    "Stranger in a Strange Land",
    "Childhood’s End",
    "The Time Machine",
    "Twenty Thousand Leagues Under the Sea",
    "The Left Hand of Darkness",
    "Contact",
    "Blindness",
    "Annihilation",
    "Solaris",
    "Foundation",
    "Last and First Men",
    "The Sparrow",
    "The Player of Games",
    "Blindsight",
    "The Fountains of Paradise",
    "Sirius"
]

test_books_mask = df_filtered['title'].isin(test_books)
df_test_books = df_filtered[test_books_mask]

df_test_books = df_test_books.reindex(columns=column_order)
df_test_books = df_test_books.sort_values(by=['ratings'], axis=0, ascending=False)

df_test_books.to_csv('./Data/top_books_TEST.csv', index=False, sep=';', encoding='utf-8-sig')