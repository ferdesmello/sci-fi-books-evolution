"""
This script reads the filtered data scraped from the Goodreads website and select 
just the data of interest to be used in the next phase.

Modules:
    - pandas
"""

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
import pandas as pd

#----------------------------------------------------------------------------------
# Main execution function
def main():
    """
    Big main function with all the selection of the top and test data.
    """

    df_filtered = pd.read_csv('./Data/Filtered/sci-fi_books_FILTERED.csv', sep = ';', encoding="utf-8-sig")
    df_filtered['decade_gb'] = df_filtered['decade'] # To use in the groupby below and keep the original decade

    # Reorder the columns
    df_filtered = df_filtered.rename(columns={"url": "url goodreads"})
    df_filtered['plot'] = ""
    df_filtered['url wikipedia'] = ""

    column_order = [
        'title', 
        'author', 
        'year',
        'decade_gb',
        'decade', 
        'rate', 
        'ratings', 
        'series', 
        'genres', 
        'synopsis',
        'review',
        'url goodreads',
        'plot',
        'url wikipedia'
    ]

    df_filtered = df_filtered.reindex(columns=column_order)

    #----------------------------------------------------------------------------------
    # Top 200 rating books for every decade

    # Group by 'decade_gb', sort by 'ratings' in descending order, and select the top 200 per group
    df_top_books = (df_filtered.groupby('decade_gb', group_keys=False)
                    .apply((lambda x: x.sort_values('ratings', ascending=False).head(200)), 
                        include_groups=False))

    column_order = [
        'title', 
        'author', 
        'year',
        'decade', 
        'rate', 
        'ratings', 
        'series', 
        'genres', 
        'synopsis',
        'review',
        'url goodreads',
        'plot',
        'url wikipedia'
    ]

    # Reorder the columns
    df_top_books = df_top_books.reindex(columns=column_order)
    df_top_books = df_top_books.sort_values(by=['year', 'author', 'title'], ascending=True)

    #----------------------------------------------------------------------------------
    # Save the dataframe
    df_top_books.to_csv('./Data/Filtered/sci-fi_books_TOP.csv', index=False, sep=';', encoding='utf-8-sig')

    print("\nTOP 200 BOOKS PER DECADE Dataframe")
    print(df_top_books.info())

    #----------------------------------------------------------------------------------
    # Sample of the total for testing

    test_books = [
        "Dune",
        "The Hitchhiker's Guide to the Galaxy",
        "Nineteen Eighty-Four",
        "Brave New World",
        "The Forever War",
        "Stranger in a Strange Land",
        "Childhood's End",
        "Foundation and Empire",
        "The Time Machine",
        "Twenty Thousand Leagues Under the Sea",
        "The Left Hand of Darkness",
        "Contact",
        "Blindness",
        "Annihilation",
        "Last and First Men",
        "Solaris",
        "Foundation",
        "Star Maker",
        "The Sparrow",
        "Sirius",
        "Consider Phlebas",
        "The Player of Games",
        "Blindsight",
        "The Fountains of Paradise",
        "Odd John",
        "The Martian",
        "Mission of Gravity",
        "Jurassic Park",
        "Flatland: A Romance of Many Dimensions",
        "The Hunger Games",
        "Neuromancer",
        "Ready Player One",
        "The Three-Body Problem",
        "The War of the Worlds",
        "R.U.R.",
        "Rendezvous with Rama"
    ]

    test_books_mask = df_filtered['title'].isin(test_books)
    df_test_books = df_filtered[test_books_mask]

    # There is more than one book titled 'Contact'. I only want the one authored by Carl Sagan.
    no_test_authors = ["Susan Grant", "Mike Duke"]
    no_test_authors_mask = ~df_test_books['author'].isin(no_test_authors)
    df_test_books = df_test_books[no_test_authors_mask]

    df_test_books = df_test_books.reindex(columns=column_order)
    #df_test_books = df_test_books.sort_values(by=['ratings'], axis=0, ascending=False)
    df_test_books = df_test_books.sort_values(by=['decade', 'year', 'author', 'title'], axis=0, ascending=True)

    df_test_books.to_csv('./Data/Filtered/sci-fi_books_TEST.csv', index=False, sep=';', encoding='utf-8-sig')

#----------------------------------------------------------------------------------
# Execution
if __name__ == "__main__":
    main()