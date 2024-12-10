import pandas as pd
import wikipedia
import re
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Dict, Any
import os

#----------------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Clean up the retrieved Wikipedia text by removing references, 
    external links, and other unwanted formatting.
    Args:
        text (str): text to be cleaned

    Returns:
        text (str): cleaned text
    """
    # Remove reference markers like [1], [2], etc.
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove multiple consecutive newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text

#----------------------------------------------------------------------------------
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_book_summary(title: str, author: str, year: int) -> Dict[str, Any]:
    """
    Fetch book summary from Wikipedia.

    Args:
        title (str): Title of the book
        author (str): Author of the book
        year (int): Publication year of the book

    Returns:
        Dict[str, Any]: A dictionary containing:
        - 'summary': Plot summary text
        - 'page_title': Wikipedia page title
        - 'page_url': Wikipedia page URL
        - 'success': Boolean indicating if summary was found
        - 'error': Error message if retrieval failed
    """

    # Book being processed
    print(f'-- {title} ({year}) {author} --')

    try:
        # Construct search query
        search_query = f'"{title}" (novel) {author}'
        
        #------------------------------------------
        # Search for pages
        search_results = wikipedia.search(search_query, results=5)
        
        #------------------------------------------
        # Filter out author page and unwanted results
        unwanted_terms = [' series', 
                          '(series)',
                          '(book series)',
                          'film)',
                          'movie)', 
                          'adaptation)', 
                          'franchise)',
                          '(disambiguation)',
                          '(comics)',
                          'miniseries)']
        
        separated_author_name = author.replace(".", ". ").replace(".  ", ". ").lower()
        unwanted_terms.append(separated_author_name)

        separated_author_name = author.replace("Jr.", "").replace(".", ". ").replace(".  ", ". ").lower()
        unwanted_terms.append(separated_author_name)

        first_name = author.split()[0]
        last_name = author.split()[-1]
        shorter_author_name = first_name + " " + last_name
        unwanted_terms.append(shorter_author_name.lower())

        author_name_without = author.replace("Jr.", "")
        first_name = author_name_without.split()[0]
        last_name = author_name_without.split()[-1]
        shorter_author_name = first_name + " " + last_name
        unwanted_terms.append(shorter_author_name.lower())

        print("Search query:", search_query)
        print("Search results:", search_results)
                
        filtered_results = [
            result for result in search_results 
            if not any(term in result.lower() for term in unwanted_terms)
        ]

        print("Filtered results:", filtered_results)

        #------------------------------------------
        # Choose the right result

        # Normalize the title for comparison (case-insensitive)
        title_lower = title.lower()
        
        # Regex to match results like 'title (novel)' or 'title (author novel)'
        novel_pattern = re.compile(rf"^{re.escape(title_lower)}.*\(.*novel\)$", re.IGNORECASE)
        
        # Prioritize results starting with the title and ending with 'novel)'
        for result in filtered_results:
            result_lower = result.lower()
            if novel_pattern.match(result_lower):
                chosen_result = result  # Match found
            else:
                chosen_result = filtered_results[0]  # Otherwise, get the first result

        #------------------------------------------
        # Simple test to check if the chosen result at least starts with the same word as the novel's title
        
        # Exception for 1984 by George Orwell (1949)
        if title == "1984":
            title = "Nineteen Eighty-Four"
        
        first_word_of_the_result = chosen_result.split()[0].lower().replace(":", "")
        first_word_of_the_title = title.split()[0].lower().replace(":", "")
        print(f'Is "{first_word_of_the_result}" == "{first_word_of_the_title}"?')
        
        if first_word_of_the_result != first_word_of_the_title:
            chosen_result = None

        if not chosen_result:
            return {
                'summary': None,
                'page_title': None,
                'page_url': None,
                'counter_plot': 0,
                'success': False,
                'error': "No suitable articles found."
            }
                
        print("Chosen result:", chosen_result)

        #------------------------------------------
        # Try to get the information
        try:
            # Get the page
            page = wikipedia.page(chosen_result, auto_suggest=False)
            
            # Fetch the HTML content
            html_content = requests.get(page.url).text
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            #------------------------------------------
            list_plot_headers = ['plot',
                                 'the plot',
                                 'plot summary',
                                 'plot synopsis',
                                 'plot introduction',
                                 'plot outline',
                                 'plot and storyline',
                                 'synopsis',
                                 'summary',
                                 'book plot',
                                 'setting and plot',
                                 'setting and synopsis'
                                 'story and significance',
                                 'story overview',
                                 'premise and plot',
                                 'narrative',
                                 'storyline',
                                 'story',
                                 'outline',
                                 'content',
                                 'overview',
                                 'premise',
                                 'introduction',
                                 'description',
                                 'fictional premise',
                                 'characters and story',
                                 'species of humans',]
        
            page_title = page.title
            page_url = page.url
            
            # Find all section headings
            section_headings = soup.find_all(['h2'])

            # Locate the correct section, checking top-priority headers first
            plot_heading = None
            for heading in section_headings:
                for header in list_plot_headers:  # Original order
                    if heading.get_text(strip=True).lower() == header:
                        plot_heading = heading
                        break
                if plot_heading:  # Exit outer loop once a match is found
                    break

            print("Plot heading:", plot_heading)
            
            if not plot_heading:
                return {
                    'summary': None,
                    'page_title': page_title,
                    'page_url': page_url,
                    'counter_plot': 0,
                    'success': False,
                    'error': "Plot section not found."
                }
            
            #------------------------------------------
            # Collect paragraphs in the Plot section
            list_brackets = ['p', 'ol','li']
            plot_paragraphs = []
            for sibling in plot_heading.find_all_next():
                if sibling.name in ['h2']:  # Stop at next section
                    break
                if sibling.name in list_brackets:  # Collect paragraphs or list of items
                    plot_paragraphs.append(sibling.get_text().strip())

            # Combine paragraphs
            plot_text = '\n\n'.join(plot_paragraphs)
            plot_text = clean_text(plot_text)

            #------------------------------------------
            if plot_paragraphs:
                if len(plot_text) > 20000: # Some plot summaries are unnecessarily long
                    return {
                        'summary': None,
                        'page_title': page_title,
                        'page_url': page_url,
                        'counter_plot': 0,
                        'success': False,
                        'error': "Plot section is too long."
                    }
                else:
                    return {
                        'summary': plot_text,
                        'page_title': page_title,
                        'page_url': page_url,
                        'counter_plot': 1,
                        'success': True
                    }
            else:
                return {
                    'summary': None,
                    'page_title': page_title,
                    'page_url': page_url,
                    'counter_plot': 0,
                    'success': False,
                    'error': "No content found in Plot section."
                }
            
        #------------------------------------------
        except Exception as e:
            return {
                'summary': None,
                'page_title': None,
                'page_url': None,
                'counter_plot': 0,
                'success': False,
                'error': str(e)
            }
        
    #------------------------------------------
    except Exception as e:
        return {
            'summary': None,
            'page_title': None,
            'page_url': None,
            'counter_plot': 0,
            'success': False,
            'error': str(e)
        }

#----------------------------------------------------------------------------------
def main():
    """
    Main execution function
    """
    #------------------------------------------
    # Read the CSV files
    input_file = './Data/sci-fi_books_TOP.csv'
    input_file_TEST = './Data/sci-fi_books_TEST.csv'
    output_file = './Data/sci-fi_books_TOP_Wiki.csv'
    output_file_TEST = './Data/sci-fi_books_TEST_Wiki.csv'

    df_TOP = pd.read_csv(input_file, sep = ';', encoding="utf-8-sig")
    df_TEST = pd.read_csv(input_file_TEST, sep = ';', encoding="utf-8-sig")

    df_TOP = df_TOP.rename(columns={"url": "url goodreads"})
    df_TOP['plot'] = ""
    df_TOP['url wikipedia'] = ""

    print(df_TOP.info())

    #----------------------------------------
    # Load existing progress if the file exists
    if os.path.exists(output_file):
        df_processed = pd.read_csv(output_file, sep=';', encoding='utf-8-sig')
        processed_books = set(df_processed['url goodreads'])
    else:
        df_processed = pd.DataFrame()
        processed_books = set()

    #------------------------------------------
    counter_books = 0
    counter_plots = 0

    # Add a new column for the plot text
    for _, book in df_TOP.iterrows():

        # Skip already processed books
        if book['url goodreads'] in processed_books:
            continue

        # Extract book details
        title = book['title']
        author = book['author']
        year = int(book['year'])
        decade = int(book['decade'])
        rate = float(book['rate'])
        ratings = int(book['ratings'])
        series = book['series']
        genres = book['genres']
        synopsis = book['synopsis']
        review = book['review']
        url_g = book['url goodreads']

        # Querying Wikipedia
        returned_dict = get_book_summary(title, author, year)
        counter_plot = returned_dict.get('counter_plot')
        returned_title = returned_dict.get('page_title')
        returned_text = returned_dict.get('summary')
        returned_url = returned_dict.get('page_url')
        
        #print(title, author)
        print("Returned title:", returned_title)
        print("Returned url:", returned_url)
        #print(returned_text)
        print()

        #----------------------------------------
        # One-row dataframe to save the progress in the present book/row
        df_progress = pd.DataFrame({
            'title': [title],
            'author': [author],
            'year': [year],
            'decade': [decade],
            'rate': [rate],
            'ratings': [ratings],
            'series': [series],
            'genres': [genres],
            'synopsis': [synopsis],
            'review': [review],
            'url goodreads': [url_g],
            'plot': [returned_text],
            'url wikipedia': [returned_url]
        })
        
        # Concatenate the one-row dataframe with the big dataframe with all anterior books/rows
        df_processed = pd.concat([df_processed, df_progress], ignore_index=True)
        df_processed.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')

        counter_books += 1
        counter_plots = counter_plots + counter_plot
    
    print(f"Analyzed {counter_books} books and added {counter_plots} plots to the file.")

    #----------------------------------------------------------------------------------
    #df_TOP = df_TOP.rename(columns={"url": "url goodreads"})
    df_TEST = df_TEST.rename(columns={"url": "url goodreads"})

    # Include two columns in sci-fi_books_TEST.csv
    df_TEST_merged = pd.merge(df_TEST, df_processed, 
                              how='left', 
                              on='url goodreads', 
                              suffixes=('_test', '_processed'))
    
    #------------------------------------------
    # Order of the columns
    column_order = ['title', 
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
                    'url wikipedia']

    # Select and rename columns
    df_TEST_merged = df_TEST_merged[['title_test',
                                     'author_test',
                                     'year_test',
                                     'decade_test',
                                     'rate_test',
                                     'ratings_test',
                                     'series_test',
                                     'genres_test',
                                     'synopsis_test',
                                     'review_test',
                                     'url goodreads',
                                     'plot',
                                     'url wikipedia']]
    df_TEST_merged.columns = column_order

    # Reorder columns
    df_processed = df_processed.reindex(columns=column_order)
    df_processed = df_processed.sort_values(by=['year', 'author', 'title'], ascending=True)

    df_TEST_merged = df_TEST_merged.reindex(columns=column_order)
    df_TEST_merged = df_TEST_merged.sort_values(by=['year', 'author', 'title'], ascending=True)

    #------------------------------------------
    # Save the CSV file
    df_processed.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')
    df_TEST_merged.to_csv(output_file_TEST, index=False, sep=';', encoding='utf-8-sig')

    print(f"Data saved to {output_file}")
    print(f"Data saved to {output_file_TEST}")

#----------------------------------------------------------------------------------
# Execution
if __name__ == "__main__":
    main()