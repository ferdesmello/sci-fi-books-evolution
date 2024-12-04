import pandas as pd
import wikipedia
import wikipediaapi
import re
import requests
from bs4 import BeautifulSoup

#----------------------------------------------------------------------------------
def clean_text(text):
    """
    Clean up the retrieved Wikipedia text by removing references, 
    external links, and other unwanted formatting.
    """
    # Remove reference markers like [1], [2], etc.
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove multiple consecutive newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text

#----------------------------------------------------------------------------------
def get_book_summary(title, author=None):
    try:
        # Construct search query
        if author:
            search_query = f'"{title}" {author} (novel)'
        else:
            search_query = f'"{title}" (novel)'
        
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
                          '(comics)']
        
        separated_author_name = author.replace(".", ". ").replace(".  ", ". ").lower()
        unwanted_terms.append(separated_author_name)

        first_name = author.split()[0]
        last_name = author.split()[-1]
        shorter_author_name = first_name + " " + last_name
        unwanted_terms.append(shorter_author_name.lower())

        print("search_query:", search_query)#--------------------------------------
        print("search_results:", search_results)#--------------------------------------
                
        filtered_results = [
            result for result in search_results 
            if not any(term in result.lower() for term in unwanted_terms)
        ]

        print("Filtered results:", filtered_results)#--------------------------------------

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

        if chosen_result.split()[0].lower() != title.split()[0].lower():
            chosen_result = None

        if not chosen_result:
            return {
                'summary': None,
                'page_title': None,
                'page_url': None,
                'success': False,
                'error': "No suitable articles found."
            }
        print(chosen_result.split()[0].lower())
        print(title.split()[0].lower())
        print("chosen_result:", chosen_result)#--------------------------------------

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
            plot_headers = ['plot',
                            'the plot',
                            'plot summary', 
                            'synopsis', 
                            'summary', 
                            'book plot', 
                            'story', 
                            'storyline', 
                            'narrative',
                            'species of humans',
                            'outline']
        
            page_title = page.title
            page_url = page.url
            
            # Find all section headings
            section_headings = soup.find_all(['h2'])

            # Locate the correct section
            plot_heading = None
            for heading in section_headings:
                if heading.get_text(strip=True).lower() in plot_headers:
                    plot_heading = heading
                    break

            print("plot_heading:", plot_heading)#--------------------------------------
            
            if not plot_heading:
                return {
                    'summary': None,
                    'page_title': page_title,
                    'page_url': page_url,
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
                return {
                    'summary': plot_text,
                    'page_title': page_title,
                    'page_url': page_url,
                    'success': True
                }
            else:
                return {
                    'summary': None,
                    'page_title': page_title,
                    'page_url': page_url,
                    'success': False,
                    'error': "No content found in Plot section."
                }
            
        #------------------------------------------
        except Exception as e:
            return {
                'summary': None,
                'page_title': None,
                'page_url': None,
                'success': False,
                'error': str(e)
            }
        
    #------------------------------------------
    except Exception as e:
        return {
            'summary': None,
            'page_title': None,
            'page_url': None,
            'success': False,
            'error': str(e)
        }

#----------------------------------------------------------------------------------
# Read the CSV file
#df = pd.read_csv('./Data/sci-fi_books_TEST.csv', sep = ';', encoding="utf-8-sig")
df = pd.read_csv('./Data/sci-fi_books_TOP.csv', sep = ';', encoding="utf-8-sig")
print(df.info())

list_plots = []
list_urls = []

# Add a new column for the plot text
for _, book in df.iterrows():
    # Extract book details
    title = book['title']
    author = book['author']

    returned_dict = get_book_summary(title, author)
    returned_title = returned_dict.get('page_title')
    returned_url = returned_dict.get('page_url')
    returned_text = returned_dict.get('summary')
    
    #print(title, author)
    print(returned_title)
    print(returned_url)
    #print(returned_text)
    print()
    list_plots.append(returned_text)
    list_urls.append(returned_url)

df["plot"] = list_plots
df["url wikipedia"] = list_urls

#----------------------------------------------------------------------------------
df = df.rename(columns={"url": "url goodreads"})

# Reordering columns
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

df = df.reindex(columns=column_order)
df = df.sort_values(by=['year', 'author', 'title'], ascending=True)

#df.to_csv('./Data/sci-fi_books_TEST_Wiki.csv', index=False, sep=';', encoding='utf-8-sig')
df.to_csv('./Data/sci-fi_books_TOP_Wiki.csv', index=False, sep=';', encoding='utf-8-sig')