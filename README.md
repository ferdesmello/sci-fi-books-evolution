# sci-fi-books-evolution

## Overview

Inspired by this project ([video](https://www.youtube.com/watch?v=nRQ2vMpw-n8), [page](https://pudding.cool/2024/07/scifi/)) about the evolution in plot of sci-fi movies and series and this [comment](https://www.youtube.com/watch?v=nRQ2vMpw-n8&lc=UgyRg89P8kRYQ2SdXrV4AaABAg) I decided to do a similar analysis but for sci-fi _novels_.

You can read in detail all the process and analysis of results [here]().

But, in short, I scraped data about thousands of sci-fi books from [Goodreads](https://www.goodreads.com/) lists and shelves, cleaned and reduced the data, selected the top 200 books per decade (or all the books if fewer than 200 per decade), and fed that into GPT-4o via the OpenAI API, asking about many plot-related things. Then, I aggregated the results in figures to see how things changed over time.

## What the code does

### 1. Scraping the data

**Goodreads_scraper_SHELF.py** reads HTML files in the folder **Saved_pages** and searches for links to the book pages and downloads data for book's _title_, _author_, publishing _year_, number of _pages_, if it is part of a _series_, average _rate_, number of _ratings_, _genres_, _synopsis_ and the longest of the first two _reviews_, saving everything in the **sci-fi_books_SHELF.csv** file at the end.
You will have to dowanlod and put the HTML files in the Saved_pages folder by hand. I didn't include them because they are big and were using too much space.

**Goodreads_scraper_LISTS.py** does the same but from a list of URLs for the initial pages of thematic book lists, saving the progress in a JSON file and everything in the **sci-fi_books_LISTS.csv** file at the end. 

All data recovered and processed is stored in the **Data** folder.

### 2. Cleaning the data

**Data_reducer_FILTERED.py** merges the datafiles, creating **sci-fi_books_BRUTE.csv**, and cleans them (many duplicates, bad data, and books not of interest), creating the **sci-fi_books_FILTERED.csv** file.

**Data_reducer_TOP.py** selects the top 200 books per decade (by the number of user ratings) and saves them as **sci-fi_books_TOP.csv**. It also creates the **sci-fi_books_TEST.csv** file, with just a small selection of the books to test the AI performance. 

### 3. Using the AI

**AI_asker_AI_ANSWERS.py** reads the **sci-fi_books_TOP.csv** file (or **sci-fi_books_TEST.csv**) and, for every _book_ in the file, sends the prompt with the book's data to OpenAI's API for GPT-4o and receives a text answer, parses it and saves it in the **sci-fi_books_AI_ANSWERS.csv** file.
Sometimes, the output from GPT-4o will not be in the right format, so that book will fail, but the run will keep going. You just need to rerun the program after the run is over for it to try again _only_ on the failed books of the prior run.

**AI_asker_AI_ANSWERS_GENDER.py** reads the **sci-fi_books_TOP.csv** file and, for every _author_ in the file, sends the prompt with the author's name to OpenAI's API for GPT-4o and receives the author's gender, and saves it in the **sci-fi_books_AI_ANSWERS_GENDER.csv** file.

For both programs, you will need to have a working API key set in your environment and credits in your OpenAI account.

### 4. Plotting the results

**Figure_maker.py** reads the **sci-fi_books_AI_ANSWERS.csv** and **sci-fi_books_AI_ANSWERS_GENDER.csv** files and makes data figures from them, saving all of them in the **Figures** folder.

## Example of a figure

The figure below is one example of the output.

![A final plot as an example.](./Figures/03%20time.png "When does most of the story take place in relation to the year the book was published? Distant past: millennia or more before; Far past: centuries before; Near past: within a few decades before; Present: within a few years; Near future: within a few decades ahead; Far future: centuries ahead; Distant future: millennia or more ahead; Multiple timelines; Uncertain.")
