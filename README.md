# sci-fi-books-evolution

## Overview

Inspired by this project ([video](https://www.youtube.com/watch?v=nRQ2vMpw-n8), [page](https://pudding.cool/2024/07/scifi/)) about the evolution in the plot of sci-fi movies and series and this [comment](https://www.youtube.com/watch?v=nRQ2vMpw-n8&lc=UgyRg89P8kRYQ2SdXrV4AaABAg) I decided to do a similar analysis but for sci-fi _novels_.

You can read in detail all the process of the data and analysis of the results [here](https://fdesmello.wordpress.com/2024/11/21/a-journey-through-170-years-of-sci-fi-novels-a-study-using-data-and-ai/).

But, in short, I scraped data about thousands of sci-fi books from [Goodreads](https://www.goodreads.com/) lists and shelves, cleaned and reduced the data, selected the top 200 novels per decade (or all the novels if fewer than 200 per decade), and fed that into GPT-4o via the OpenAI API, asking about many plot-related things. Then, I aggregated the results in figures to see how things changed over time.

## What the code does

### 1. Scraping the data

**Goodreads_scraper_SHELF.py** reads HTML files in the folder **Saved_pages** and searches for links to the book pages and downloads data for book's _title_, _author_, _year_ published, number of _pages_, if it is part of a _series_, average _rate_, number of _ratings_, _genres_, _synopsis_ and the longest of the first two _reviews_, saving everything in the **sci-fi_books_SHELF.csv** file at the end.

You will have to download and put the HTML files in the Saved_pages folder by hand. I didn't include them here because they are big and were using too much space.

**Goodreads_scraper_LISTS.py** does the same but from a list of URLs for the initial pages of thematic book lists, saving the progress in a JSON file and everything in the **sci-fi_books_LISTS.csv** file at the end. 

All data recovered and processed is stored in the **Data** folder.

### 2. Cleaning the data

**Data_reducer_FILTERED.py** merges the datafiles, creating **sci-fi_books_BRUTE.csv**, and cleans them (many duplicates, bad data, and books not of interest), creating the **sci-fi_novels_FILTERED.csv** file.

**Data_reducer_TOP.py** selects the top 200 novels per decade (by the number of user ratings) and saves them as **sci-fi_novels_TOP.csv**. It also creates the **sci-fi_novels_TEST.csv** file, with just a small selection of the novels to test the AI performance. 

If this is your first time running everything, you can proceed to the next step. But if you have already run everything to the end and are just adding some books from the scraper, that may change which books are in the top 200 per decade. Some rows from the AI answers' CSV may need to be excluded, and/or new ones may need to be processed. For this, run **Data_fixer.py** _before_ the next step.

### 3. Prompting the LLM

**AI_asker_AI_ANSWERS.py** reads the **sci-fi_books_TOP.csv** file (or **sci-fi_books_TEST.csv**) and, for every novel in the file, sends the prompt with the novel's data to OpenAI's API for GPT-4o and receives a text answer, parses it and saves it in the **sci-fi_books_AI_ANSWERS.csv** file.

Sometimes, the output from GPT-4o will not be in the right format, so that book will fail, but the run will keep going. You just need to rerun the program after the run is over for it to try again _only_ on the failed books of the prior run.

The output of using **sci-fi_books_TOP.csv** is stored in the **Variability_in_Answers** folder. There are already ten of those files to be used with the **Figure_maker.py** program to have an estimate of the model change in answer for each book and question.

**AI_asker_AI_ANSWERS_GENDER.py** reads the **sci-fi_books_TOP.csv** file and, for every _author_ in the file, sends the prompt with the author's name to OpenAI's API for GPT-4o and receives the author's gender, and saves it in the **sci-fi_books_AI_ANSWERS_GENDER.csv** file.

For both programs, you will need to have a working API key set in your environment and credits in your OpenAI account for them to work.

### 4. Plotting the results

**Figure_maker.py** reads the **sci-fi_books_AI_ANSWERS.csv** and the **sci-fi_books_AI_ANSWERS_GENDER.csv** files, it also reads all the files in the **Variability_in_Answers** folder, and it makes figures from them, saving all of them in the **Figures** folder.

## Example of a figure

The figure below is one example of the output.

![A final plot as an example.](./Figures/03%20time.png "When does most of the story take place in relation to the year the book was published? Distant past: millennia or more before; Far past: centuries before; Near past: within a few decades before; Present: within a few years; Near future: within a few decades ahead; Far future: centuries ahead; Distant future: millennia or more ahead; Multiple timelines; Uncertain.")
