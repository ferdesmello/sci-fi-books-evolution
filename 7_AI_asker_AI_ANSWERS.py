"""
This script uses GPT-5, via the OPENAI API, to answer questions about the plot 
of the books scraped before, parses the answers and saves it.

Modules:
    - os
    - pandas
    - openai
    - dotenv
    - tenacity
    - requests
    - logging
    - typing
    - winsound (only for Windows)
"""

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type, 
    wait_fixed
)
from requests.exceptions import RequestException
import logging
from typing import List
import winsound

#----------------------------------------------------------------------------------
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from the .env file
load_dotenv(dotenv_path='../KEYs/My_OPENAI_API_Key.env')

# Get the API key from the environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Configure the OpenAI client with the loaded API key
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
    print("OpenAI API key loaded successfully.")
else:
    print("Error: OPENAI_API_KEY not found in the environment variables. Make sure your .env file is correctly configured.")

#----------------------------------------------------------------------------------
@retry(
    retry=retry_if_exception_type((RequestException, Exception)), # Retry on API errors or network issues
    wait=wait_exponential(multiplier=1, min=4, max=60), # Exponential backoff: starts at 4 seconds, max 60 seconds
    stop=stop_after_attempt(10) # Stop after 10 attempts
)
def analyze_book(title: str , author: str, year: int, synopsis: str, review: str, plot: str, genres: List[str]) -> str: 
    """
    Prompts GPT-5 to analyse the data of each book and answers questions about it.

    Args:
        title (str): Book title.
        author (str): Book author.
        year (int): Book publishing year.
        synopsis (str): Goodreads synopsis for the book.
        review (str): Chosen Goodreads review for the book.
        plot (str): Wikipedia plot for the book.
        genres (List[str]): Goodreads list of genres.

    Returns:
        answer (str): GPT-5 text answer.
    """

    prompt = f"""
    You are a helpful assistant and scholar of comparative sci-fi literature who analyzes book plots based on your own knowledge and provided information.
    You received the task to carefully consider the plot of the book "{title}" by {author}, published in {year}, focusing on key elements that will help you answer the following questions. 

    **Output Formatting Instructions**:
    Follow this exact format!

    Provide a concise paragraph summarizing the central and relevant elements of the book, including:
    - Themes central to the story or that significantly impact the plot;
    - Memorable locations or time settings unique to the story;
    - The main character, if there is one, and secondary characters;
    - Important creatures (aliens, robots, AIs, etc.) central to the story;
    - Notable technologies that play a key role in the plot.

    After the summary, leave one blank line and provide one answer per line in the following format:
    Question number, followed by a period, then the selected answer from the alternatives given, followed by a colon, and then the detailed but short justification in a single sentence.
    Example:
    1. Moderate: Scientific plausibility and speculation are equally important in the story.
    2. Mixed: The story balances elements of both soft and hard sciences.
    3. Heavy: The story is emotionally demanding, with complex ideas and scientific concepts.
    ...

    Important Notes:
    Ensure there are no line breaks, extra spaces, or symbols between answers.
    If no clear summary is possible, provide a brief explanation before answering.
    Answers must be one of the provided options.
    Answer in English.

    **Questions**:

    1. How important is scientific accuracy and plausibility in the story?
        Very low: Scientific accuracy is minimal or irrelevant, leaning heavily towards fantasy or speculation;
        Low: Some attention to plausibility, but scientific accuracy is clearly secondary to other elements;
        Moderate: A balanced mix, scientific plausibility is considered but not always prioritized;
        High: Scientific plausibility is important and generally consistent with known science;
        Very high: Scientific accuracy is a central component, with the narrative driven by plausible scientific details;
        Uncertain: Not enough information to say, unclear.
    2. What is the main disciplinary focus of the story?
        Soft sciences: Psychology, sociology, politics, philosophy, anthropology, etc., dominate the narrative;
        Leaning soft sciences: Soft topics dominate, with some hard sciences elements;
        Mixed: Balances elements of both soft and hard sciences equally;
        Leaning hard sciences: Hard topics dominate, with some soft sciences elements;
        Hard sciences: Physics, astronomy, biology, engineering, mathematics, etc., dominate the narrative;
        Uncertain: Not enough information to say, unclear.
    3. Is the story considered more of a light or heavy reading experience?
        Very light: Easily accessible, fast read, minimal intellectual demands, focus on entertainment, humor, or adventure;
        Light: Somewhat accessible, with some thoughtful elements and themes, but still focused on entertainment;
        Balanced: A mix of light and heavy elements, moderately complex and deep;
        Heavy: Intellectually or emotionally demanding, with complex ideas and deeper themes;
        Very heavy: Challenging, slow read, dense in language, themes, or ideas, focus on philosophical or intricate scientific concepts;
        Uncertain: Not enough information to say, unclear.
    4. When does most of the story take place in relation to the year the book was published?
        Distant past: Millennia or more before; 
        Far past: Centuries before; 
        Near past: Within a few decades before; 
        Present: Within a few years; 
        Near future: Within a few decades ahead; 
        Far future: Centuries ahead; 
        Distant future: Millennia or more ahead; 
        Multiple timelines: distinct time periods without a single dominant timeframe; 
        Uncertain: Not enough information to say, unclear.
    5. What is the mood of the story?
        Very optimistic: Overwhelmingly positive, uplifting, hopeful; 
        Optimistic: Positive outlook but with moments of pessimism; 
        Balanced: A mix of positive and negative moods without leaning towards one; 
        Pessimistic: Negative outlook but with moments of optimism; 
        Very pessimistic: Overwhelmingly negative, bleak, hopeless;
        Uncertain: Not enough information to say, unclear.
    6. What is the social-political scenario depicted in the story?
        Utopic: Ideal or perfect society;
        Leaning utopic: Significant prosperity and desirable elements, but with some flaws;
        Balanced: A mix of both strengths and flaws elements, or an ordinary view of society;
        Leaning dystopic: Significant problems and undesirable elements, but with some strengths;
        Dystopic: Bleak, deeply flawed, authoritarian, and oppressive;
        Uncertain: Not enough information to say, unclear.
    7. Is most of the story set on planet Earth?
        Yes;
        No;
        Uncertain: Not enough information to say, unclear.
    8. Is the story set in a post-apocalyptic world?
        Yes: Clear post-apocalyptic state, after a civilization-collapsing event;
        Somewhat: Just some elements are present, or the collapse is partial or local;
        No: It's not set in a post-apocalyptic world;
        Uncertain: Not enough information to say, unclear.
    9. Are there any depictions of extraterrestrial life forms or alien technology in the story?
        Yes: Extraterrestrial beings (aliens, non-terrestrial organisms, creatures not originating on Earth, etc.), or alien technology are mentioned or depicted;
        No: No extraterrestrial life forms or technology are present;
        Uncertain: Not enough information to say, unclear.
    10. How are the extraterrestrial life forms generally depicted in the story?
        Not applicable: No extraterrestrial life forms present, answered No to the prior question;
        Good: Friendly, virtuous, helpful, or heroic; 
        Leaning good: Generally positive or benign but with flaws or minor conflicts; 
        Ambivalent: Morally ambiguous, showing both positive and negative traits, or multifaceted; 
        Leaning bad: Generally antagonistic or threatening but not entirely villainous; 
        Bad: Hostile, villainous, antagonistic, or threatening; 
        Uncertain: Not enough information to say, unclear, lack of (moral) characterization, or answered Uncertain to the last question.
    11. Are there any depictions of robots or artificial intelligences in the story?
        Yes;
        No;
        Uncertain: Not enough information to say, unclear.
    12. How are the robots or artificial intelligences generally depicted in the story?
        Not applicable: No robots or artificial intelligences present, answered No to the prior question;
        Good: Friendly, virtuous, helpful, or heroic; 
        Leaning good: Generally positive or benign but with flaws or minor conflicts; 
        Ambivalent: Morally ambiguous, showing both positive and negative traits, or multifaceted; 
        Leaning bad: Generally antagonistic or threatening but not entirely villainous; 
        Bad: Hostile, villainous, antagonistic, or threatening; 
        Uncertain: Not enough information to say, unclear, lack of (moral) characterization, or answered Uncertain to the last question.
    13. Is there a single protagonist or main character?
        Yes; 
        No: No clear single protagonist or main character, multiple protagonists or main characters, or a collective;
        Uncertain: Not enough information to say, unclear.
    14. What is the nature of the single protagonist or main character?
        Not applicable: No clear single protagonist or main character (multiple human or non-human main characters, a collective, etc.), answered No to the prior question;
        Human: The protagonist is a single human being (even if a cyborg or augmented);
        Non-human: The protagonist is not a single human (animal, robot, AI, alien, creature, abstract concept, polygon, etc.);
        Uncertain: Not enough information to say, unclear, or answered Uncertain to the last question.
    15. What is the gender of the single protagonist or main character, as depicted in the story?
        Not applicable: No clear single protagonist or main character to assign a gender to (multiple human or non-human main characters, a collective, etc.), or answered Not applicable to the prior question;
        Male: The single protagonist (human or non-human) is male; 
        Female: The single protagonist (human or non-human) is female; 
        Other: The single protagonist (human or non-human) is non-binary, genderfluid, gender ambiguous or fluid, or another gender identity that is not male nor female;
        Uncertain: Not enough information to say, unclear, or answered Uncertain to the last question.
    16. Are there any depictions of virtual reality in the story?
        Yes: Virtual reality, immersive digital environments, augmented reality, etc., have a central or significant role in the plot;
        Somewhat: Some form is present, but it has a minor or background role;
        No: Not present in any form;
        Uncertain: Not enough information to say, unclear.
    17. How are technology and science depicted in the story?
        Good: Optimistic and beneficial portrayal; 
        Leaning good: Mostly positive but with some issues; 
        Ambivalent: Balanced view with both positive and negative consequences; 
        Leaning bad: Largely negative portrayal but with redeeming features; 
        Bad: Pessimistic, harmful, or destructive portrayal; 
        Uncertain: Not enough information to say, unclear, or a lack of (moral) characterization.
    18. How central is the critique or reflection of specific social issues to the story?
        Core: The critique of social issues (inequality, war, discrimination, political oppression, etc.) is the main driver of the plot or a key theme;
        Major: It plays a significant role but is a secondary theme;
        Minor: It is a subtle or minimal theme in the story;
        Absent: Not present;
        Uncertain: Not enough information to say, unclear.
    19. How central is an ecological or environmental message to the story?
        Core: Main driver of the plot or key theme;
        Major: Significant role but secondary theme;
        Minor: Subtle role or minimal theme;
        Absent: Not present;
        Uncertain: Not enough information to say, unclear.

    When answering, consider how the following synopsis, review, plot summary, and genres may provide relevant context for each question.

    Book synopsis: {synopsis}
    Partial review: {review}
    Plot summary: {plot}
    Genres the book fits in: {genres}
    """

    # Call the OpenAI API with the crafted prompt
    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt,
            max_output_tokens=3000,
            reasoning={"effort": "low"}, # can be "low", "medium", or "high"
            text={"verbosity":"low"} # can be "low", "medium", or "high"
        )

        # Extract and print the response
        answer = response.output_text
        print(f'\n"{title}" by {author}, {year}')
        print(answer)

    except Exception as e:
        print("Error:", e)

    return answer

#----------------------------------------------------------------------------------
@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
def ask_to_AI(df: pd.DataFrame, output_file: str) -> pd.DataFrame:
    """
    Sends book data to the prompt function, parses the returned answers, and merges everything together.

    Args:
        df (pandas.DataFrame): Dataframe with all books information.
        output_file (str): Output file name to save the progress.

    Returns:
        df_processed (pandas.DataFrame): Dataframe with all original books information and 
            processed answers about them from GPT-5.
    """

    #----------------------------------------

    # Valid answers for each question
    valid_answers = [
        {"Very low", "Low", "Moderate", "High", "Very high", "Uncertain"},
        {"Soft sciences", "Leaning soft sciences", "Mixed", "Leaning hard sciences", "Hard sciences", "Uncertain"},
        {"Very light", "Light", "Balanced", "Heavy", "Very heavy", "Uncertain"},
        {"Distant past", "Far past", "Near past", "Present", "Near future", "Far future", "Distant future", "Multiple timelines", "Uncertain"},
        {"Very optimistic", "Optimistic", "Balanced", "Pessimistic", "Very pessimistic", "Uncertain"},
        {"Utopic", "Leaning utopic", "Balanced", "Leaning dystopic", "Dystopic", "Uncertain"},
        {"Yes", "No", "Uncertain"},
        {"Yes", "Somewhat", "No", "Uncertain"},
        {"Yes", "No", "Uncertain"},
        {"Not applicable", "Good", "Leaning good", "Ambivalent", "Leaning bad", "Bad", "Uncertain"},
        {"Yes", "No", "Uncertain"},
        {"Not applicable", "Good", "Leaning good", "Ambivalent", "Leaning bad", "Bad", "Uncertain"},
        {"Yes", "No", "Uncertain"},
        {"Not applicable", "Human", "Non-human", "Uncertain"},
        {"Not applicable", "Male", "Female", "Other", "Uncertain"},
        {"Yes", "Somewhat", "No", "Uncertain"},
        {"Not applicable", "Good", "Leaning good", "Ambivalent", "Leaning bad", "Bad", "Uncertain"},
        {"Core", "Major", "Minor", "Absent", "Uncertain"},
        {"Core", "Major", "Minor", "Absent", "Uncertain"},
    ]

    question_columns = [
        '1 accuracy',
        '2 discipline',
        '3 light heavy',
        '4 time',
        '5 mood',
        '6 social political',
        '7 on Earth',
        '8 post apocalyptic',
        '9 aliens',
        '10 aliens are',
        '11 robots and AI',
        '12 robots and AI are',
        '13 protagonist',
        '14 protagonist nature',
        '15 protagonist gender',
        '16 virtual',
        '17 tech and science',
        '18 social issues',
        '19 enviromental',
    ]

    justification_columns = [
        'justifying accuracy',
        'justifying discipline',
        'justifying light heavy',
        'justifying time',
        'justifying mood',
        'justifying social political',
        'justifying on Earth',
        'justifying post apocalyptic',
        'justifying aliens',
        'justifying aliens are',
        'justifying robots and AI',
        'justifying robots and AI are',
        'justifying protagonist',
        'justifying protagonist nature',
        'justifying protagonist gender',
        'justifying virtual',
        'justifying tech and science',
        'justifying social issues',
        'justifying enviromental'
    ]

    #----------------------------------------
    # Load existing progress if the file exists
    if os.path.exists(output_file):
        df_processed = pd.read_csv(output_file, sep=';', encoding='utf-8-sig')
        processed_books = set(df_processed['url goodreads'])
    else:
        df_processed = pd.DataFrame()
        processed_books = set()
    for _, book in df.iterrows():
        # Skip already processed books
        if book['url goodreads'] in processed_books:
            continue

        # Extract book details
        title = book['title']
        author = book['author']
        year = int(book['year'])
        decade = int(book['decade'])
        #pages = book['pages']
        rate = float(book['rate'])
        ratings = int(book['ratings'])
        series = book['series']
        genres = book['genres']
        synopsis = book['synopsis']
        review = book['review']
        url_g = book['url goodreads']
        plot = book['plot']
        url_w = book['url wikipedia']

        forbidden_titles = [] # Empty for now. Add titles here if you want to force "No plot available."

        if pd.isna(plot) or len(plot) < 100 or title in forbidden_titles:
            plot = 'No plot available.'
        #print(f"\nPlot: {plot}")

        if len(plot) > (len(synopsis) + len(review)):# and title not in forbidden_titles:
            synopsis = 'No synopsis available.'
            review = 'No review available.'
        #print(plot)
        #print(synopsis)
        #print(review)
        #----------------------------------------
        try:
            # Get the AI's answers for the book
            AI_answers = analyze_book(title, author, year, synopsis, review, plot, genres)

            # Split answers into a list of, hopefully, 21 items
            answers = []
            justifications = []
            lines = [line.strip() for line in AI_answers.split('\n')]

            # Guarantee at least two lines (paragraph + at least one answer)
            if len(lines) < 2 or len(lines) > 21:
                lines = ["", ""]

            # First line is always the paragraph text
            paragraph = lines[0]

            # Detect where answers start (skip blank second line if present)
            start_idx = 2 if len(lines) == 21 and lines[1] == "" else 1

            # Process each answer line
            for line in lines[start_idx:]:
                if not line:
                    continue

                # Split into number + rest
                parts_number_text = line.split('. ', 1)
                if len(parts_number_text) < 2:
                    logging.warning(f"Skipping malformed line (no '. '): {line}")
                    continue

                # Split into answer + justification
                parts_answer_just = parts_number_text[1].split(': ', 1)
                answer = parts_answer_just[0].strip()
                justification = parts_answer_just[1].strip() if len(parts_answer_just) == 2 else None

                answers.append(answer)
                justifications.append(justification)

            #----------------------------------------
            # Append answers to respective lists
            complete_answer = AI_answers

            # Check if we have exactly 19 answers and 19 justifications
            if (len(answers) == 19) & (len(justifications) == 19):

                # One pass: distribute each answer into the right result list
                for i in range(len(answers)):

                    if answers[i] in valid_answers[i]:
                        continue
                    else:
                        answers[i] = None
                        justifications[i] = None
                        logging.warning(f"Invalid answer for question {i+1} in book: {title}")

            else:
                # Something went wrong and better fill a row of None in each list
                for i in range(len(answers)):
                    answers[i] = None
                    justifications[i] = None

                logging.warning(f"Unexpected number of answers for book: {title}\nFound {len(answers)}/19 answers and {len(justifications)}/19 justifications.")
                
            #----------------------------------------
            # One-row dataframe to save the progress in the present book/row
            data = {
                "title": [title],
                "author": [author],
                "year": [year],
                "paragraph": [paragraph],
                "complete answer": [complete_answer],
                "decade": [decade],
                # "pages": [pages],
                "rate": [rate],
                "ratings": [ratings],
                "series": [series],
                "genres": [genres],
                "synopsis": [synopsis],
                "review": [review],
                "url goodreads": [url_g],
                "plot": [plot],
                "url wikipedia": [url_w],
            }

            # Add the repeated questions/justifications columns
            if (len(answers) == 19):
                for i in range(19):
                    data[question_columns[i]] = [answers[i]]
                    data[justification_columns[i]] = [justifications[i]]

            else:
                for i in range(19):
                    data[question_columns[i]] = [None]
                    data[justification_columns[i]] = [None]
                logging.warning(f"Filling None for all answers and justifications for book: {title}")

            # Build DataFrame
            df_progress = pd.DataFrame(data)

            #----------------------------------------
            column_order = [
                'title', 
                'author', 
                'year',
                'paragraph',
                '1 accuracy',
                'justifying accuracy',
                '2 discipline',
                'justifying discipline',
                '3 light heavy',
                'justifying light heavy',
                '4 time',
                'justifying time',
                '5 mood',
                'justifying mood',
                '6 social political',
                'justifying social political',
                '7 on Earth',
                'justifying on Earth',
                '8 post apocalyptic',
                'justifying post apocalyptic',
                '9 aliens',
                'justifying aliens',
                '10 aliens are',
                'justifying aliens are',
                '11 robots and AI',
                'justifying robots and AI',
                '12 robots and AI are',
                'justifying robots and AI are',
                '13 protagonist',
                'justifying protagonist',
                '14 protagonist nature',
                'justifying protagonist nature',
                '15 protagonist gender',
                'justifying protagonist gender',
                '16 virtual',
                'justifying virtual',
                '17 tech and science',
                'justifying tech and science',
                '18 social issues',
                'justifying social issues',
                '19 enviromental',
                'justifying enviromental',
                'complete answer',
                'decade', 
                'rate', 
                'ratings', 
                'series', 
                'genres', 
                'synopsis',
                'review',
                'url goodreads',
                'plot',
                'url wikipedia',
            ]

            # Check for duplicate columns in column_order
            if len(column_order) != len(set(column_order)):
                raise ValueError("Duplicate columns in column_order")

            # Reorder the columns
            df_progress = df_progress.reindex(columns=column_order)

            #----------------------------------------
            # Concatenate the one-row dataframe with the big dataframe with all anterior books/rows
            df_processed = pd.concat([df_processed, df_progress], ignore_index=True)
            df_processed.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')
            logging.info(f"Progress saved for book: {title}")

        except Exception as e:
            # Log the full AI response that caused the error for debugging
            logging.error(f"Failed to analyze book {title}. Error: {e}")
            logging.error(f"Problematic AI response: {AI_answers}")
            raise  # Re-raise the exception to trigger a retry

    return df_processed

#----------------------------------------------------------------------------------
def main():
    """
    Main execution function for the script.
    Calls the AI asker function, orders the data, and saves it in a CSV file.
    """
        
    #------------------------------------------
    # Name of the input file
    #input_file = './Data/Filtered/sci-fi_books_TEST_Wiki.csv'
    #input_file = './Data/Filtered/sci-fi_books_TEST_Wiki_small.csv'
    input_file = './Data/Filtered/sci-fi_books_TOP_Wiki.csv'

    # Name of the output file
    #output_file = './Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_01.csv'
    #output_file = './Data/Answers/sci-fi_books_AI_ANSWERS_TEST.csv'
    #output_file = './Data/Answers/sci-fi_books_AI_ANSWERS_TEST_small.csv'
    output_file = './Data/Answers/sci-fi_books_AI_ANSWERS.csv'

    #------------------------------------------
    # Load book data to send to the AI
    df = pd.read_csv(input_file, sep=';', encoding="utf-8-sig")

    # Ask the AI about ALL the books
    df_processed = ask_to_AI(df, output_file)

    # Retyping columns of the processed dataframe
    df_processed['year'] = df_processed['year'].astype(int)
    df_processed['decade'] = df_processed['decade'].astype(int)
    df_processed['rate'] = df_processed['rate'].astype(float)
    df_processed['ratings'] = df_processed['ratings'].astype(int)

    #------------------------------------------
    # Sometimes the AI output is not formatted right or is null

    # Get all columns from "paragraph" through "complete answer"
    cols_to_check = df_processed.loc[:, "paragraph":"complete answer"].columns

    # Drop rows with any null in that slice
    df_processed = df_processed.dropna(axis=0, subset=cols_to_check, how="any", ignore_index=True)
    df_processed = df_processed.sort_values(by = ['decade', 'year', 'author', 'title'], ascending=True)

    #------------------------------------------
    size_in = df.shape[0] # Number of rows
    size_out = df_processed.shape[0] # Number of rows
    missing_books = size_in - size_out # Difference in number of rows

    #------------------------------------------
    df_processed.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')
    print(f"Data saved to {output_file}")

    return missing_books, df_processed # How many books were not processed right and the DataFrame

#----------------------------------------------------------------------------------
# Execution
if __name__ == "__main__":

    missing_books = 1
    max_retries = 20
    attempt = 1

    while (missing_books != 0) and (attempt <= max_retries):
        missing_books, df_processed = main()
        print(f"Book(s) missing: {missing_books}.\nAttempts made: {attempt}.")
        attempt += 1
    
    print('\n', df_processed.info())

    winsound.Beep(800, 500) # Play a 800 Hz beep for 500 milliseconds
    winsound.Beep(500, 500) # Play a 500 Hz beep for 500 milliseconds
    winsound.Beep(300, 500) # Play a 300 Hz beep for 500 milliseconds