"""
This script uses GPT-5, via the OPENAI API, to answer questions about the gender 
of the authors of the books scraped before, parses the answers and saves it.

Modules:
    - os
    - pandas
    - openai
    - dotenv
    - winsound (only for Windows)
"""

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import winsound

#----------------------------------------------------------------------------------
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
def analyze_author(author: str) -> str:
    """
    Prompts GPT-5 to analyse the author gender of each book.

    Args:
        author (str): Book author.

    Returns:
        answer (str): GPT-5 processed gender answer.
    """

    # Create the prompt with the gathered data
    prompt = f"""
    You are a helpful assistant and scholar of comparative sci-fi literature who analyzes book plots based on your own knowledge and provided information.
    Consider the writer {author}.

    **Output Formatting Instructions**:
    Follow this exact format!
    Please, give only one-word answers without any punctuation marks and only from the set of four alternatives given.
    If it is given two names, answer Uncertain, unless they are of the same gender, then answer their shared gender.
    If it is a well-known pseudonym, answer with the real author's gender; otherwise, answer Uncertain.
    
    **Question**:
    What is the gender of {author}?
        Male;
        Female;
        Other;
        Uncertain.
    """
    
    # Call the OpenAI API with the crafted prompt
    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=prompt,
            max_output_tokens=200,
            reasoning={"effort": "low"}, # can be "low", "medium", or "high"
            text={"verbosity":"low"} # can be "low", "medium", or "high"
        )

        # Extract and print the response
        answer = response.output_text
        print(f'{author} is {answer}')

    except Exception as e:
        print("Error:", e)

    return answer

#----------------------------------------------------------------------------------
def main():
    """
    Main execution function for the script.
    Calls the AI asker function, orders the data, and saves it in a CSV file.

    Returns:
        missing_authors (int): Number of authors not processed correctly.
    """
    
    # Name of the input file
    input_file = './Data/Filtered/sci-fi_books_TOP.csv'
    #input_file = './Data/Filtered/sci-fi_books_AI_ANSWERS.csv'

    # Name of the output file
    output_file = './Data/Answers/sci-fi_books_AI_ANSWERS_GENDER.csv'

    #----------------------------------------------------------------------------------
    # Load book data to send to the AI
    df = pd.read_csv(input_file, sep=';', encoding="utf-8-sig")
    #print(df.info())

    authors_list = list(set(df['author'].str.strip()))
    print("length =",len(authors_list))

    #----------------------------------------------------------------------------------
    # Main operation

    # Create a list to store the results
    results = []
    number = 0

    # Load existing progress if the file exists
    if os.path.exists(output_file):
        df_authors = pd.read_csv(output_file, sep=';', encoding='utf-8-sig')
    else:
        df_authors = pd.DataFrame(columns=['author', 'gender'])

    processed_authors = set(df_authors['author'].values)

    # Iterate through the author names and query GPT-5
    for name in authors_list:
        # Skip already processed authors
        if name in processed_authors:
            continue
        
        gender = analyze_author(name)
        
        # Checking if the output is valid and adding to results
        answer_set = {
            "Male",
            "Female",
            "Other",
            "Uncertain"
            }

        try:
            if gender.strip() not in answer_set:
                print(f"Warning: Unexpected answer for {name}")
                results.append((name, None))

            else:
                results.append((name, gender))
                number += 1

        except Exception as e:
            print("Error:", e)
            results.append((name, None))

    print(f"\nAdded {number} author(s) and their gender to the list.\n")

    # Convert to a DataFrame and add it to the end of the present DataFrame
    df_added_authors = pd.DataFrame(results, columns=['author', 'gender'])
    df_authors = pd.concat([df_authors, df_added_authors], ignore_index=True)

    # Drop rows with any null value in gender
    df_authors = df_authors.dropna(axis=0, subset=['gender'], how="any", ignore_index=True)

    # Sort the DataFrame
    df_authors = df_authors.sort_values(by=['gender', 'author'], ascending=True)
    df_authors = df_authors.reset_index(drop=True)


    #----------------------------------------------------------------------------------
    df_authors.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')
    print(f"\nData saved to {output_file}")

    for row in df_authors['author']:
        if row not in authors_list:
            print(f"\nCheck this extra author name: {row}.")

    #------------------------------------------
    size_in = len(authors_list) # Number of items
    size_out = df_authors.shape[0] # Number of rows
    missing_authors = size_in - size_out # Difference in number of rows

    return missing_authors, df_authors # How many authors were not processed right and the DataFrame

#----------------------------------------------------------------------------------
# Execution
if __name__ == "__main__":

    missing_authors = 1
    max_retries = 10
    attempt = 0

    while (missing_authors != 0) and (attempt < max_retries):
        missing_authors, df_authors = main()
        print(f"Author(s) missing: {missing_authors}.\nAttempts made: {attempt}.")
        attempt += 1
        
    #------------------------------------------
    Count = df_authors.value_counts(subset='gender', normalize=False, sort=True, ascending=False, dropna=True)
    Fraction = df_authors.value_counts(subset='gender', normalize=True, sort=True, ascending=False, dropna=True).mul(100).round(1)

    print(df_authors.info())
    print("\n", Count)
    print("\n", Fraction)  

    winsound.Beep(800, 500) # Play a 800 Hz beep for 500 milliseconds
    winsound.Beep(500, 500) # Play a 500 Hz beep for 500 milliseconds
    winsound.Beep(300, 500) # Play a 300 Hz beep for 500 milliseconds