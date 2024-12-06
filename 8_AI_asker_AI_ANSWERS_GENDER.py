import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

#----------------------------------------------------------------------------------
# Main execution function
def main():
    
    # Name of the input file
    #input_file = './Data/top_books_TEST.csv'
    input_file = './Data/sci-fi_books_AI_ANSWERS.csv'

    # Name of the output file
    output_file = './Data/sci-fi_books_AI_ANSWERS_GENDER.csv'

    #----------------------------------------------------------------------------------
    # Load book data to send to the AI
    df = pd.read_csv(input_file, sep=';', encoding="utf-8-sig")
    #print(df.info())

    authors_list = list(set(df['author']))
    print("length =",len(authors_list))
    print("\n")

    #----------------------------------------------------------------------------------
    # Load the OpenAI API key
    load_dotenv(dotenv_path='../KEYs/My_OPENAI_API_Key.env')
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    #------------------------------------------
    # Function to query the AI about author gender
    def analyze_author(author):
        # Create the prompt with the gathered data
        prompt = f"""
        Consider the writer {author}.
        What is {author} gender?
            Male; Female; Other.
        Please, give only one-word answers and only from the three alternatives given, following the example:
        Male
        """
        
        # Call the OpenAI API with the crafted prompt
        ChatCompletion = client.chat.completions.create(
            messages = [
                {"role": "system", "content": "You are a helpful assistant and scholar of comparative sci-fi literature who analyzes the gender of writers based on your own knowledge about them and their names."},
                {"role": "user", "content": prompt}
            ],
            #model = "gpt-4o-mini-2024-07-18",
            model = "gpt-4o-2024-08-06",
            #model = "gpt-4o",
            max_tokens = 10, # Adjust based on the detail needed
            temperature = 0.2 # Adjust for factual response vs. creativity balance
        )
        
        # Extract and print the response
        answer = ChatCompletion.choices[0].message.content
        print(f'{author} is {answer}')
        #print(prompt)
        #print(answer)

        return answer

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

    # Iterate through the author names and query GPT-4o
    for name in authors_list:
        # Skip already processed authors
        if name in processed_authors:
            continue
        
        gender = analyze_author(name)
        results.append((name, gender))
        number += 1

    print(f"Added {number} author(s) and their gender to the list.")

    # Convert to a DataFrame and add it to the end of the present DataFrame
    df_added_authors = pd.DataFrame(results, columns=['author', 'gender'])
    df_authors = pd.concat([df_authors, df_added_authors], ignore_index=True)

    df_authors = df_authors.sort_values(by=['gender', 'author'], ascending=True)
    df_authors = df_authors.reset_index(drop=True)

    #------------------------------------------
    Count = df_authors.value_counts(subset='gender', normalize=False, sort=True, ascending=False, dropna=True)
    Fraction = df_authors.value_counts(subset='gender', normalize=True, sort=True, ascending=False, dropna=True).mul(100).round(1)

    print("\n")
    print(df_authors.info())
    print("\n")
    print(Count)
    print("\n")
    print(Fraction)

    #----------------------------------------------------------------------------------
    df_authors.to_csv(output_file, index=False, sep=';', encoding='utf-8-sig')
    print(f"Data saved to {output_file}")

#----------------------------------------------------------------------------------
# Execution
if __name__ == "__main__":
    main()