"""
This script process all of the data gathered in the previous steps for GPT-4 and makes figures to present the data.

Modules:
    - pandas
    - matplotlib.pyplot
    - seaborn
    - scipy.stats
    - numpy
    - plotly
    - typing
"""

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy.stats import chi2_contingency
import numpy as np
import plotly.express as px
from typing import List

#---------------------------------------------------------------------------------------------------
print("Reading and processing tha data...")

# Read the data
df_filtered = pd.read_csv("./Data/Filtered/sci-fi_books_FILTERED.csv", sep=";", encoding="utf-8-sig")
df_top = pd.read_csv("./Data/Filtered/sci-fi_books_TOP_Wiki.csv", sep=";", encoding="utf-8-sig")
#df_top_AI = pd.read_csv("./Data/Answers/sci-fi_books_AI_ANSWERS_old.csv", sep=";", encoding="utf-8-sig")
df_top_AI = pd.read_csv("./Data/Answers/sci-fi_books_AI_ANSWERS.csv", sep=";", encoding="utf-8-sig")
df_top_AI_gender = pd.read_csv("./Data/Answers/sci-fi_books_AI_ANSWERS_GENDER.csv", sep=";", encoding="utf-8-sig")

#print(df_top_AI.info())
#print(df.head())

#---------------------------------------------------------------------------------------------------
# Exclude books of before 1860 (allmost none)

mask_all = df_filtered['decade'] >= 1860
df_filtered = df_filtered[mask_all]

mask_top = df_top['decade'] >= 1860
df_top = df_top[mask_top]

mask_top_AI = df_top_AI['decade'] >= 1860
df_top_AI = df_top_AI[mask_top_AI]

#---------------------------------------------------------------------------------------------------
# Include author gender to data

# Create a dictionary from df_top_AI_gender
author_gender_dict = df_top_AI_gender.set_index('author')["gender"].to_dict()

# Map the "author gender" based on 'author' column from df_top_AI
df_top_AI["author gender"] = df_top_AI['author'].map(author_gender_dict).fillna('Uncertain')

#---------------------------------------------------------------------------------------------------
# For the boxplots

# Add a column to each DataFrame to label the dataset
df_filtered['dataset'] = 'Filtered sample'
df_top['dataset'] = 'Top sample'

# Concatenate dataframes
df_filtered_200 = pd.concat([df_filtered, df_top])

#---------------------------------------------------------------------------------------------------
# General information of the FILTERED sample of books
#'title', 'author', 'year', 'decade', 'rate', 'ratings', 'genres', 'synopsis', 'review', 'url'
print("\nFILTERED books.")

book_per_decade = df_filtered['decade'].value_counts()
mean_per_decade = df_filtered.groupby('decade')[['rate', 'ratings']].mean()
sdv_per_decade = df_filtered.groupby('decade')[['rate', 'ratings']].std()

print(book_per_decade.sort_index(ascending = False))

# Create a dictionary by passing Series objects as values
frame_1 = {'quantity': book_per_decade,
           'avr rate': mean_per_decade['rate'],
           'std rate': sdv_per_decade['rate'],
           'avr ratings': mean_per_decade['ratings'],
           'std ratings': sdv_per_decade['ratings']}
 
# Create DataFrame by passing Dictionary
df_all = pd.DataFrame(frame_1)
df_all = (df_all
          .reset_index(drop = False)
          .sort_values(by = ['decade'], ascending = True)
          .reset_index(drop = True))

#print(df_all.info())
#print(df_all)

#---------------------------------------------------------------------------------------------------
# General information of the 200 PER DECADE sample of books
#'title', 'author', 'year', 'decade', 'rate', 'ratings', 'genres', 'synopsis', 'review', 'url'
print("\n200 PER DECADE books.")

book_per_decade_200 = df_top['decade'].value_counts()
mean_per_decade_200 = df_top.groupby('decade')[['rate', 'ratings']].mean()
sdv_per_decade_200 = df_top.groupby('decade')[['rate', 'ratings']].std()

print(book_per_decade_200.sort_index(ascending = False))

# Create a dictionary by passing Series objects as values
frame_2 = {'quantity': book_per_decade_200,
           'avr rate': mean_per_decade_200['rate'],
           'std rate': sdv_per_decade_200['rate'],
           'avr ratings': mean_per_decade_200['ratings'],
           'std ratings': sdv_per_decade_200['ratings']}
 
# Create DataFrame by passing Dictionary
df_all_200 = pd.DataFrame(frame_2)
df_all_200 = (df_all_200
              .reset_index(drop = False)
              .sort_values(by = ['decade'], ascending = True)
              .reset_index(drop = True))

#---------------------------------------------------------------------------------------------------
# Process for the complex figures
#---------------------------------------------------------------------------------------------------
# Author and protagonist gender

df_top_AI_new = df_top_AI.copy()
df_top_AI_new["genders"] = df_top_AI_new["author gender"] + " / " + df_top_AI_new['18 protagonist gender']

# Count the occurrences of each category per decade
category_counts_genders = pd.crosstab(df_top_AI_new['decade'], df_top_AI_new["genders"])

#------------------------------------
# Define a list of the specific 'Other' columns you want to combine
other_columns_to_combine = ['Other / Male',
                            'Other / Female',
                            'Other / Other',
                            'Other / Uncertain',
                            'Other / Not applicable']

# Create the 'Other / All combined' column, handling missing columns gracefully
category_counts_genders['Other / All combined'] = 0
for col in other_columns_to_combine:
    if col in category_counts_genders.columns:
        category_counts_genders['Other / All combined'] = category_counts_genders['Other / All combined'] + category_counts_genders[col]

# Drop the original 'Other' columns, handling missing columns gracefully
columns_to_drop = other_columns_to_combine
columns_to_drop_existing = [col for col in columns_to_drop if col in category_counts_genders.columns]
category_counts_genders = category_counts_genders.drop(columns=columns_to_drop_existing, errors='ignore')

#------------------------------------
# Define a list of the specific 'Uncertain' columns you want to combine
uncertain_columns_to_combine = ['Uncertain / Male',
                                'Uncertain / Female',
                                'Uncertain / Other',
                                'Uncertain / Uncertain',
                                'Uncertain / Not applicable']

# Create the 'Uncertain / All combined' column, handling missing columns gracefully
category_counts_genders['Uncertain / All combined'] = 0
for col in uncertain_columns_to_combine:
    if col in category_counts_genders.columns:
        category_counts_genders['Uncertain / All combined'] = category_counts_genders['Uncertain / All combined'] + category_counts_genders[col]

# Drop the original 'Uncertain' columns, handling missing columns gracefully
columns_to_drop = uncertain_columns_to_combine
columns_to_drop_existing = [col for col in columns_to_drop if col in category_counts_genders.columns]
category_counts_genders = category_counts_genders.drop(columns=columns_to_drop_existing, errors='ignore')

#------------------------------------
# Normalize the counts to get percentages
category_percent_genders = category_counts_genders.div(category_counts_genders.sum(axis=1), axis=0) * 100

#print(category_percent_genders)
print("\ngenders", df_top_AI_new["genders"].unique())
#print("\n")

#---------------------------------------------------------------------------------------------------
# Variation in Answers

# File names in the normally ordered alternatives
file_names = [
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_01.csv",
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_02.csv",
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_03.csv",
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_04.csv",
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_05.csv",
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_06.csv",
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_07.csv",
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_08.csv",
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_09.csv",
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_10.csv",
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_11.csv",
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_12.csv",
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_13.csv",
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_14.csv",
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_15.csv",
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_16.csv",
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_17.csv",
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_18.csv",
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_19.csv",
    "./Data/Variability_in_Answers/sci-fi_books_AI_ANSWERS_TEST_20.csv"
    ]

column_order = [
    '1 accuracy',
    '2 discipline',
    '3 light heavy',
    '4 time',
    '5 mood',
    '6 ending',

    '7 social political',
    '8 politically unified',
    '9 on Earth',
    '10 post apocalyptic',
    '11 conflict',

    '12 aliens',
    '13 aliens are',
    '14 robots and AI',
    '15 robots and AI are',

    '16 protagonist',
    '17 protagonist nature',
    '18 protagonist gender',
    '19 protagonist is',

    '20 virtual',
    '21 virtual is',
    '22 biotech',
    '23 biotech is',
    '24 transhuman',
    '25 transhuman is',
    '26 tech and science',

    '27 social issues',
    '28 enviromental'
]

dfs = [] # List of DataFrames

# Process the DataFrames for variation in answers
#for file_name in file_names_s:
for file_name in file_names:
    df_file_name = pd.read_csv(file_name, sep=";", encoding="utf-8-sig")

    df_file_name['year'] = df_file_name['year'].astype('string')
    df_file_name['id'] = df_file_name['title'] + " (" + df_file_name['year'] + ") " + df_file_name['author']
    df_file_name = df_file_name.set_index('id')

    df_file_name = df_file_name.reindex(columns=column_order)

    #print(df_file_name.iloc[18,4])

    dfs.append(df_file_name)

 # Create a 3D array (rows x columns x runs)
data_array = np.stack([df.to_numpy() for df in dfs], axis=-1) # Shape: (rows, columns, runs)

#-------------------------
# Difference two by two of the answers

# Initialize a dataframe to store the sum of differences
comparison_sum = pd.DataFrame(0, index=dfs[0].index, columns=dfs[0].columns)

# Count the number of comparisons made (for averaging later)
num_comparisons = 0

# Compare each pair of dataframes
for i in range(len(dfs)):
    for j in range(i + 1, len(dfs)):
        # Compare dataframes element-wise, convert booleans to integers (1 for different, 0 for same)
        difference = (dfs[i] != dfs[j]).astype(int)
        
        # Add to comparison sum
        comparison_sum += difference
        
        # Increment the comparison count
        num_comparisons += 1

# Calculate the mean difference for each cell
df_mean_difference = comparison_sum / num_comparisons

# Add column and row of means
df_mean_difference['Mean'] = df_mean_difference.mean(axis=1)
df_mean_difference.loc['Mean'] = df_mean_difference.mean(axis=0)

#-----------------------------
# Percent Agreement / Mode Consistency

# Initialize empty arrays to store modes and percent agreements
rows, cols = data_array.shape[0], data_array.shape[1]
modes = np.empty((rows, cols), dtype=object)
percent_agreement = np.empty((rows, cols))

# Compute the mode and agreement level for each cell
for i in range(rows):
    for j in range(cols):
        answers = data_array[i, j, :] # All answers for a particular cell across dataframes
        unique_values, counts = np.unique(answers, return_counts=True)  # Unique values and their counts
        mode_index = np.argmax(counts) # Index of the mode (most frequent value)
        modes[i, j] = unique_values[mode_index] # Mode value
        percent_agreement[i, j] = (counts[mode_index] / len(dfs)) * 100 # Agreement as a percentage

# Convert modes and percent_agreement back to DataFrames for easier interpretation
df_modes = pd.DataFrame(modes, columns=dfs[0].columns, index=dfs[0].index)
# Convert to DataFrames
df_percent_agreement = pd.DataFrame(percent_agreement, index=dfs[0].index, columns=dfs[0].columns)

# Add column and row of means
df_percent_agreement['Mean'] = df_percent_agreement.mean(axis=1)
df_percent_agreement.loc['Mean'] = df_percent_agreement.mean(axis=0)
print(df_percent_agreement)
#-----------------------------
# Shannon Entropy (Diversity Index)
def shannon_entropy(values):
    _, counts = np.unique(values, return_counts=True) # values, counts
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

# Apply the entropy function along the last axis (runs)
entropy_values = np.apply_along_axis(shannon_entropy, 2, data_array)

# Convert to DataFrames
df_entropy = pd.DataFrame(entropy_values, index=dfs[0].index, columns=dfs[0].columns)

# Add column and row of means
df_entropy['Mean'] = df_entropy.mean(axis=1)
df_entropy.loc['Mean'] = df_entropy.mean(axis=0)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
# Make figures
print("Making the figures...")
#---------------------------------------------------------------------------------------------------
# Custom dark gray color
custom_dark_gray = (0.2, 0.2, 0.2)

#---------------------------------------------------------------------------------------------------
# Figure 1, Quantities
print("  Making book counts...")

# Creates a figure object with size 12x6 inches
figure_c1 = plt.figure(1, figsize = (12, 6))
gs = figure_c1.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure_c1.add_subplot(gs[0])

# Bar plot all sample
bars = ax1.bar(x = df_all['decade'],
               height = df_all['quantity'],
               width = 9, 
               align = 'center',
               color = "#385AC2",
               alpha = 1.0,
               edgecolor = custom_dark_gray,
               linewidth = 0.0,
               label = "Filtered sample")

# Bar plot top 200
ax1.bar(x = df_all_200['decade'],
        height = df_all_200['quantity'], 
        width = 9, 
        align = 'center', 
        color = "none",
        edgecolor = "#AE305D", 
        hatch = "////",
        linewidth = 0.0,
        label = "Top sample")

# Add the count labels above each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f'{int(height)}',
        ha='center',
        va='bottom',
        color=custom_dark_gray
    )

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Quantity", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Number of Sci-fi Books per Decade In The Samples", fontsize = 14, pad = 5, color = custom_dark_gray)
#ax1.grid(True, linestyle = "dotted", linewidth = "1.0", zorder = 0, alpha = 1.0)

# Legend-------------------------------------------
ax1.legend(frameon = False, 
           #labelspacing = 10.0,
           loc = 'upper left')

# Axes-------------------------------------------
ax1.minorticks_on()
ax1.tick_params(which = "major", direction = "out", length = 3, labelsize = 10, color = custom_dark_gray)
ax1.tick_params(which = "minor", direction = "out", length = 0, color = custom_dark_gray)
ax1.tick_params(which = "both", bottom = True, top = False, left = False, right = False, color = custom_dark_gray)
ax1.tick_params(labelbottom = True, labeltop = False, labelleft = False, labelright = False, color = custom_dark_gray)
ax1.tick_params(axis = 'both', colors = custom_dark_gray)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
#ax1.spines['right'].set_color(custom_dark_gray)
#ax1.spines['top'].set_color(custom_dark_gray)
#ax1.spines['left'].set_color(custom_dark_gray)
ax1.spines['bottom'].set_color(custom_dark_gray)

# Get all decades between the minimum and maximum
all_decades = np.arange(df_all['decade'].min(), df_all['decade'].max() + 10, 10)
# Set x-ticks to show each decade
plt.xticks(all_decades, rotation=90)

# Save image-------------------------------------------
plt.savefig("./Figures/00 Quantities.png", bbox_inches = 'tight')
#plt.savefig("./Figures/00 Quantities.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/00 Quantities.svg", format = 'svg', transparent = True, bbox_inches = 'tight')
plt.close(figure_c1)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
# Figure 2, Quantities
print("  Making rate and ratings...")

# Creates a figure object with size 14x8 inches
figure_c2 = plt.figure(2, figsize = (14, 8))
gs = figure_c2.add_gridspec(ncols = 2, nrows = 1)

#figure_c2.subplots_adjust(hspace = 0.5)

#-----------------------------------------
# Create the main plot
ax1 = figure_c2.add_subplot(gs[0])

# Specify custom colors for each dataset
custom_palette = {'Filtered sample': '#385AC2',
                  'Top sample': '#AE305D'}

# Create the boxplot with hue
sns.boxplot(x = 'decade', 
            y = 'rate', 
            hue = 'dataset', 
            data = df_filtered_200, 
            palette = custom_palette,
            #width = 0.8,
            #gap = 0.3,
            fliersize = 2.0,
            fill = True,
            linecolor = custom_dark_gray)

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
ax1.set_ylabel("Average rate", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Distribution of Average Rate per Decade", fontsize = 14, pad = 5, color = custom_dark_gray)
#ax1.grid(True, linestyle = "dotted", linewidth = "1.0", zorder = 0, alpha = 1.0)
ax1.yaxis.grid(True, linestyle = "dotted", linewidth = "1.0", zorder = 0, alpha = 0.5)

# Legend-------------------------------------------
ax1.legend(frameon = False, 
           #labelspacing = 10.0,
           loc = 'upper left')

# Axes-------------------------------------------
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)

ax1.minorticks_on()
ax1.tick_params(which = "major", direction = "out", length = 3, labelsize = 10, color = custom_dark_gray)
ax1.tick_params(which = "minor", direction = "out", length = 0, color = custom_dark_gray)
ax1.tick_params(which = "both", bottom = True, top = False, left = False, right = False, color = custom_dark_gray)
ax1.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False, color = custom_dark_gray)
ax1.tick_params(axis = 'both', colors = custom_dark_gray)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
#ax1.spines['top'].set_visible(False)
#ax1.spines['right'].set_color(custom_dark_gray)
#ax1.spines['left'].set_color(custom_dark_gray)
ax1.spines['bottom'].set_color(custom_dark_gray)

#---------------------------------------------------------------------------------------------------
# Create the main plot
ax2 = figure_c2.add_subplot(gs[1])

# Specify custom colors for each dataset
custom_palette = {'Filtered sample': '#385AC2',
                  'Top sample': '#AE305D'}

# Create the boxplot with hue
sns.boxplot(x = 'decade', 
            y = 'ratings', 
            hue = 'dataset', 
            data = df_filtered_200, 
            palette = custom_palette,
            #width = 0.8,
            #gap = 0.3,
            fliersize = 2.0,
            fill = True,
            linecolor = custom_dark_gray,
            log_scale = True)

# Design-------------------------------------------
ax2.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
ax2.set_ylabel("Number of ratings", fontsize = 12, color = custom_dark_gray)
ax2.set_title("Distribution of Number of Ratings per Decade", fontsize = 14, pad = 5, color = custom_dark_gray)
#ax2.grid(True, linestyle = "dotted", linewidth = "1.0", zorder = 0, alpha = 1.0)
ax2.yaxis.grid(True, linestyle = "dotted", linewidth = "1.0", zorder = 0, alpha = 0.5)

# Legend-------------------------------------------
ax2.legend(frameon = False, 
           #labelspacing = 10.0,
           loc = 'upper left')

# Axes-------------------------------------------
for tick in ax2.get_xticklabels():
    tick.set_rotation(90)

#ax2.set_yscale('log')

ax2.minorticks_on()
ax2.tick_params(which = "major", direction = "out", length = 3, labelsize = 10, color = custom_dark_gray)
ax2.tick_params(which = "minor", direction = "out", length = 0, color = custom_dark_gray)
ax2.tick_params(which = "both", bottom = True, top = False, left = False, right = False, color = custom_dark_gray)
ax2.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False, color = custom_dark_gray)
ax2.tick_params(axis = 'both', colors = custom_dark_gray)

ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
#ax2.spines['top'].set_visible(False)
#ax2.spines['right'].set_color(custom_dark_gray)
#ax2.spines['left'].set_color(custom_dark_gray)
ax2.spines['bottom'].set_color(custom_dark_gray)

# Save image-------------------------------------------
plt.savefig("./Figures/00 Rates and Ratings.png", bbox_inches = 'tight')
#plt.savefig("./Figures/00 Rates and Ratings.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/00 Rates and Ratings.svg", format = 'svg', transparent = True, bbox_inches = 'tight')
plt.close(figure_c2)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def figure_maker (number: int, 
                  column_name: str,
                  df_top: pd.DataFrame,
                  category_order: List[str], 
                  custom_colors: List[str], 
                  title: str, 
                  printing_name: str) -> None:
    
    """
    Function to make most of the figures.

    Args:
        number (int): Number of the figure.
        df_top (Dataframe): Dataframe with all data.
        column_name (str): Name of data column to be used.
        category_order (List[str]): Order of the categories.
        custom_colors (List[str]): Color of the categories.
        title (str): Figure title.
        printing_name (str): Name to be printed as the file name and label.

    Returns:
        No return.
    """

    #---------------------------------------------------------------------------------------------------
    # Creates a figure object with size 12x6 inches
    figure = plt.figure(number, figsize = (12, 6))
    gs = figure.add_gridspec(ncols = 1, nrows = 1)

    # Create the main plot
    ax1 = figure.add_subplot(gs[0])

    # Custom dark gray color
    custom_dark_gray = (0.2, 0.2, 0.2)

    # Custom label spacing
    custom_label_spacing = {
        1: 1.0,
        2: 12.0,
        3: 10.0,
        4: 8.0,
        5: 6.0,
        6: 4.5,
        7: 4.0,
        8: 3.2,
        9: 2.8,
        10: 2.4,
        11: 2.0,
        12: 1.6
    }

    #-------------------------------------------
     # Count the occurrences of each category per decade
    category_counts = pd.crosstab(df_top['decade'], df_top[column_name])
    # Normalize the counts to get percentages
    category_percent = category_counts.div(category_counts.sum(axis = 1), axis = 0) * 100

    #-------------------------------------------
    # Create a new DataFrame with all desired categories, filling with 0 if missing
    all_categories_df = pd.DataFrame(columns=category_order).astype(float)
    category_percent = pd.concat([all_categories_df, category_percent]).convert_dtypes().convert_dtypes().fillna(0.0)

    # Reorder the columns in the DataFrame according to the desired category order
    category_percent = category_percent[category_order]

    # Bar plot-------------------------------------------
    category_percent.plot(kind = 'bar',
                          stacked = True,
                          ax = ax1,
                          color = custom_colors,
                          width = 1.0,
                          alpha = 1.0,
                          label = printing_name)

    # Design-------------------------------------------
    ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
    #ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
    ax1.set_title(title, fontsize = 14, pad = 5, color = custom_dark_gray)
    #ax1.yaxis.grid(True, linestyle = "dotted", linewidth = "1.0", zorder = 0, alpha = 1.0)

    # Format the y-axis to show percentages
    ax1.yaxis.set_major_formatter(PercentFormatter())

    # Legend-------------------------------------------
    # Get handles and labels
    handles, labels = ax1.get_legend_handles_labels()

    # Reverse the order
    handles.reverse()
    labels.reverse()

    # Pass the reversed handles and labels to the legend
    ax1.legend(handles, 
            labels, 
            bbox_to_anchor = (0.99, 0.00, 0.50, 0.95), 
            frameon = False, 
            labelspacing = custom_label_spacing[len(category_order)],
            loc = 'center left')

    # Axes-------------------------------------------
    ax1.minorticks_on()
    ax1.tick_params(which = "major", direction = "out", length = 3, labelsize = 10, color = custom_dark_gray)
    ax1.tick_params(which = "minor", direction = "out", length = 0, color = custom_dark_gray)
    ax1.tick_params(which = "both", bottom = True, top = False, left = True, right = False, color = custom_dark_gray)
    ax1.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False, color = custom_dark_gray)
    ax1.tick_params(axis = 'both', colors = custom_dark_gray)

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    #ax1.spines['left'].set_visible(False)
    ax1.spines['left'].set_color(custom_dark_gray)
    ax1.spines['bottom'].set_color(custom_dark_gray)

    # Save image-------------------------------------------
    plt.savefig(f"./Figures/{printing_name}.png", bbox_inches = 'tight')
    #plt.savefig(f"./Figures/{printing_name}.eps", transparent = True, bbox_inches = 'tight')
    # Transparence will be lost in .eps, save in .svg for transparences
    #plt.savefig(f"./Figures/{printing_name}.svg", format = 'svg', transparent = True, bbox_inches = 'tight')
    plt.close(figure)

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------
# Figure 3, Series (top)
print("  Making series (top)...")

# Desired order of the categories
category_order = ['yes',
                  'no']

# Custom colors for each category
custom_colors = ['#AE305D', 
                 '#385AC2']

figure_maker (3, # number
              "series", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Is the novel part of a series? (top sample)", # title
              "00 series (top 200)") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 4, Series (filtered)
print("    Making series (filtered)...")

# Desired order of the categories
category_order = ['yes',
                  'no']

# Custom colors for each category
custom_colors = ['#AE305D', 
                 '#385AC2']

figure_maker (50, # number
              "series", # column_name
              df_filtered, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Is the novel part of a series? (filtered sample)", # title
              "00 series (filtered)") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 5, Series (top 50)
print("    Making series (top 50)...")

# Create decade columns to use in the groupby below and keep the original decade
df_top_AI_2 = df_top_AI.copy(deep=True)
df_top_AI_2['decade_gb'] = df_top_AI_2['decade']

# Group by 'decade_gb', sort by 'ratings' in descending order, and select the top 50 per group
df_top_50 = (df_top_AI_2.groupby('decade_gb', group_keys=False)
                .apply((lambda x: x.sort_values('ratings', ascending=False).head(50)), 
                    include_groups=False))

#---------------------------------------------
# Desired order of the categories
category_order = ['yes',
                  'no']

# Custom colors for each category
custom_colors = ['#AE305D', 
                 '#385AC2']

figure_maker (51, # number
              "series", # column_name
              df_top_50, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Is the novel part of a series? (top 50)", # title
              "00 series (top 50)") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 4, author gender
print("  Making author gender...")

# Desired order of the categories
category_order = ['Male',
                  'Female',
                  'Other',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#AE305D',
                 '#8B3FCF',
                 '#FFD700']

figure_maker (4, # number
              "author gender", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the gender of the author?", # title
              "00 author gender") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 5, author gender accuracy
print("  Making author gender accuracy...")

# Desired order of the categories
category_order = ['Male',
                  'Female',
                  'Other',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#AE305D',
                 '#8B3FCF',
                 '#FFD700']

# Define the condition to select rows
mask_1 = df_top_AI['1 accuracy'] == "High"
mask_2 = df_top_AI['1 accuracy'] == "Very high"
# Filter the dataframe to exclude those rows
df_top_AI_masked = df_top_AI[mask_1 | mask_2]

figure_maker (5, # number
              "author gender", # column_name
              df_top_AI_masked, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the gender of the author for (very) high accuracy sci-fi?", # title
              "00 author gender accuracy") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 6, author gender accuracy
print("  Making author gender discipline...")

# Desired order of the categories
category_order = ['Male',
                  'Female',
                  'Other',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#AE305D',
                 '#8B3FCF',
                 '#FFD700']

# Define the condition to select rows
mask_1 = df_top_AI['2 discipline'] == "Leaning hard sciences"
mask_2 = df_top_AI['2 discipline'] == "Hard sciences"
# Filter the dataframe to exclude those rows
df_top_AI_masked = df_top_AI[mask_1 | mask_2]

figure_maker (6, # number
              "author gender", # column_name
              df_top_AI_masked, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the gender of the author for (leaning) hard sciences sci-fi?", # title
              "00 author gender discipline") # printing_name and label

#---------------------------------------------------------------------------------------------------
# figures for the questions/answers
#---------------------------------------------------------------------------------------------------
# Figure 7 - 1 accuracy
print("  Making 1 accuracy...")

# Desired order of the categories
category_order = ['Very low',
                  'Low',
                  'Moderate', 
                  'High',
                  'Very high',
                  
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',
                 
                 '#FFD700']

figure_maker (7, # number
              "1 accuracy", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "How important is scientific accuracy and plausibility in the story?", # title
              "01 accuracy") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 8 - 2 discipline
print("  Making 2 discipline...")

# Desired order of the categories
category_order = ['Soft sciences',
                  'Leaning soft sciences',
                  'Mixed', 
                  'Leaning hard sciences',
                  'Hard sciences',

                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#FFD700']

figure_maker (8, # number
              "2 discipline", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the main disciplinary focus of the story?", # title
              "02 discipline") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 9 - 3 light heavy
print("  Making 3 light heavy...")

# Desired order of the categories
category_order = ['Very heavy',
                  'Heavy',
                  'Balanced', 
                  'Light',
                  'Very light',

                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#FFD700']

figure_maker (9, # number
              "3 light heavy", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Is the story more of a light or heavy reading experience?", # title
              "03 light heavy") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 10 - 4 time
print("  Making 4 time...")

# Desired order of the categories
category_order = ['Distant past',
                  'Far past',
                  'Near past',
                  'Present',
                  'Near future',
                  'Far future',
                  'Distant future',

                  'Multiple timelines',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D', # Distant past
                 '#CF5D5F',
                 '#E3937B',
                 '#8B3FCF', # Present
                 '#6CACEB',
                 '#5580D0',
                 '#385AC2', # Distant future

                 '#008000', # Multiple timelines
                 '#FFD700'] # Uncertain

figure_maker (10, # number
              "4 time", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "When does most of the story take place in relation to the year the book was published?", # title
              "04 time") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 11 - 5 mood
print("  Making 5 mood...")

# Desired order of the categories
category_order = ['Very pessimistic',
                  'Pessimistic',
                  'Balanced',
                  'Optimistic',
                  'Very optimistic',

                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#FFD700']

figure_maker (11, # number
              "5 mood", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the mood of the story?", # title
              "05 mood") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 12 - 6 ending
print("  Making 6 ending...")

# Desired order of the categories
category_order = ['Very negative',
                  'Negative',
                  'Ambivalent',
                  'Positive',
                  'Very positive',

                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#FFD700']

figure_maker (12, # number
              "6 ending", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the overall mood and outcome of the story's ending?", # title
              "06 ending") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 13 - 7 social political
print("  Making 7 social political...")

# Desired order of the categories
category_order = ['Dystopic',
                  'Leaning dystopic',
                  'Balanced',
                  'Leaning utopic',
                  'Utopic',

                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#FFD700']

figure_maker (13, # number
              "7 social political", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the social-political scenario depicted in the story?", # title
              "07 social political") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 14 - 8 politically unified
print("  Making 8 politically unified...")

# Desired order of the categories
category_order = ['Yes',
                  'Somewhat',
                  'No',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#8B3FCF',
                 '#385AC2',
                 '#FFD700']

figure_maker (14, # number
              "8 politically unified", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Is a unified, planetary-level or multi-planet state or government depicted in the story?", # title
              "08 politically unified") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 15 - 9 on Earth
print("  Making 9 on Earth...")

# Desired order of the categories
category_order = ['Yes',
                  'Somewhat',
                  'No',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#8B3FCF',
                 '#AE305D',
                 '#FFD700']

figure_maker (15, # number
              "9 on Earth", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Is most of the story set on planet Earth?", # title
              "09 on Earth") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 16 - 10 post apocalyptic
print("  Making 10 post apocalyptic...")

# Desired order of the categories
category_order = ['Yes',
                  'Somewhat',
                  'No',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#8B3FCF',
                 '#385AC2',
                 '#FFD700']

figure_maker (16, # number
              "10 post apocalyptic", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Is the setting of the story post-apocalyptic?", # title
              "10 post apocalyptic") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 17 - 11 conflict
print("  Making 11 conflict...")

# Desired order of the categories
category_order = ['Internal',
                  'Interpersonal',
                  'Societal',
                  'Synthetic',
                  'Technological',
                  'Extraterrestrial',
                  'Natural',
                  'Mixed',

                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D', # Internal
                 '#CF5D5F', # Interpersonal
                 '#E3937B', # Societal
                 '#385AC2', # Synthetic
                 '#5580D0', # Technological
                 '#6CACEB', # Extraterrestrial
                 '#008000', # Natural
                 '#8B3FCF', # Mixed

                 '#FFD700'] # Uncertain

figure_maker (17, # number
              "11 conflict", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the dominant type of conflict in the story?", # title
              "11 conflict") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 18 - 12 aliens
print("  Making 12 aliens...")

# Desired order of the categories
category_order = ['Yes',
                  'Somewhat',
                  'No',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#8B3FCF',
                 '#385AC2',
                 '#FFD700']

figure_maker (18, # number
              "12 aliens", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Does the story depict extraterrestrial life forms or alien technology?", # title
              "12 aliens") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 19 - 13a aliens are
print("  Making 13a aliens are...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Non-moral',
                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700',
                 '#D3D3D3']

figure_maker (19, # number
              "13 aliens are", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts extraterrestrial life forms, how are they generally portrayed?", # title
              "13a aliens are") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 20 - 13b aliens are
print("    Making 13b aliens are...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Non-moral',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700']

# Filter the dataframe to exclude some rows
df_top_filtered = df_top_AI[df_top_AI["13 aliens are"] != "Not applicable"]

figure_maker (20, # number
              "13 aliens are", # column_name
              df_top_filtered, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts extraterrestrial life forms, how are they generally portrayed?", # title
              "13b aliens are") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 21 - 4 time
print("  14 robots and AI...")

# Desired order of the categories
category_order = ['Yes',
                  'Somewhat',
                  'No',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#8B3FCF',
                 '#385AC2',
                 '#FFD700']

figure_maker (21, # number
              "14 robots and AI", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Does the story depict robots or artificial intelligences?", # title
              "14 robots and AI") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 22 - 15a robots and AI are
print("  Making 15a robots and AI are...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad', 
                  'Ambivalent', 
                  'Leaning good',
                  'Good',
                
                  'Non-moral',
                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700',
                 '#D3D3D3']

figure_maker (22, # number
              "15 robots and AI are", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts robots or artificial intelligences, how are they generally portrayed?", # title
              "15a robots and AI are") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 23 - 15b robots and AI are
print("    Making 15b robots and AI are...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad', 
                  'Ambivalent', 
                  'Leaning good',
                  'Good',
                
                  'Non-moral',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700']

# Filter the dataframe to exclude some rows
df_top_filtered = df_top_AI[df_top_AI["15 robots and AI are"] != "Not applicable"]

figure_maker (23, # number
              "15 robots and AI are", # column_name
              df_top_filtered, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts robots or artificial intelligences, how are they generally portrayed?", # title
              "15b robots and AI are") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 24 - 16 protagonist
print("  Making 16 protagonist...")

# Desired order of the categories
category_order = ['Yes',
                  'Somewhat',
                  'No',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#8B3FCF',
                 '#385AC2',
                 '#FFD700']

figure_maker (24, # number
              "16 protagonist", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Is there a single main character or protagonist in the story?", # title
              "16 protagonist") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 25 - 17a protagonist nature
print("  Making 17a protagonist nature...")

# Desired order of the categories
category_order = ['Human', 
                  'Non-human',

                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#AE305D',

                 '#FFD700',
                 '#D3D3D3']

figure_maker (25, # number
              "17 protagonist nature", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story has a single protagonist or main character, what is their nature?", # title
              "17a protagonist nature") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 26 - 17b protagonist nature
print("  Making 17b protagonist nature...")

# Desired order of the categories
category_order = ['Human', 
                  'Non-human',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#AE305D',
                 '#FFD700']

# Filter the dataframe to exclude some rows
df_top_filtered = df_top_AI[df_top_AI["17 protagonist nature"] != "Not applicable"]

figure_maker (26, # number
              "17 protagonist nature", # column_name
              df_top_filtered, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story has a single protagonist or main character, what is their nature?", # title
              "17b protagonist nature") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 27 - 18a protagonist gender
print("  Making 18a protagonist gender...")

# Desired order of the categories
category_order = ['Male', 
                  'Female',
                  'Other',

                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#AE305D',
                 '#8B3FCF',

                 '#FFD700',
                 '#D3D3D3']

figure_maker (27, # number
              "18 protagonist gender", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story has a single protagonist or main character, what is their gender?", # title
              "18a protagonist gender") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 28 - 18ba protagonist gender
print("    Making 18b protagonist gender...")

# Desired order of the categories
category_order = ['Male', 
                  'Female',
                  'Other',

                  'Uncertain']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#AE305D',
                 '#8B3FCF',

                 '#FFD700']

# Filter the dataframe to exclude some rows
df_top_filtered = df_top_AI[df_top_AI["18 protagonist gender"] != "Not applicable"]

figure_maker (28, # number
              "18 protagonist gender", # column_name
              df_top_filtered, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story has a single protagonist or main character, what is their gender?", # title
              "18b protagonist gender") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 29 - 18c protagonist gender accuracy
print("    Making 18c protagonist gender accuracy...")

# Desired order of the categories
category_order = ['Male', 
                  'Female',
                  'Other',

                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#AE305D',
                 '#8B3FCF',

                 '#FFD700',
                 '#D3D3D3']

# Define the condition to select rows
mask_1 = df_top_AI['1 accuracy'] == "High"
mask_2 = df_top_AI['1 accuracy'] == "Very high"
# Filter the dataframe to exclude those rows
df_top_AI_masked = df_top_AI[mask_1 | mask_2]

figure_maker (29, # number
              "18 protagonist gender", # column_name
              df_top_AI_masked, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the gender of the single protagonist or main character \nfor (very) high accuracy sci-fi?", # title
              "18c protagonist gender accuracy") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 30 - 18d protagonist gender discipline
print("    Making 18d protagonist gender discipline...")

# Desired order of the categories
category_order = ['Male', 
                  'Female',
                  'Other',

                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#AE305D',
                 '#8B3FCF',

                 '#FFD700',
                 '#D3D3D3']

# Define the condition to select rows
mask_1 = df_top_AI['2 discipline'] == "Leaning hard sciences"
mask_2 = df_top_AI['2 discipline'] == "Hard sciences"
# Filter the dataframe to exclude those rows
df_top_AI_masked = df_top_AI[mask_1 | mask_2]

figure_maker (30, # number
              "18 protagonist gender", # column_name
              df_top_AI_masked, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "What is the gender of the single protagonist or main character \nfor (leaning) hard discipline sci-fi?", # title
              "18d protagonist gender discipline") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 31 - 19a protagonist is
print("  Making 19a protagonist is...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Non-moral',
                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700',
                 '#D3D3D3']

figure_maker (31, # number
              "19 protagonist is", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story has a single protagonist or main character, how are they generally portrayed?", # title
              "19a protagonist is") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 32 - 19b protagonist is
print("  Making 19b protagonist is...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Non-moral',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700']

# Filter the dataframe to exclude those rows
df_top_filtered = df_top_AI[df_top_AI['19 protagonist is'] != "Not applicable"]

figure_maker (32, # number
              "19 protagonist is", # column_name
              df_top_filtered, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story has a single protagonist or main character, how are they generally portrayed?", # title
              "19b protagonist is") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 33 - 20 virtual
print("  Making 20 virtual...")

# Desired order of the categories
category_order = ['Yes',
                  'Somewhat',
                  'No',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#8B3FCF',
                 '#385AC2',
                 '#FFD700']

figure_maker (33, # number
              "20 virtual", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Does the story depict virtual or augmented reality?", # title
              "20 virtual") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 34 - 21a virtual is
print("  Making 21a virtual is...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Instrumental',
                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700',
                 '#D3D3D3']

figure_maker (34, # number
              "21 virtual is", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts virtual or augmented reality, how are they generally depicted?", # title
              "21a virtual is") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 35 - 21b virtual is
print("    Making 21b virtual is...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Instrumental',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700']

# Filter the dataframe to exclude those rows
df_top_filtered = df_top_AI[df_top_AI['21 virtual is'] != "Not applicable"]

figure_maker (35, # number
              "21 virtual is", # column_name
              df_top_filtered, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts virtual or augmented reality, how are they generally depicted?", # title
              "21b virtual is") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 36 - 22 biotech
print("  Making 22 biotech...")

# Desired order of the categories
category_order = ['Yes',
                  'Somewhat',
                  'No',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#8B3FCF',
                 '#385AC2',
                 '#FFD700']

figure_maker (36, # number
              "22 biotech", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Does the story depict biotechnology, genetic engineering, or human biological alteration?", # title
              "22 biotech") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 37 - 23a biotech is
print("  Making 23a biotech is...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Instrumental',
                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700',
                 '#D3D3D3']

figure_maker (37, # number
              "23 biotech is", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts biotechnology, genetic engineering, or human biological alteration, \nhow are they generally depicted?", # title
              "23a biotech is") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 38 - 23b biotech is
print("    Making 23b biotech is...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Instrumental',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700']

# Filter the dataframe to exclude those rows
df_top_filtered = df_top_AI[df_top_AI['23 biotech is'] != "Not applicable"]

figure_maker (38, # number
              "23 biotech is", # column_name
              df_top_filtered, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts biotechnology, genetic engineering, or human biological alteration, \nhow are they generally depicted?", # title
              "23b biotech is") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 39 - 24 transhuman
print("  Making 24 transhuman...")

# Desired order of the categories
category_order = ['Yes',
                  'Somewhat',
                  'No',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#8B3FCF',
                 '#385AC2',
                 '#FFD700']

figure_maker (39, # number
              "24 transhuman", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "Does the story depict transhumanism or the transcendence of human limitations?", # title
              "24 transhuman") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 40 - 25a transhuman is
print("  Making 25a transhuman is...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Instrumental',
                  'Uncertain',
                  'Not applicable']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700',
                 '#D3D3D3']

figure_maker (40, # number
              "25 transhuman is", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts transhumanism or the transcendence of human limitations, \nhow are they generally depicted?", # title
              "25a transhuman is") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 41 - 25b transhuman is
print("    Making 25b transhuman is...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad',
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Instrumental',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700']

# Filter the dataframe to exclude those rows
df_top_filtered = df_top_AI[df_top_AI['25 transhuman is'] != "Not applicable"]

figure_maker (41, # number
              "25 transhuman is", # column_name
              df_top_filtered, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "If the story depicts transhumanism or the transcendence of human limitations, \nhow are they generally depicted?", # title
              "25b transhuman is") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 42 - 26 tech and science
print("  Making 26 tech and science...")

# Desired order of the categories
category_order = ['Bad', 
                  'Leaning bad', 
                  'Ambivalent', 
                  'Leaning good',
                  'Good',

                  'Instrumental',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#AE305D',
                 '#CF5D5F',
                 '#8B3FCF',
                 '#5580D0',
                 '#385AC2',

                 '#008000',
                 '#FFD700']

figure_maker (42, # number
              "26 tech and science", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "How are science and technology depicted in the story?", # title
              "26 tech and science") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 43 - 27 social issues
print("  Making 27 social issues...")

# Desired order of the categories
category_order = ['Core', 
                  'Major',
                  'Minor',

                  'Absent',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#5580D0',
                 '#6CACEB',

                 '#AE305D',
                 '#FFD700']

figure_maker (43, # number
              "27 social issues", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "How central is the critique of specific social issues in the story?", # title
              "27 social issues") # printing_name and label

#---------------------------------------------------------------------------------------------------
# Figure 44 - 28 enviromental
print("  Making 28 enviromental...")

# Desired order of the categories
category_order = ['Core', 
                  'Major',
                  'Minor',

                  'Absent',
                  'Uncertain']

# Custom colors for each category
custom_colors = ['#385AC2',
                 '#5580D0',
                 '#6CACEB',

                 '#AE305D',
                 '#FFD700']

figure_maker (44, # number
              "28 enviromental", # column_name
              df_top_AI, # df_top
              category_order, # category_order
              custom_colors, # custom_colors
              "How central are ecological or environmental themes in the story?", # title
              "28 enviromental") # printing_name and label

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
# For the Tests and more
#---------------------------------------------------------------------------------------------------
# Figure 45 - Author and protagonist heatmap
print("  Making author and protagonist heatmap...")

#-------------------------------------------
# Creates a figure object with size 18x5 inches
figure_t1 = plt.figure(45, figsize = (18, 5))
gs = figure_t1.add_gridspec(ncols = 2, nrows = 1)

# Define the desired order for Author Gender (y-axis)
author_gender_order = ['Male', 
                       'Female', 
                       'Other', 
                       'Uncertain']

# Define the desired order for Protagonist Gender (x-axis)
protagonist_gender_order = ['Male', 
                            'Female', 
                            'Other', 
                            #'Uncertain', 
                            'Not applicable']

#-------------------------------------------
# Step 1: Create a contingency table (cross-tab)
contingency_table = pd.crosstab(df_top_AI["author gender"], df_top_AI['18 protagonist gender'])

contingency_table = contingency_table.div(contingency_table.sum(axis=1), axis=0) * 100

# Reindex rows and columns according to the desired order
contingency_table = contingency_table.reindex(index=author_gender_order, columns=protagonist_gender_order)

# Create a formatted string version of the percentages with % symbol for annotation
annot = contingency_table.map(lambda x: f'{x:.2f}%')

print("\n")
print("Contingency Table:")
print(contingency_table)

#-------------------------------------------
# Create the main plot
ax1 = figure_t1.add_subplot(gs[0])
#sns.heatmap(contingency_table, annot=True, cmap='coolwarm', fmt="d")
sns.heatmap(contingency_table, cmap='coolwarm', annot=annot, fmt="")

# Set the title and labels
ax1.set_title('Contingency Table Heatmap (All decades)')
ax1.set_xlabel('Protagonist Gender')
ax1.set_ylabel('Author Gender')

# Rotate and align the labels
for tick in ax1.get_xticklabels():
    tick.set_rotation(45)
    tick.set_ha('right') # Horizontal alignment (right aligns better for rotated labels)
    tick.set_va('top') # Vertical alignment to ensure no overlap

#-------------------------------------------
# Step 2: Perform Chi-Square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Results
print("Chi-Square Test Results:")
print(f"Chi2 Statistic: {chi2}")
print(f"P-Value: {p_value}")

# Interpret the result
if p_value < 0.05:
    print("There IS a significant correlation between Author Gender and Protagonist Gender.")
else:
    print("There is NO significant correlation between Author Gender and Protagonist Gender.")

#-------------------------------------------
mask = df_top_AI['decade'] >= 2000
df_top_AI_2000 = df_top_AI[mask]

# Step 1: Create a contingency table (cross-tab)
contingency_table_2000 = pd.crosstab(df_top_AI_2000["author gender"], df_top_AI_2000['18 protagonist gender'])

contingency_table_2000 = contingency_table_2000.div(contingency_table_2000.sum(axis = 1), axis = 0) * 100

# Reindex rows and columns according to the desired order
contingency_table_2000 = contingency_table_2000.reindex(index=author_gender_order, columns=protagonist_gender_order)

# Create a formatted string version of the percentages with % symbol for annotation
annot_2000 = contingency_table_2000.map(lambda x: f'{x:.2f}%')

print("\n")
print("Contingency Table 2000s:")
print(contingency_table_2000)

#-------------------------------------------
# Create the main plot
ax2 = figure_t1.add_subplot(gs[1])
#sns.heatmap(contingency_table, annot=True, cmap='coolwarm', fmt="d")
sns.heatmap(contingency_table_2000, cmap='coolwarm', annot=annot_2000, fmt="")

# Set the title and labels
ax2.set_title('Contingency Table Heatmap (2000s, 2010s, 2020s)')
ax2.set_xlabel('Protagonist Gender')
ax2.set_ylabel('Author Gender')

# Rotate and align the labels
for tick in ax2.get_xticklabels():
    tick.set_rotation(45)
    tick.set_ha('right') # Horizontal alignment (right aligns better for rotated labels)
    tick.set_va('top') # Vertical alignment to ensure no overlap

#-------------------------------------------
# Step 2: Perform Chi-Square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table_2000)

# Results
print("Chi-Square Test Results:")
print(f"Chi2 Statistic: {chi2}")
print(f"P-Value: {p_value}")

# Interpret the result
if p_value < 0.05:
    print("There is a significant correlation between Author Gender and Protagonist Gender.")
else:
    print("There is no significant correlation between Author Gender and Protagonist Gender.")

# Save image-------------------------------------------
plt.savefig("./Figures/00 author and protagonist heatmap.png", bbox_inches = 'tight')
#plt.savefig("./Figures/00 author and protagonist heatmap.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/00 author and protagonist heatmap.svg", format = 'svg', transparent = True, bbox_inches = 'tight')
plt.close(figure_t1)

#---------------------------------------------------------------------------------------------------
# Figure 46 - Author and protagonist gender
print("  Making author and protagonist gender...")

#------------------------------------------
# Creates a figure object with size 12x6 inches
figure_t2 = plt.figure(46, figsize = (12, 6))
gs = figure_t2.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure_t2.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_genders = ['Male / Male',
                          'Male / Female',
                          'Male / Other',

                          'Male / Uncertain',
                          'Male / Not applicable',
                          
                          'Female / Male',
                          'Female / Female',
                          'Female / Other',

                          'Female / Uncertain',
                          'Female / Not applicable',
                          
                          'Other / All combined',
                          'Uncertain / All combined']

# Create a new DataFrame with all desired categories, filling with 0 if missing
all_categories_df_genders = pd.DataFrame(columns=category_order_genders).astype(float)
category_percent_genders = pd.concat([all_categories_df_genders, category_percent_genders]).convert_dtypes().fillna(0.0)

# Reorder the columns in the DataFrame according to the desired category order
category_percent_genders = category_percent_genders[category_order_genders]

# Define custom colors for each category
custom_colors_genders = ['#385AC2',
                         '#5580D0',
                         '#6CACEB',

                         '#FFD700',
                         '#D3D3D3',
                         
                         '#AE305D',
                         '#CF5D5F',
                         '#E3937B',

                         '#FFD700',
                         '#D3D3D3',
                         
                         '#8B3FCF',
                         '#FFD700']
# Bar plot-------------------------------------------
category_percent_genders.plot(kind = 'bar',
                              stacked = True,
                              ax = ax1,
                              color = custom_colors_genders,
                              width = 1.0,
                              alpha = 1.0,
                              label = "genders")

ax1.text(17.57, 
         106, 
         r"Author / Protagonist", 
         fontsize = 10.7, 
         zorder = 1,)
         #color = custom_dark_gray)

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("What is the gender of the author and the protagonist?", fontsize = 14, pad = 5, color = custom_dark_gray)
#ax1.yaxis.grid(True, linestyle = "dotted", linewidth = "1.0", zorder = 0, alpha = 1.0)

# Format the y-axis to show percentages
ax1.yaxis.set_major_formatter(PercentFormatter())

# Legend-------------------------------------------
# Get handles and labels
handles, labels = ax1.get_legend_handles_labels()

# Reverse the order
handles.reverse()
labels.reverse()

# Pass the reversed handles and labels to the legend
ax1.legend(handles, 
           labels, 
           bbox_to_anchor = (0.99, 0.00, 0.50, 0.95), 
           frameon = False, 
           labelspacing = 1.6,
           loc = 'center left')

# Axes-------------------------------------------
ax1.minorticks_on()
ax1.tick_params(which = "major", direction = "out", length = 3, labelsize = 10, color = custom_dark_gray)
ax1.tick_params(which = "minor", direction = "out", length = 0, color = custom_dark_gray)
ax1.tick_params(which = "both", bottom = True, top = False, left = True, right = False, color = custom_dark_gray)
ax1.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False, color = custom_dark_gray)
ax1.tick_params(axis = 'both', colors = custom_dark_gray)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
#ax1.spines['left'].set_visible(False)
ax1.spines['left'].set_color(custom_dark_gray)
ax1.spines['bottom'].set_color(custom_dark_gray)

# Dark lines-------------------------------------------
# Get number of bars (decades) from x-axis
n_decades = len(category_percent_genders.index)

# Define the boundaries (after which category the separator should be drawn)
boundaries = [4, 9, 11] # Indices between Male, Female, and Other/Uncertain
bar_width = 1.0

for b in boundaries:
    # Cumulative sum up to that boundary
    cumulative = category_percent_genders.iloc[:, :b+1].cumsum(axis=1)
    y_values = cumulative.iloc[:, -1].values # top per bar

    # Build step path
    xs = []
    ys = []
    for x, y in enumerate(y_values):
        # left edge
        xs.append(x - bar_width/2)
        ys.append(y)
        # right edge
        xs.append(x + bar_width/2)
        ys.append(y)

    # Now draw with step connections
    ax1.plot(xs, 
             ys, 
             color=custom_dark_gray, 
             linewidth = 1.2, 
             linestyle='solid')

    # Add vertical connectors between bars
    for i in range(len(y_values) - 1):
        ax1.plot(
            [i + bar_width/2, i + 1 - bar_width/2],
            [y_values[i], y_values[i+1]],
            color = custom_dark_gray,
            linewidth = 1.2,
            linestyle='solid'
        )

# Save image-------------------------------------------
plt.savefig("./Figures/00 author and protagonist gender.png", bbox_inches = 'tight')
#plt.savefig("./Figures/00 author and protagonist gender.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/00 author and protagonist gender.svg", format = 'svg', transparent = True, bbox_inches = 'tight')
plt.close(figure_t2)

#---------------------------------------------------------------------------------------------------
# Figure 47 - Variation in Answer
print("  Making Variation in Answers...")

#-----------------------------------------
# Creates a figure object with size 14x14 inches
figure_t3 = plt.figure(47, figsize = (14, 14))
gs = figure_t3.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure_t3.add_subplot(gs[0])

# Heatmap-------------------------------------------
# Set var_flag to the measure of variation in answers that you want:
# 1 Difference two by two of the answers
# 2 Percent Agreement (Mode Consistency)
# 3 Shannon Entropy (Diversity Index)
var_flag = 1

# Define tolerance for near-zero values
TOLERANCE = 1e-6

# Custom format function for decimal number show
def custom_format(val):
    # Treat values within the tolerance as zero
    if abs(val) < TOLERANCE:
        return 0
    elif abs(val) >= 100:
        return f"{val:.0f}" # No decimals for numbers >= 100
    elif abs(val) >= 10:
        return f"{val:.1f}" # One decimal for numbers >= 10 and < 100
    else:
        return f"{val:.2f}" # Two decimals for numbers < 10
        
#-----------------------------
# 1 Difference two by two of the answers
if var_flag == 1:
    sns.heatmap(df_mean_difference, 
                annot=df_mean_difference.map(custom_format), 
                cmap="coolwarm",
                fmt="", 
                cbar_kws={'label': 'Mean Difference Score'},
                annot_kws={"size": 8})

#-----------------------------
# 2 Percent Agreement (Mode Consistency)
elif var_flag == 2:
    sns.heatmap(df_percent_agreement, 
                annot=True, 
                cmap="coolwarm_r", # coolwarm_r: reversed coolwarm
                fmt=".3g",
                cbar_kws={'label': 'Percent agreement for the main answer'},
                annot_kws={"size": 8})

#-----------------------------
# 3 Shannon Entropy (Diversity Index)
elif var_flag == 3:
    sns.heatmap(df_entropy, 
                annot=df_entropy.map(custom_format), 
                cmap="coolwarm",
                fmt="", 
                cbar_kws={'label': 'Shannon index'},
                annot_kws={"size": 8})

#-----------------------------
# Add thicker lines to separate the last row and column

# Horizontal line above mean row
ax1.hlines(len(df_mean_difference) - 1, 
           *ax1.get_xlim(), 
           colors = custom_dark_gray, 
           linewidth=2.0)
# Vertical line before mean column
ax1.vlines(len(df_mean_difference.columns) - 1, 
           *ax1.get_ylim(), 
           colors = custom_dark_gray, 
           linewidth=2.0)

# Design-------------------------------------------
#ax1.set_xlabel("Questions", fontsize = 12, color = custom_dark_gray)
ax1.set_ylabel("", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Variability in Answers", fontsize = 14, pad = 5, color = custom_dark_gray)
#ax1.yaxis.grid(True, linestyle = "dotted", linewidth = "1.0", zorder = 0, alpha = 1.0)

# Axes-------------------------------------------
ax1.minorticks_on()
ax1.tick_params(which = "major", direction = "out", length = 3, labelsize = 10, color = custom_dark_gray)
ax1.tick_params(which = "minor", direction = "out", length = 0, color = custom_dark_gray)
ax1.tick_params(which = "both", bottom = True, top = False, left = True, right = False, color = custom_dark_gray)
ax1.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False, color = custom_dark_gray)
ax1.tick_params(axis = 'both', colors = custom_dark_gray)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
#ax1.spines['left'].set_visible(False)
ax1.spines['left'].set_color(custom_dark_gray)
ax1.spines['bottom'].set_color(custom_dark_gray)

# Move x-axis labels to the top
ax1.xaxis.tick_top()

# Rotate and align the labels
for tick in ax1.get_xticklabels():
    tick.set_rotation(45)
    tick.set_ha('left')  # Horizontal alignment (left aligns better for rotated labels)
    tick.set_va('bottom')  # Vertical alignment to ensure no overlap

# Save image-------------------------------------------
if var_flag == 1:
    plt.savefig("./Figures/00 variation difference.png", bbox_inches = 'tight')
elif var_flag == 2:
    plt.savefig("./Figures/00 variation mode.png", bbox_inches = 'tight')
elif var_flag == 3:
    plt.savefig("./Figures/00 variation entropy.png", bbox_inches = 'tight')

plt.close(figure_t3)

#---------------------------------------------------------------------------------------------------
# Figure 48 - protagonist classes
print("  Making protagonist classes (All decades)...")

#------------------------------------------
color_map = {
    "Total": '#385AC2',
    "Yes": '#385AC2', 
    "No": '#AE305D', 
    "Uncertain": '#FFD700',
    "Not applicable": '#D3D3D3', 
    "Human": '#385AC2', 
    "Non-human": '#AE305D', 
    "Male": '#385AC2', 
    "Female": '#AE305D', 
    "Other": '#8B3FCF',
    "Good": '#385AC2', 
    "Leaning good": '#5580D0', 
    "Ambivalent": '#8B3FCF', 
    "Leaning bad": '#CF5D5F', 
    "Bad": '#AE305D', 
    "Non-moral": '#008000'
}

path = [
    px.Constant("Total"),
    "author gender",
    "16 protagonist",
    "17 protagonist nature", 
    "18 protagonist gender", 
    "19 protagonist is"
    ]

# List of column names
column_titles = [
    "Author",
    "Protagonist",
    "Nature",
    "Gender",
    "Moral"
]

# Create a dictionary to map each column title to its x-position.
# You will need to adjust these values to perfectly center the labels
# on your specific chart.
x_positions = {
    "Author": 0.25,
    "Protagonist": 0.42,
    "Nature": 0.58,
    "Gender": 0.75,
    "Moral": 0.91
}

#-----------------------------
figure_classes_all = px.icicle(
    df_top_AI,
    path=path,
    color=path[-1],
    color_discrete_map=color_map
)

# Create a list of text colors based on the label's value
text_colors = []
for label in color_map.keys():
    if label == "Not applicable":
        text_colors.append("black") # Use default color
    else:
        text_colors.append('white') # Use white for all other labels

figure_classes_all.update_traces(
    textinfo="label+percent entry",
    root_color="#D3D3D3",
    marker_colors=[color_map.get(label, "#CCCCCC") for label in figure_classes_all.data[0].labels],
    textfont=dict(color=text_colors)
)

#-----------------------------
# Create a list of annotations for each column title
annotations = []
for title in column_titles:
    annotations.append(
        dict(
            text=title,
            x=x_positions.get(title), # Get the x-position from the dictionary
            y=1.0,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14, color="black"),
            xanchor="center",
            yanchor="bottom"
        )
    )

#-----------------------------
# Update layout with all changes
figure_classes_all.update_layout(
    #uniformtext=dict(minsize=5, mode='hide'),
    title_text="Authors and Types of Protagonist (all decades)",
    title_font_size=20,
    title_x=0.5,
    margin=dict(t=50, l=20, r=20, b=20),
    annotations=annotations
)

# Save image-------------------------------------------
figure_classes_all.write_image("./Figures/00 protagonist classes (all decades).png", scale=3)

#---------------------------------------------------------------------------------------------------
# Figure 49 - protagonist classes
print("  Making protagonist classes (2000s-2020s)...")

#mask = df_top_AI['decade'] >= 2000
#df_top_AI_2000 = df_top_AI[mask]

#-----------------------------
figure_classes_2000 = px.icicle(
    df_top_AI_2000,
    path=path,
    color=path[-1],
    color_discrete_map=color_map
)

# Create a list of text colors based on the label's value
text_colors = []
for label in color_map.keys():
    if label == "Not applicable":
        text_colors.append("black") # Use default color
    else:
        text_colors.append('white') # Use white for all other labels

figure_classes_2000.update_traces(
    textinfo="label+percent entry",
    root_color="#D3D3D3",
    marker_colors=[color_map.get(label, "#CCCCCC") for label in figure_classes_2000.data[0].labels],
    textfont=dict(color=text_colors)
)

#-----------------------------
# Create a list of annotations for each column title
annotations = []
for title in column_titles:
    annotations.append(
        dict(
            text=title,
            x=x_positions.get(title), # Get the x-position from the dictionary
            y=1.0,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14, color="black"),
            xanchor="center",
            yanchor="bottom"
        )
    )

#-----------------------------
# Update layout with all changes
figure_classes_2000.update_layout(
    #uniformtext=dict(minsize=5, mode='hide'),
    title_text="Authors and Types of Protagonist (2000s-2020s)",
    title_font_size=20,
    title_x=0.5,
    margin=dict(t=50, l=20, r=20, b=20),
    annotations=annotations
)

# Save image-------------------------------------------
figure_classes_2000.write_image("./Figures/00 protagonist classes (2000s-2020s).png", scale=3)

#---------------------------------------------------------------------------------------------------
print("All done.")

# Show figures-------------------------------------------------------------------------------------------
# plt.show() # Too many figures (> 20), so better not to show them.