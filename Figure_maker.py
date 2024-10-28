import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy.stats import chi2_contingency
import seaborn as sns
import numpy as np
from scipy.stats import mode

#----------------------------------------------------------------------------------
print("Reading and processing tha data...")

# Reading the data
df_filtered = pd.read_csv("./Data/sci-fi_books_FILTERED.csv", sep=";", encoding="utf-8-sig")
df_top = pd.read_csv("./Data/sci-fi_books_TOP.csv", sep=";", encoding="utf-8-sig")
df_top_AI = pd.read_csv("./Data/sci-fi_books_AI_ANSWERS.csv", sep=";", encoding="utf-8-sig")
df_top_AI_gender = pd.read_csv("./Data/sci-fi_books_AI_ANSWERS_GENDER.csv", sep=";", encoding="utf-8-sig")

#print(df_top_AI.info())
#print(df.head())

#----------------------------------------------------------------------------------
# Excluding books of before 1860 (allmost none)

mask_all = df_filtered['decade'] >= 1860
df_filtered = df_filtered[mask_all]

mask_top = df_top['decade'] >= 1860
df_top = df_top[mask_top]

mask_top_AI = df_top_AI['decade'] >= 1860
df_top_AI = df_top_AI[mask_top_AI]

#----------------------------------------------------------------------------------
# Including author gender to data

# Create a dictionary from df_top_AI_gender
author_gender_dict = df_top_AI_gender.set_index('author')['gender'].to_dict()

# Map the 'gender' based on 'author' column from df_top_AI
df_top_AI['gender'] = df_top_AI['author'].map(author_gender_dict).fillna('Unknown')

#----------------------------------------------------------------------------------
# For the boxplots

# Add a column to each DataFrame to label the dataset
df_filtered['dataset'] = 'Filtered sample'
df_top['dataset'] = 'Top sample'

# Concatenate dataframes
df_filtered_200 = pd.concat([df_filtered, df_top])

#----------------------------------------------------------------------------------
# General information of the FILTERED sample of books
#'title', 'author', 'year', 'decade', 'rate', 'ratings', 'genres', 'synopsis', 'review', 'url'
print("\nFILTERED books.")

book_per_decade = df_filtered['decade'].value_counts()
mean_per_decade = df_filtered.groupby('decade')[['rate', 'ratings']].mean()
sdv_per_decade = df_filtered.groupby('decade')[['rate', 'ratings']].std()

print(book_per_decade.sort_index(ascending = False))

# Creating a dictionary by passing Series objects as values
frame_1 = {'quantity': book_per_decade,
           'avr rate': mean_per_decade['rate'],
           'std rate': sdv_per_decade['rate'],
           'avr ratings': mean_per_decade['ratings'],
           'std ratings': sdv_per_decade['ratings']}
 
# Creating DataFrame by passing Dictionary
df_all = pd.DataFrame(frame_1)
df_all = (df_all
          .reset_index(drop = False)
          .sort_values(by = ['decade'], ascending = True)
          .reset_index(drop = True))

#print(df_all.info())
#print(df_all)

#----------------------------------------------------------------------------------
# General information of the 200 PER DECADE sample of books
#'title', 'author', 'year', 'decade', 'rate', 'ratings', 'genres', 'synopsis', 'review', 'url'
print("\n200 PER DECADE books.")

book_per_decade_200 = df_top['decade'].value_counts()
mean_per_decade_200 = df_top.groupby('decade')[['rate', 'ratings']].mean()
sdv_per_decade_200 = df_top.groupby('decade')[['rate', 'ratings']].std()

print(book_per_decade_200.sort_index(ascending = False))

# Creating a dictionary by passing Series objects as values
frame_2 = {'quantity': book_per_decade_200,
           'avr rate': mean_per_decade_200['rate'],
           'std rate': sdv_per_decade_200['rate'],
           'avr ratings': mean_per_decade_200['ratings'],
           'std ratings': sdv_per_decade_200['ratings']}
 
# Creating DataFrame by passing Dictionary
df_all_200 = pd.DataFrame(frame_2)
df_all_200 = (df_all_200
              .reset_index(drop = False)
              .sort_values(by = ['decade'], ascending = True)
              .reset_index(drop = True))

# Part in a series of books------------------------------------------
# Count the occurrences of each category per decade
category_counts_series = pd.crosstab(df_top['decade'], df_top['series'])
# Normalize the counts to get percentages
category_percent_series = category_counts_series.div(category_counts_series.sum(axis = 1), axis = 0) * 100

# Author gender------------------------------------------
# Count the occurrences of each category per decade
category_counts_gender = pd.crosstab(df_top_AI['decade'], df_top_AI['gender'])
# Normalize the counts to get percentages
category_percent_gender = category_counts_gender.div(category_counts_gender.sum(axis = 1), axis = 0) * 100

#----------------------------------------------------------------------------------
# Processing for the questions/answers

# 1 soft hard------------------------------------------
# Count the occurrences of each category per decade
category_counts_1 = pd.crosstab(df_top_AI['decade'], df_top_AI['1 soft hard'])
# Normalize the counts to get percentages
category_percent_1 = category_counts_1.div(category_counts_1.sum(axis = 1), axis = 0) * 100

# 2 light heavy------------------------------------------
# Count the occurrences of each category per decade
category_counts_2 = pd.crosstab(df_top_AI['decade'], df_top_AI['2 light heavy'])
# Normalize the counts to get percentages
category_percent_2 = category_counts_2.div(category_counts_2.sum(axis = 1), axis = 0) * 100

# 3 time------------------------------------------
# Count the occurrences of each category per decade
category_counts_3 = pd.crosstab(df_top_AI['decade'], df_top_AI['3 time'])
# Normalize the counts to get percentages
category_percent_3 = category_counts_3.div(category_counts_3.sum(axis = 1), axis = 0) * 100

# 4 tone------------------------------------------
# Count the occurrences of each category per decade
category_counts_4 = pd.crosstab(df_top_AI['decade'], df_top_AI['4 mood'])
# Normalize the counts to get percentages
category_percent_4 = category_counts_4.div(category_counts_4.sum(axis = 1), axis = 0) * 100

# 5 social political------------------------------------------
# Count the occurrences of each category per decade
category_counts_5 = pd.crosstab(df_top_AI['decade'], df_top_AI['5 social political'])
# Normalize the counts to get percentages
category_percent_5 = category_counts_5.div(category_counts_5.sum(axis = 1), axis = 0) * 100

# 6 on Earth------------------------------------------
# Count the occurrences of each category per decade
category_counts_6 = pd.crosstab(df_top_AI['decade'], df_top_AI['6 on Earth'])
# Normalize the counts to get percentages
category_percent_6 = category_counts_6.div(category_counts_6.sum(axis = 1), axis = 0) * 100

# 7 post apocalyptic------------------------------------------
# Count the occurrences of each category per decade
category_counts_7 = pd.crosstab(df_top_AI['decade'], df_top_AI['7 post apocalyptic'])
# Normalize the counts to get percentages
category_percent_7 = category_counts_7.div(category_counts_7.sum(axis = 1), axis = 0) * 100

# 8 aliens------------------------------------------
# Count the occurrences of each category per decade
category_counts_8 = pd.crosstab(df_top_AI['decade'], df_top_AI['8 aliens'])
# Normalize the counts to get percentages
category_percent_8 = category_counts_8.div(category_counts_8.sum(axis = 1), axis = 0) * 100

# 9 aliens are------------------------------------------
# Count the occurrences of each category per decade
category_counts_9 = pd.crosstab(df_top_AI['decade'], df_top_AI['9 aliens are'])
# Normalize the counts to get percentages
category_percent_9 = category_counts_9.div(category_counts_9.sum(axis = 1), axis = 0) * 100

# 10 robots and AI------------------------------------------
# Count the occurrences of each category per decade
category_counts_10 = pd.crosstab(df_top_AI['decade'], df_top_AI['10 robots and AI'])
# Normalize the counts to get percentages
category_percent_10 = category_counts_10.div(category_counts_10.sum(axis = 1), axis = 0) * 100

# 11 robots and AI are------------------------------------------
# Count the occurrences of each category per decade
category_counts_11 = pd.crosstab(df_top_AI['decade'], df_top_AI['11 robots and AI are'])
# Normalize the counts to get percentages
category_percent_11 = category_counts_11.div(category_counts_11.sum(axis = 1), axis = 0) * 100

# 12 protagonist------------------------------------------
# Count the occurrences of each category per decade
category_counts_12 = pd.crosstab(df_top_AI['decade'], df_top_AI['12 protagonist'])
# Normalize the counts to get percentages
category_percent_12 = category_counts_12.div(category_counts_12.sum(axis = 1), axis = 0) * 100

# 13 protagonist is------------------------------------------
# Count the occurrences of each category per decade
category_counts_13 = pd.crosstab(df_top_AI['decade'], df_top_AI['13 protagonist is'])
# Normalize the counts to get percentages
category_percent_13 = category_counts_13.div(category_counts_13.sum(axis = 1), axis = 0) * 100

# 15 virtual------------------------------------------
# Count the occurrences of each category per decade
category_counts_14 = pd.crosstab(df_top_AI['decade'], df_top_AI['14 virtual'])
# Normalize the counts to get percentages
category_percent_14 = category_counts_14.div(category_counts_14.sum(axis = 1), axis = 0) * 100

# 14 tech and science------------------------------------------
# Count the occurrences of each category per decade
category_counts_15 = pd.crosstab(df_top_AI['decade'], df_top_AI['15 tech and science'])
# Normalize the counts to get percentages
category_percent_15 = category_counts_15.div(category_counts_15.sum(axis = 1), axis = 0) * 100

# 16 social issues------------------------------------------
# Count the occurrences of each category per decade
category_counts_16 = pd.crosstab(df_top_AI['decade'], df_top_AI['16 social issues'])
# Normalize the counts to get percentages
category_percent_16 = category_counts_16.div(category_counts_16.sum(axis = 1), axis = 0) * 100

# 17 enviromental------------------------------------------
# Count the occurrences of each category per decade
category_counts_17 = pd.crosstab(df_top_AI['decade'], df_top_AI['17 enviromental'])
# Normalize the counts to get percentages
category_percent_17 = category_counts_17.div(category_counts_17.sum(axis = 1), axis = 0) * 100

#----------------------------------------------------------------------------------
# Processing for the complex figures
#----------------------------------------------------------------------------------
# Author and protagonist gender

df_top_AI_new = df_top_AI.copy()
df_top_AI_new["genders"] = df_top_AI_new['gender'] + " / " + df_top_AI_new['13 protagonist is']

# Count the occurrences of each category per decade
category_counts_genders = pd.crosstab(df_top_AI_new['decade'], df_top_AI_new["genders"])

# Sum columns to create a new 'Other' category (combining the Others)
category_counts_genders['Other / All combined'] = (category_counts_genders['Other / Male'] + 
                                                   category_counts_genders['Other / Female'] + 
                                                   category_counts_genders['Other / Non-human'] + 
                                                   category_counts_genders['Other / Not applicable'])

# Drop the original Others columns
category_counts_genders = category_counts_genders.drop(columns=['Other / Male', 
                                                                'Other / Female',
                                                                'Other / Non-human',
                                                                'Other / Not applicable'])

# Normalize the counts to get percentages
category_percent_genders = category_counts_genders.div(category_counts_genders.sum(axis = 1), axis = 0) * 100

print("\ngenders",df_top_AI_new["genders"].unique())
#print("\n")

#----------------------------------------------------------------------------------
# Variance in Answers

# Reading test answers
df1 = pd.read_csv("./Data/Variability in Answers/sci-fi_books_AI_ANSWERS_TEST_1.csv", sep=";", encoding="utf-8-sig")
df2 = pd.read_csv("./Data/Variability in Answers/sci-fi_books_AI_ANSWERS_TEST_2.csv", sep=";", encoding="utf-8-sig")
df3 = pd.read_csv("./Data/Variability in Answers/sci-fi_books_AI_ANSWERS_TEST_3.csv", sep=";", encoding="utf-8-sig")
df4 = pd.read_csv("./Data/Variability in Answers/sci-fi_books_AI_ANSWERS_TEST_4.csv", sep=";", encoding="utf-8-sig")
df5 = pd.read_csv("./Data/Variability in Answers/sci-fi_books_AI_ANSWERS_TEST_5.csv", sep=";", encoding="utf-8-sig")
df6 = pd.read_csv("./Data/Variability in Answers/sci-fi_books_AI_ANSWERS_TEST_6.csv", sep=";", encoding="utf-8-sig")
df7 = pd.read_csv("./Data/Variability in Answers/sci-fi_books_AI_ANSWERS_TEST_7.csv", sep=";", encoding="utf-8-sig")
df8 = pd.read_csv("./Data/Variability in Answers/sci-fi_books_AI_ANSWERS_TEST_8.csv", sep=";", encoding="utf-8-sig")
df9 = pd.read_csv("./Data/Variability in Answers/sci-fi_books_AI_ANSWERS_TEST_9.csv", sep=";", encoding="utf-8-sig")
df10 = pd.read_csv("./Data/Variability in Answers/sci-fi_books_AI_ANSWERS_TEST_10.csv", sep=";", encoding="utf-8-sig")

# Setting novel id as index
df1['year'] = df1['year'].astype('string')
df2['year'] = df2['year'].astype('string')
df3['year'] = df3['year'].astype('string')
df4['year'] = df4['year'].astype('string')
df5['year'] = df5['year'].astype('string')
df6['year'] = df6['year'].astype('string')
df7['year'] = df7['year'].astype('string')
df8['year'] = df8['year'].astype('string')
df9['year'] = df9['year'].astype('string')
df10['year'] = df10['year'].astype('string')

df1['id'] = df1['title'] + " (" + df1['year'] + ") " + df1['author']
df2['id'] = df2['title'] + " (" + df2['year'] + ") " + df2['author']
df3['id'] = df3['title'] + " (" + df3['year'] + ") " + df3['author']
df4['id'] = df4['title'] + " (" + df4['year'] + ") " + df4['author']
df5['id'] = df5['title'] + " (" + df5['year'] + ") " + df5['author']
df6['id'] = df6['title'] + " (" + df6['year'] + ") " + df6['author']
df7['id'] = df7['title'] + " (" + df7['year'] + ") " + df7['author']
df8['id'] = df8['title'] + " (" + df8['year'] + ") " + df8['author']
df9['id'] = df9['title'] + " (" + df9['year'] + ") " + df9['author']
df10['id'] = df10['title'] + " (" + df10['year'] + ") " + df10['author']

df1 = df1.set_index('id')
df2 = df2.set_index('id')
df3 = df3.set_index('id')
df4 = df4.set_index('id')
df5 = df5.set_index('id')
df6 = df6.set_index('id')
df7 = df7.set_index('id')
df8 = df8.set_index('id')
df9 = df9.set_index('id')
df10 = df10.set_index('id')

#-------------------------------------------
# Selecting columns
column_order = ['1 soft hard',
                '2 light heavy',
                '3 time',
                '4 mood',
                '5 setting',
                '6 on Earth',
                '7 post apocalyptic',
                '8 aliens',
                '9 aliens are',
                '10 robots and AI',
                '11 robots and AI are',
                '12 protagonist',
                '13 protagonist is',
                '14 virtual',
                '15 tech and science',
                '16 social issues',
                '17 enviromental',]

df1 = df1.reindex(columns=column_order)
df2 = df2.reindex(columns=column_order)
df3 = df3.reindex(columns=column_order)
df4 = df4.reindex(columns=column_order)
df5 = df5.reindex(columns=column_order)
df6 = df6.reindex(columns=column_order)
df7 = df7.reindex(columns=column_order)
df8 = df8.reindex(columns=column_order)
df9 = df9.reindex(columns=column_order)
df10 = df10.reindex(columns=column_order)

#-------------------------------------------
# List of dataframes
dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10]

#-----------------------------
# Stack DataFrames to create a 3D array (rows x columns x runs)
data_array = np.stack([df.to_numpy() for df in dfs], axis=-1) # Shape: (rows, columns, runs)

#-------------------------
# Difference two by two of the answers

# Initialize a dataframe to store the sum of differences
comparison_sum = pd.DataFrame(0, index=df1.index, columns=df1.columns)

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

print("num_comparisons =", num_comparisons)

# Calculate the mean of all cells
overall_mean = df_mean_difference.values.mean()
print("Mean of all cells:", overall_mean)

# Add column and row of means
df_mean_difference['Mean'] = df_mean_difference.mean(axis=1)
df_mean_difference.loc['Mean'] = df_mean_difference.mean(axis=0)

# Display the resulting mean difference dataframe
print(df_mean_difference)

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
df_modes = pd.DataFrame(modes, columns=df1.columns, index=df1.index)
# Convert to DataFrames
df_percent_agreement = pd.DataFrame(percent_agreement, index=df1.index, columns=df1.columns)

# Add column and row of means
df_percent_agreement['Mean'] = df_percent_agreement.mean(axis=1)
df_percent_agreement.loc['Mean'] = df_percent_agreement.mean(axis=0)

#-----------------------------
# Shannon Entropy (Diversity Index)
def shannon_entropy(values):
    _, counts = np.unique(values, return_counts=True) # values, counts
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities + 1e-9)) # Avoid log(0)

# Apply the entropy function along the last axis (runs)
entropy_values = np.apply_along_axis(shannon_entropy, 2, data_array)

# Convert to DataFrames
df_entropy = pd.DataFrame(entropy_values, index=df1.index, columns=df1.columns)

# Add column and row of means
df_entropy['Mean'] = df_entropy.mean(axis=1)
df_entropy.loc['Mean'] = df_entropy.mean(axis=0)

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Making figures
print("Making the figures...")
#----------------------------------------------------------------------------------
# Custom dark gray color
custom_dark_gray = (0.2, 0.2, 0.2)
#----------------------------------------------------------------------------------
# Figure 1, Quantities
print("  Making book counts...")

# Creates a figure object with size 12x6 inches
figure_c1 = plt.figure(1, figsize = (12, 6))
gs = figure_c1.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure_c1.add_subplot(gs[0])

# Bar plot all sample
ax1.bar(x = df_all['decade'],
        height = df_all['quantity'], 
        width = 9, 
        #bottom = None, 
        align = 'center', 
        #data = None,
        color = "#385AC2",
        #hatch = '\\',
        alpha = 1.0,
        edgecolor = custom_dark_gray,
        linewidth = 0.0,
        # #tick_label = 0,
        label = "Filtered sample")

# Bar plot top 200
ax1.bar(x = df_all_200['decade'],
        height = df_all_200['quantity'], 
        width = 9, 
        #bottom = None, 
        align = 'center', 
        #data = None,
        color = "#AE305D",
        #hatch = '/',
        alpha = 1.0,
        edgecolor = custom_dark_gray,
        linewidth = 0.0,
        # #tick_label = 0,
        label = "Top sample")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
ax1.set_ylabel("Quantity", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Sci-fi Books per Decade", fontsize = 14, pad = 5, color = custom_dark_gray)
#ax1.grid(True, linestyle = "dotted", linewidth = "1.0", zorder = 0, alpha = 1.0)

# Legend-------------------------------------------
ax1.legend(frameon = False, 
           #labelspacing = 10.0,
           loc = 'upper left')

# Axes-------------------------------------------
ax1.minorticks_on()
ax1.tick_params(which = "major", direction = "out", length = 3, labelsize = 10, color = custom_dark_gray)
ax1.tick_params(which = "minor", direction = "out", length = 0, color = custom_dark_gray)
ax1.tick_params(which = "both", bottom = True, top = False, left = True, right = False, color = custom_dark_gray)
ax1.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False, color = custom_dark_gray)
ax1.tick_params(axis = 'both', colors = custom_dark_gray)

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
#ax1.spines['right'].set_color(custom_dark_gray)
#ax1.spines['top'].set_color(custom_dark_gray)
ax1.spines['left'].set_color(custom_dark_gray)
ax1.spines['bottom'].set_color(custom_dark_gray)

# Saving image-------------------------------------------
plt.savefig("./Figures/00 Quantities.png", bbox_inches = 'tight')
#plt.savefig("./Figures/00 Quantities.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/00 Quantities.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
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

#----------------------------------------------------------------------------------
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

# Saving image-------------------------------------------
plt.savefig("./Figures/00 Rates and Ratings.png", bbox_inches = 'tight')
#plt.savefig("./Figures/00 Rates and Ratings.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/00 Rates and Ratings.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Figure 3, Series
print("  Making series...")

# Creates a figure object with size 12x6 inches
figure_c3 = plt.figure(3, figsize = (12, 6))
gs = figure_c3.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure_c3.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_series = ['yes',
                         'no']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_series = category_percent_series[category_order_series]

# Define custom colors for each category
custom_colors_series = ['#AE305D', 
                        '#385AC2']

# Bar plot-------------------------------------------
category_percent_series.plot(kind = 'bar',
                             stacked = True,
                             ax = ax1,
                             color = custom_colors_series,
                             width = 1.0,
                             alpha = 1.0,
                             label = "series")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Is the book part of a series?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 12.0,
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

# Saving image-------------------------------------------
plt.savefig("./Figures/00 series.png", bbox_inches = 'tight')
#plt.savefig("./Figures/00 series.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/00 series.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Figure 4, author gender
print("  Making author gender...")

# Creates a figure object with size 12x6 inches
figure_c4 = plt.figure(4, figsize = (12, 6))
gs = figure_c4.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure_c4.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_gender = ['Male',
                         'Female',
                         'Other']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_gender = category_percent_gender[category_order_gender]

# Define custom colors for each category
custom_colors_gender = ['#385AC2',
                        '#AE305D',
                        '#8B3FCF']

# Bar plot-------------------------------------------
category_percent_gender.plot(kind = 'bar',
                             stacked = True,
                             ax = ax1,
                             color = custom_colors_gender,
                             width = 1.0,
                             alpha = 1.0,
                             label = "gender")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("What is the gender of the author?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 10.0,
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

# Saving image-------------------------------------------
plt.savefig("./Figures/00 author gender.png", bbox_inches = 'tight')
#plt.savefig("./Figures/00 author gender.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/00 author gender.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# figures for the questions/answers
#----------------------------------------------------------------------------------
# Figure 5 - 1 soft hard
print("  Making 1 soft hard...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure1 = plt.figure(5, figsize = (12, 6))
gs = figure1.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure1.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_1 = ['Very hard',
                    'Hard',
                    'Mixed', 
                    'Soft',
                    'Very soft']
# Reorder the columns in the DataFrame according to the desired category order
category_percent_1 = category_percent_1[category_order_1]

# Define custom colors for each category
custom_colors_1 = ['#AE305D',
                   '#CF5D5F',
                   '#8B3FCF',
                   '#5580D0',
                   '#385AC2']

# Bar plot-------------------------------------------
category_percent_1.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_1,
                        width = 1.0,
                        alpha = 1.0,
                        label = "1 soft hard")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Is the book considered more soft or hard sci-fi?", fontsize = 14, color = custom_dark_gray)
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
           labelspacing = 5.0,
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
#ax1.spines['top'].set_color(custom_dark_gray)
#ax1.spines['left'].set_visible(False)
ax1.spines['left'].set_color(custom_dark_gray)
ax1.spines['bottom'].set_color(custom_dark_gray)

# Saving image-------------------------------------------
plt.savefig("./Figures/01 soft hard.png", bbox_inches = 'tight')
#plt.savefig("./Figures/01 soft hard.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/01 soft hard.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 6 - 1 light heavy
print("  Making 2 light heavy...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure2 = plt.figure(6, figsize = (12, 6))
gs = figure2.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure2.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_2 = ['Very heavy',
                    'Heavy',
                    'Balanced', 
                    'Light',
                    'Very light']
# Reorder the columns in the DataFrame according to the desired category order
category_percent_2 = category_percent_2[category_order_2]

# Define custom colors for each category
custom_colors_2 = ['#AE305D',
                   '#CF5D5F',
                   '#8B3FCF',
                   '#5580D0',
                   '#385AC2']

# Bar plot-------------------------------------------
category_percent_2.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_2,
                        width = 1.0,
                        alpha = 1.0,
                        label = "2 light heavy")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Is the book considered more of a light or heavy reading experience?", fontsize = 14, color = custom_dark_gray)
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
           labelspacing = 5.0,
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
#ax1.spines['top'].set_color(custom_dark_gray)
#ax1.spines['left'].set_visible(False)
ax1.spines['left'].set_color(custom_dark_gray)
ax1.spines['bottom'].set_color(custom_dark_gray)

# Saving image-------------------------------------------
plt.savefig("./Figures/02 light heavy.png", bbox_inches = 'tight')
#plt.savefig("./Figures/02 light heavy.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/02 light heavy.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 7 - 3 time
print("  Making 3 time...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure3 = plt.figure(7, figsize = (12, 6))
gs = figure3.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure3.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_3 = ['Distant past',
                    'Far past',
                    'Near past',
                    'Present',
                    'Near future',
                    'Far future',
                    'Distant future',

                    'Multiple timelines',
                    'Uncertain']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_3 = category_percent_3[category_order_3]

# Define custom colors for each category
custom_colors_3 = ['#AE305D', # Distant past
                   '#CF5D5F',
                   '#E3937B',
                   '#8B3FCF',
                   '#6CACEB',
                   '#5580D0',
                   '#385AC2', # Distant future

                   'green',     # Multiple timelines
                   'lightgrey'] # Uncertain

# Bar plot-------------------------------------------
category_percent_3.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_3,
                        width = 1.0,
                        alpha = 1.0,
                        label = "3 time")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("When does most of the story take place in relation to the year the book was published?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 2.8,
           loc='center left')

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

# Saving image-------------------------------------------
plt.savefig("./Figures/03 time.png", bbox_inches = 'tight')
#plt.savefig("./Figures/03 time.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/03 time.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 8 - 4 mood
print("  Making 4 mood...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure4 = plt.figure(8, figsize = (12, 6))
gs = figure4.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure4.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_4 = ['Very pessimistic',
                    'Pessimistic',
                    'Balanced',
                    'Optimistic',
                    'Very optimistic']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_4 = category_percent_4[category_order_4]

# Define custom colors for each category
custom_colors_4 = ['#AE305D',
                   '#CF5D5F',
                   '#8B3FCF',
                   '#5580D0',
                   '#385AC2']

# Bar plot-------------------------------------------
category_percent_4.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_4,
                        width = 1.0,
                        alpha = 1.0,
                        label = "4 mood")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("What is the mood of the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 5.0,
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

# Saving image-------------------------------------------
plt.savefig("./Figures/04 mood.png", bbox_inches = 'tight')
#plt.savefig("./Figures/04 mood.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/04 mood.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 9 - 5 setting
print("  Making 5 social political...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure5 = plt.figure(9, figsize = (12, 6))
gs = figure5.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure5.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_5 = ['Dystopic',
                    'Leaning dystopic',
                    'Balanced',
                    'Leaning utopic',
                    'Utopic',

                    'Uncertain']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_5 = category_percent_5[category_order_5]

# Define custom colors for each category
custom_colors_5 = ['#AE305D',
                   '#CF5D5F',
                   '#8B3FCF',
                   '#5580D0',
                   '#385AC2',

                   'lightgrey']

# Bar plot-------------------------------------------
category_percent_5.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_5,
                        width = 1.0,
                        alpha = 1.0,
                        label = "5 social political")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("What is the social and political scenario depicted in the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 4.0,
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

# Saving image-------------------------------------------
plt.savefig("./Figures/05 social political.png", bbox_inches = 'tight')
#plt.savefig("./Figures/05 social political.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/05 social political.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 10 - 6 on Earth
print("  Making 6 on Earth...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure6 = plt.figure(10, figsize = (12, 6))
gs = figure6.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure6.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_6 = ['Yes',
                    'No']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_6 = category_percent_6[category_order_6]

# Define custom colors for each category
custom_colors_6 = ['#385AC2',
                   '#AE305D']

# Bar plot-------------------------------------------
category_percent_6.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_6,
                        width = 1.0,
                        alpha = 1.0,
                        label = "6 on Earth")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Is most of the story set on Earth?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 12.0,
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

# Saving image-------------------------------------------
plt.savefig("./Figures/06 on Earth.png", bbox_inches = 'tight')
#plt.savefig("./Figures/06 on Earth.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/06 on Earth.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 11 - 7 post apocalyptic
print("  Making 7 post apocalyptic...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure7 = plt.figure(11, figsize = (12, 6))
gs = figure7.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure7.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_7 = ['Yes',
                    'Somewhat',
                    'No']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_7 = category_percent_7[category_order_7]

# Define custom colors for each category
custom_colors_7 = ['#AE305D',
                   '#8B3FCF',
                   '#385AC2']

# Bar plot-------------------------------------------
category_percent_7.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_7,
                        width = 1.0,
                        alpha = 1.0,
                        label = "7 post apocalyptic")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Is the story set in a post-apocalyptic world (after a civilization-collapsing event)?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 8.0,
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

# Saving image-------------------------------------------
plt.savefig("./Figures/07 post apocalyptic.png", bbox_inches = 'tight')
#plt.savefig("./Figures/07 post apocalyptic.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/07 post apocalyptic.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 12 - 8 aliens
print("  Making 8 aliens...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure8 = plt.figure(12, figsize = (12, 6))
gs = figure8.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure8.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_8 = ['Yes',
                    'No']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_8 = category_percent_8[category_order_8]

# Define custom colors for each category
custom_colors_8 = ['#AE305D',
                   '#385AC2']

# Bar plot-------------------------------------------
category_percent_8.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_8,
                        width = 1.0,
                        alpha = 1.0,
                        label = "8 aliens")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Are there any depictions or mentions of non-terrestrial life forms \n(e.g., aliens, extraterrestrial organisms, creatures not originating from Earth, even if non-sentient) \nor alien technology in the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 12.0,
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

# Saving image-------------------------------------------
plt.savefig("./Figures/08 aliens.png", bbox_inches = 'tight')
#plt.savefig("./Figures/08 aliens.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/08 aliens.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 13 - 9 aliens are
print("  Making 9 aliens are...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure9 = plt.figure(13, figsize = (12, 6))
gs = figure9.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure9.add_subplot(gs[0])

# Define the desired order of the categories
category_order_9 = ['Bad', 
                    'Leaning bad',
                    'Ambivalent', 
                    'Leaning good',
                    'Good',

                    'Uncertain',
                    'Not applicable']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_9 = category_percent_9[category_order_9]

# Define custom colors for each category
custom_colors_9 = ['#AE305D',
                   '#CF5D5F',
                   '#8B3FCF',
                   '#5580D0',
                   '#385AC2',

                   'green',
                   'lightgrey']

# Bar plot-------------------------------------------
category_percent_9.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_9,
                        width = 1.0,
                        alpha = 1.0,
                        label = "9 aliens are")

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
           labelspacing = 4.0,
           loc = 'center left')

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("How are the non-terrestrial life forms generally depicted in the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
#ax1.yaxis.grid(True, linestyle = "dotted", linewidth = "1.0", zorder = 0, alpha = 1.0)

# Format the y-axis to show percentages
ax1.yaxis.set_major_formatter(PercentFormatter())

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

# Saving image-------------------------------------------
plt.savefig("./Figures/09 aliens are.png", bbox_inches = 'tight')
#plt.savefig("./Figures/09 aliens are.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/09 aliens are.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 14 - 10 robots and AI
print("  Making 10 robots and AI...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure10 = plt.figure(14, figsize = (12, 6))
gs = figure10.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure10.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_10 = ['Yes',
                     'No']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_10 = category_percent_10[category_order_10]

# Define custom colors for each category
custom_colors_10 = ['#AE305D',
                    '#385AC2']

# Bar plot-------------------------------------------
category_percent_10.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_10,
                         width = 1.0,
                         alpha = 1.0,
                         label = "10 robots and AI")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Are there any depictions of robots or artificial intelligences in the story? \n(just automatic systems, advanced technology, or programs do not count)", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 12.0,
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

# Saving image-------------------------------------------
plt.savefig("./Figures/10 robots and AI.png", bbox_inches = 'tight')
#plt.savefig("./Figures/10 robots and AI.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/10 robots and AI.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 15 - 11 robots and AI are
print("  Making 11 robots and AI are...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure11 = plt.figure(15, figsize = (12, 6))
gs = figure11.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure11.add_subplot(gs[0])
# Figure 12 - 11 robots and AI are

#-------------------------------------------
# Define the desired order of the categories
category_order_11 = ['Bad', 
                     'Leaning bad', 
                     'Ambivalent', 
                     'Leaning good',
                     'Good',

                     'Uncertain',
                     'Not applicable']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_11 = category_percent_11[category_order_11]

# Define custom colors for each category
custom_colors_11 = ['#AE305D',
                    '#CF5D5F',
                    '#8B3FCF',
                    '#5580D0',
                    '#385AC2',

                    'green',
                    'lightgrey']

# Bar plot-------------------------------------------
category_percent_11.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_11,
                         width = 1.0,
                         alpha = 1.0,
                         label = "11 robots and AI are")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("How are the robots or artificial intelligences generally depicted in the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 4.0,
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

# Saving image-------------------------------------------
plt.savefig("./Figures/11 robots and AI are.png", bbox_inches = 'tight')
#plt.savefig("./Figures/11 robots and AI are.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/11 robots and AI are.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 16 - 12 protagonist
print("  Making 12 protagonist...")
#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure12 = plt.figure(16, figsize = (12, 6))
gs = figure12.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure12.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_12 = ['Yes', 
                     'No']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_12 = category_percent_12[category_order_12]

# Define custom colors for each category
custom_colors_12 = ['#AE305D',
                    '#385AC2']

# Bar plot-------------------------------------------
category_percent_12.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_12,
                         width = 1.0,
                         alpha = 1.0,
                         label = "12 protagonist")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Is there a single protagonist or main character?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 12.0,
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

# Saving image-------------------------------------------
plt.savefig("./Figures/12 protagonist.png", bbox_inches = 'tight')
#plt.savefig("./Figures/12 protagonist.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/12 protagonist.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 17 - 13 protagonist is
print("  Making 13 protagonist is...")
#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure13 = plt.figure(17, figsize = (12, 6))
gs = figure13.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure13.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_13 = ['Male', 
                     'Female',

                     'Other',
                     'Non-human',

                     'Not applicable']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_13 = category_percent_13[category_order_13]

# Define custom colors for each category
custom_colors_13 = ['#385AC2',
                    '#AE305D',

                    '#8B3FCF',
                    'green',

                    'lightgrey']

# Bar plot-------------------------------------------
category_percent_13.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_13,
                         width = 1.0,
                         alpha = 1.0,
                         label = "13 protagonist is")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("What is the gender of the single protagonist or main character?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 5.0,
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

# Saving image-------------------------------------------
plt.savefig("./Figures/13 protagonist is.png", bbox_inches = 'tight')
#plt.savefig("./Figures/13 protagonist is.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/13 protagonist is.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 18 - 14 virtual
print("  Making 14 virtual...")
#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure14 = plt.figure(18, figsize = (12, 6))
gs = figure14.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure14.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_14 = ['Yes', 
                     'Somewhat',
                     'No']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_14 = category_percent_14[category_order_14]

# Define custom colors for each category
custom_colors_14 = ['#AE305D',
                    '#8B3FCF',
                    '#385AC2']

# Bar plot-------------------------------------------
category_percent_14.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_14,
                         width = 1.0,
                         alpha = 1.0,
                         label = "14 virtual")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Are there any depictions of virtual reality or immersive digital environments \n(e.g., simulations, augmented reality) in the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 8.0,
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

# Saving image-------------------------------------------
plt.savefig("./Figures/14 virtual.png", bbox_inches = 'tight')
#plt.savefig("./Figures/14 virtual.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/14 virtual.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 19 - 15 tech and science
print("  Making 15 tech and science...")
#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure15 = plt.figure(19, figsize = (12, 6))
gs = figure15.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure15.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_15 = ['Bad', 
                     'Leaning bad', 
                     'Ambivalent', 
                     'Leaning good',
                     'Good',

                     'Uncertain']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_15 = category_percent_15[category_order_15]

# Define custom colors for each category
custom_colors_15 = ['#AE305D',
                    '#CF5D5F',
                    '#8B3FCF',
                    '#5580D0',
                    '#385AC2',

                    'lightgrey']

# Bar plot-------------------------------------------
category_percent_15.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_15,
                         width = 1.0,
                         alpha = 1.0,
                         label = "15 tech and science")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("How are technology and science depicted in the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 4.0,
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

# Saving image-------------------------------------------
plt.savefig("./Figures/15 tech and science.png", bbox_inches = 'tight')
#plt.savefig("./Figures/15 tech and science.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/15 tech and science.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 20 - 16 social issues
print("  Making 16 social issues...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure16 = plt.figure(20, figsize = (12, 6))
gs = figure16.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure16.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_16 = ['Core', 
                     'Major',
                     'Minor',

                     'Absent']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_16 = category_percent_16[category_order_16]

# Define custom colors for each category
custom_colors_16 = ['#385AC2',
                    '#5580D0',
                    '#6CACEB',

                    '#AE305D']

# Bar plot-------------------------------------------
category_percent_16.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_16,
                         width = 1.0,
                         alpha = 1.0,
                         label = "16 social issues")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("How central is the critique or reflection of specific social issues \n(e.g., inequality, war, discrimination, political oppression) to the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 6.0,
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

# Saving image-------------------------------------------
plt.savefig("./Figures/16 social issues.png", bbox_inches = 'tight')
#plt.savefig("./Figures/16 social issues.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/16 social issues.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 21 - 17 enviromental
print("  Making 17 enviromental...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure17 = plt.figure(21, figsize = (12, 6))
gs = figure17.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure17.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_17 = ['Core', 
                     'Major',
                     'Minor',

                     'Absent']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_17 = category_percent_17[category_order_17]

# Define custom colors for each category
custom_colors_17 = ['#385AC2',
                    '#5580D0',
                    '#6CACEB',

                    '#AE305D']

# Bar plot-------------------------------------------
category_percent_17.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_17,
                         width = 1.0,
                         alpha = 1.0,
                         label = "17 enviromental")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("How central is an ecological or environmental message to the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
           labelspacing = 6.0,
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

# Saving image-------------------------------------------
plt.savefig("./Figures/17 enviromental.png", bbox_inches = 'tight')
#plt.savefig("./Figures/17 enviromental.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/17 enviromental.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# For the Tests and more
#----------------------------------------------------------------------------------
# Figure 22 - Author and protagonist heatmap
print("  Making author and protagonist heatmap...")

#-------------------------------------------
# Creates a figure object with size 16x5 inches
figure_t1 = plt.figure(22, figsize = (16, 5))
gs = figure_t1.add_gridspec(ncols = 2, nrows = 1)

#-------------------------------------------
# Step 1: Create a contingency table (cross-tab)
contingency_table = pd.crosstab(df_top_AI['gender'], df_top_AI['13 protagonist is'])

contingency_table = contingency_table.div(contingency_table.sum(axis = 1), axis = 0) * 100

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

#-------------------------------------------
# Step 2: Perform Chi-Square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Results
print("Chi-Square Test Results:")
print(f"Chi2 Statistic: {chi2}")
print(f"P-Value: {p_value}")

# Interpret the result
if p_value < 0.05:
    print("There is a significant correlation between Author Gender and Protagonist Gender.")
else:
    print("There is no significant correlation between Author Gender and Protagonist Gender.")

#-------------------------------------------
mask = df_top_AI['decade'] >= 2000
df_top_AI_2000 = df_top_AI[mask]

# Step 1: Create a contingency table (cross-tab)
contingency_table_2000 = pd.crosstab(df_top_AI_2000['gender'], df_top_AI_2000['13 protagonist is'])

contingency_table_2000 = contingency_table_2000.div(contingency_table_2000.sum(axis = 1), axis = 0) * 100

# Create a formatted string version of the percentages with % symbol for annotation
annot_2000 = contingency_table_2000.map(lambda x: f'{x:.2f}%')

print("\n")
print("Contingency Table 2000:")
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

# Saving image-------------------------------------------
plt.savefig("./Figures/00 author and protagonist heatmap.png", bbox_inches = 'tight')
#plt.savefig("./Figures/00 author and protagonist heatmap.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/00 author and protagonist heatmap.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 23 - Author and protagonist gender
print("  Making author and protagonist heatmap...")

#------------------------------------------
# Creates a figure object with size 12x6 inches
figure_t2 = plt.figure(23, figsize = (12, 6))
gs = figure_t2.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure_t2.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_genders = ['Male / Male',
                          'Male / Female',
                          'Male / Other',
                          'Male / Non-human',
                          'Male / Not applicable',
                          
                          'Female / Male',
                          'Female / Female',
                          'Female / Other',
                          'Female / Non-human',
                          'Female / Not applicable',
                          
                          'Other / All combined']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_genders = category_percent_genders[category_order_genders]

# Define custom colors for each category
custom_colors_genders = ['#385AC2',
                         '#5580D0',
                         '#6CACEB',
                         '#A4C8ED',
                         '#CFE3F7',
                         
                         '#AE305D',
                         '#CF5D5F',
                         '#E3937B',
                         '#EFB8A7',
                         '#FBD9CF',
                         
                         '#8B3FCF']

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
           labelspacing = 2.0,
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

# Saving image-------------------------------------------
plt.savefig("./Figures/00 author and protagonist gender.png", bbox_inches = 'tight')
#plt.savefig("./Figures/00 author and protagonist gender.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/00 author and protagonist gender.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 24 - Author and protagonist gender
print("  Making Variance in Answers...")

#----------------------------------------------------------------------------------
# Creates a figure object with size 10x14 inches
figure_t3 = plt.figure(24, figsize = (10, 14))
gs = figure_t3.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure_t3.add_subplot(gs[0])

# Heatmap-------------------------------------------
# Set var_flag to the measure of variation in answers that you want:
# 1 Difference two by two of the answers
# 2 Percent Agreement / Mode Consistency
# 3 Shannon Entropy (Diversity Index)
var_flag = 3

# 1 Difference two by two of the answers
if var_flag == 1:
    sns.heatmap(df_mean_difference, 
                annot=True, 
                cmap="coolwarm",
                fmt=".2f", 
                cbar_kws={'label': 'Mean Difference Score'},
                annot_kws={"size": 8})
    #annot=True, fmt=".1%", cmap="YlGnBu", cbar_kws={'format': '%.0f%%'}

# 2 Percent Agreement / Mode Consistency
elif var_flag == 2:
    sns.heatmap(df_percent_agreement, 
                annot=True, 
                cmap="coolwarm_r", # coolwarm_r:reversed coolwarm
                fmt=".0f", 
                cbar_kws={'label': 'Percent Agreement'},
                annot_kws={"size": 8})
    #annot=True, fmt=".1%", cmap="YlGnBu", cbar_kws={'format': '%.0f%%'}

# 3 Shannon Entropy (Diversity Index)
elif var_flag == 3:
    sns.heatmap(df_entropy, 
                annot=True, 
                cmap="coolwarm",
                fmt=".2f", 
                cbar_kws={'label': 'Shannon Index'},
                annot_kws={"size": 8})
    #annot=True, fmt=".1%", cmap="YlGnBu", cbar_kws={'format': '%.0f%%'}

# Add thicker lines to separate the last row and column
"""
ax.get_xlim() returns a tuple with the x-axis limits, like (0.0, 4.0).
Using *ax.get_xlim() unpacks this tuple into two separate arguments. 
So, hlines interprets it as hlines(y, xmin=0.0, xmax=4.0).
"""
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

# Saving image-------------------------------------------
plt.savefig("./Figures/00 variation.png", bbox_inches = 'tight')
#plt.savefig("./Figures/00 variation.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/00 variation.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
print("All done.")

# Showing figures-------------------------------------------------------------------------------------------
plt.show() # You must call plt.show() to make graphics appear.