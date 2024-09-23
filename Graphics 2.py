import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

#----------------------------------------------------------------------------------
# reading the data
df_filtered = pd.read_csv("./Data/sci-fi_books_FILTERED.csv", sep=";")

df_200 = pd.read_csv("./Data/top_sci-fi_books_200_PER_DECADE.csv", sep=";")

df_200_AI = pd.read_csv("./Data/AI_answers_to_sci-fi_books.csv", sep=";")

#print(df_200_AI.info())
#print(df.head())

#----------------------------------------------------------------------------------
# Excluding books of before 1860 (allmost none)

mask_all = df_filtered['decade'] >= 1860
df_filtered = df_filtered[mask_all]

mask_200 = df_200['decade'] >= 1860
df_200 = df_200[mask_200]

mask_200_AI = df_200_AI['decade'] >= 1860
df_200_AI = df_200_AI[mask_200_AI]

#----------------------------------------------------------------------------------
# For the boxplots

# Add a column to each DataFrame to label the dataset
df_filtered['dataset'] = 'All sample'
df_200['dataset'] = 'Top 200'

# Concatenate dataframes
df_filtered_200 = pd.concat([df_filtered, df_200])

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

book_per_decade_200 = df_200['decade'].value_counts()
mean_per_decade_200 = df_200.groupby('decade')[['rate', 'ratings']].mean()
sdv_per_decade_200 = df_200.groupby('decade')[['rate', 'ratings']].std()

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

#------------------------------------------
# Count the occurrences of each category per decade
category_counts_series = pd.crosstab(df_200['decade'], df_200['series'])
# Normalize the counts to get percentages
category_percent_series = category_counts_series.div(category_counts_series.sum(axis = 1), axis = 0) * 100

#print(df_all_200.info())
#print(df_all_200)

#----------------------------------------------------------------------------------
# Processing for the questions/answers

# 1 soft hard------------------------------------------
# Count the occurrences of each category per decade
category_counts_1 = pd.crosstab(df_200_AI['decade'], df_200_AI['1 soft hard'])
# Normalize the counts to get percentages
category_percent_1 = category_counts_1.div(category_counts_1.sum(axis = 1), axis = 0) * 100

# 2 time------------------------------------------
# Count the occurrences of each category per decade
category_counts_2 = pd.crosstab(df_200_AI['decade'], df_200_AI['2 time'])
# Normalize the counts to get percentages
category_percent_2 = category_counts_2.div(category_counts_2.sum(axis = 1), axis = 0) * 100

# 3 tone------------------------------------------
# Count the occurrences of each category per decade
category_counts_3 = pd.crosstab(df_200_AI['decade'], df_200_AI['3 tone'])
# Normalize the counts to get percentages
category_percent_3 = category_counts_3.div(category_counts_3.sum(axis = 1), axis = 0) * 100

# 4 setting------------------------------------------
# Count the occurrences of each category per decade
category_counts_4 = pd.crosstab(df_200_AI['decade'], df_200_AI['4 setting'])
# Normalize the counts to get percentages
category_percent_4 = category_counts_4.div(category_counts_4.sum(axis = 1), axis = 0) * 100

# 5 on Earth------------------------------------------
# Count the occurrences of each category per decade
category_counts_5 = pd.crosstab(df_200_AI['decade'], df_200_AI['5 on Earth'])
# Normalize the counts to get percentages
category_percent_5 = category_counts_5.div(category_counts_5.sum(axis = 1), axis = 0) * 100

# 6 post apocalyptic------------------------------------------
# Count the occurrences of each category per decade
category_counts_6 = pd.crosstab(df_200_AI['decade'], df_200_AI['6 post apocalyptic'])
# Normalize the counts to get percentages
category_percent_6 = category_counts_6.div(category_counts_6.sum(axis = 1), axis = 0) * 100

# 7 aliens------------------------------------------
# Count the occurrences of each category per decade
category_counts_7 = pd.crosstab(df_200_AI['decade'], df_200_AI['7 aliens'])
# Normalize the counts to get percentages
category_percent_7 = category_counts_7.div(category_counts_7.sum(axis = 1), axis = 0) * 100

# 8 aliens are------------------------------------------
# Count the occurrences of each category per decade
category_counts_8 = pd.crosstab(df_200_AI['decade'], df_200_AI['8 aliens are'])
# Normalize the counts to get percentages
category_percent_8 = category_counts_8.div(category_counts_8.sum(axis = 1), axis = 0) * 100

# 9 robots and AI------------------------------------------
# Count the occurrences of each category per decade
category_counts_9 = pd.crosstab(df_200_AI['decade'], df_200_AI['9 robots and AI'])
# Normalize the counts to get percentages
category_percent_9 = category_counts_9.div(category_counts_9.sum(axis = 1), axis = 0) * 100

# 10 robots and AI are------------------------------------------
# Count the occurrences of each category per decade
category_counts_10 = pd.crosstab(df_200_AI['decade'], df_200_AI['10 robots and AI are'])
# Normalize the counts to get percentages
category_percent_10 = category_counts_10.div(category_counts_10.sum(axis = 1), axis = 0) * 100

# 11 protagonist------------------------------------------
# Count the occurrences of each category per decade
category_counts_11 = pd.crosstab(df_200_AI['decade'], df_200_AI['11 protagonist'])
# Normalize the counts to get percentages
category_percent_11 = category_counts_11.div(category_counts_11.sum(axis = 1), axis = 0) * 100

# 12 protagonist is------------------------------------------
# Count the occurrences of each category per decade
category_counts_12 = pd.crosstab(df_200_AI['decade'], df_200_AI['12 protagonist is'])
# Normalize the counts to get percentages
category_percent_12 = category_counts_12.div(category_counts_12.sum(axis = 1), axis = 0) * 100

# 13 tech and science------------------------------------------
# Count the occurrences of each category per decade
category_counts_13 = pd.crosstab(df_200_AI['decade'], df_200_AI['13 tech and science'])
# Normalize the counts to get percentages
category_percent_13 = category_counts_13.div(category_counts_13.sum(axis = 1), axis = 0) * 100

# 14 virtual------------------------------------------
# Count the occurrences of each category per decade
category_counts_14 = pd.crosstab(df_200_AI['decade'], df_200_AI['14 virtual'])
# Normalize the counts to get percentages
category_percent_14 = category_counts_14.div(category_counts_14.sum(axis = 1), axis = 0) * 100

# 15 social issues------------------------------------------
# Count the occurrences of each category per decade
category_counts_15 = pd.crosstab(df_200_AI['decade'], df_200_AI['15 social issues'])
# Normalize the counts to get percentages
category_percent_15 = category_counts_15.div(category_counts_15.sum(axis = 1), axis = 0) * 100

# 16 enviromental------------------------------------------
# Count the occurrences of each category per decade
category_counts_16 = pd.crosstab(df_200_AI['decade'], df_200_AI['16 enviromental'])
# Normalize the counts to get percentages
category_percent_16 = category_counts_16.div(category_counts_16.sum(axis = 1), axis = 0) * 100

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
        label = "All sample")

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
        label = "Top 200")

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
plt.savefig("./Figures/Quantities.png", bbox_inches = 'tight')
#plt.savefig("./Figures/Quantity_all.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/Quantity_all.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Figure 2, Series
print("  Making series...")

# Creates a figure object with size 12x6 inches
figure_c2 = plt.figure(2, figsize = (12, 6))
gs = figure_c2.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure_c2.add_subplot(gs[0])

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
plt.savefig("./Figures/Series.png", bbox_inches = 'tight')
#plt.savefig("./Figures/Series.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/Series.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Figure 3, Quantities
print("  Making rate and ratings...")

# Creates a figure object with size 14x8 inches
figure_c3 = plt.figure(3, figsize = (14, 8))
gs = figure_c3.add_gridspec(ncols = 2, nrows = 1)

#figure_c3.subplots_adjust(hspace = 0.5)

#-----------------------------------------
# Create the main plot
ax1 = figure_c3.add_subplot(gs[0])

# Specify custom colors for each dataset
custom_palette = {'All sample': '#385AC2',
                  'Top 200': '#AE305D'}

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
ax2 = figure_c3.add_subplot(gs[1])

# Specify custom colors for each dataset
custom_palette = {'All sample': '#385AC2',
                  'Top 200': '#AE305D'}

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
plt.savefig("./Figures/Rates_and_Ratings.png", bbox_inches = 'tight')
#plt.savefig("./Figures/Rates_and_Ratings.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/Rates_and_Ratings.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# figures for the questions/answers
#----------------------------------------------------------------------------------
# Figure 4 - 1 soft hard
print("  Making 1 soft hard...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure1 = plt.figure(4, figsize = (12, 6))
gs = figure1.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure1.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_1 = ['Very soft',
                    'Soft',
                    'Mixed', 
                    'Hard',
                    'Very hard']
# Reorder the columns in the DataFrame according to the desired category order
category_percent_1 = category_percent_1[category_order_1]

# Define custom colors for each category
custom_colors_1 = ['#385AC2',
                   '#5580D0',
                   '#8B3FCF',
                   '#CF5D5F',
                   '#AE305D']

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
plt.savefig("./Figures/1 soft hard.png", bbox_inches = 'tight')
#plt.savefig("./Figures/1 soft hard.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/1 soft hard.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 5 - 2 time
print("  Making 2 time...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure2 = plt.figure(5, figsize = (12, 6))
gs = figure2.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure2.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_2 = ['Distant past',
                    'Far past',
                    'Near past',
                    'Present',
                    'Near future',
                    'Far future',
                    'Distant future',

                    'Multiple timelines',
                    'Uncertain']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_2 = category_percent_2[category_order_2]

# Define custom colors for each category
custom_colors_2 = ['#AE305D', # Distant past
                   '#CF5D5F',
                   '#E3937B',
                   '#8B3FCF',
                   '#6CACEB',
                   '#5580D0',
                   '#385AC2', # Distant future

                   'green',     # Multiple timelines
                   'lightgrey'] # Uncertain

# Bar plot-------------------------------------------
category_percent_2.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_2,
                        width = 1.0,
                        alpha = 1.0,
                        label = "2 time")

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
plt.savefig("./Figures/2 time.png", bbox_inches = 'tight')
#plt.savefig("./Figures/2 time.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/2 time.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 6 - 3 tone
print("  Making 3 tone...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure3 = plt.figure(6, figsize = (12, 6))
gs = figure3.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure3.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_3 = ['Very pessimistic',
                    'Pessimistic',
                    'Balanced',
                    'Optimistic',
                    'Very optimistic']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_3 = category_percent_3[category_order_3]

# Define custom colors for each category
custom_colors_3 = ['#AE305D',
                   '#CF5D5F',
                   '#8B3FCF',
                   '#5580D0',
                   '#385AC2']

# Bar plot-------------------------------------------
category_percent_3.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_3,
                        width = 1.0,
                        alpha = 1.0,
                        label = "3 tone")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("What is the tone of the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
plt.savefig("./Figures/3 tone.png", bbox_inches = 'tight')
#plt.savefig("./Figures/3 tone.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/3 tone.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 7 - 4 setting
print("  Making 4 setting...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure4 = plt.figure(7, figsize = (12, 6))
gs = figure4.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure4.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_4 = ['Dystopic',
                    'Leaning dystopic',
                    'Balanced',
                    'Leaning utopic',
                    'Utopic',

                    'Uncertain']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_4 = category_percent_4[category_order_4]

# Define custom colors for each category
custom_colors_4 = ['#AE305D',
                   '#CF5D5F',
                   '#8B3FCF',
                   '#5580D0',
                   '#385AC2',

                   'lightgrey']

# Bar plot-------------------------------------------
category_percent_4.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_4,
                        width = 1.0,
                        alpha = 1.0,
                        label = "4 setting")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("How is the social and political setting depicted in the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
plt.savefig("./Figures/4 setting.png", bbox_inches = 'tight')
#plt.savefig("./Figures/4 setting.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/4 setting.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 8 - 5 on Earth
print("  Making 5 on Earth...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure5 = plt.figure(8, figsize = (12, 6))
gs = figure5.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure5.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_5 = ['Yes',
                    'No']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_5 = category_percent_5[category_order_5]

# Define custom colors for each category
custom_colors_5 = ['#AE305D',
                   '#385AC2']

# Bar plot-------------------------------------------
category_percent_5.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_5,
                        width = 1.0,
                        alpha = 1.0,
                        label = "5 on Earth")

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
plt.savefig("./Figures/5 on Earth.png", bbox_inches = 'tight')
#plt.savefig("./Figures/5 on Earth.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/5 on Earth.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 9 - 6 post apocalyptic
print("  Making 6 post apocalyptic...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure6 = plt.figure(9, figsize = (12, 6))
gs = figure6.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure6.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_6 = ['Yes',
                    'Somewhat',
                    'No']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_6 = category_percent_6[category_order_6]

# Define custom colors for each category
custom_colors_6 = ['#AE305D',
                   '#8B3FCF',
                   '#385AC2']

# Bar plot-------------------------------------------
category_percent_6.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_6,
                        width = 1.0,
                        alpha = 1.0,
                        label = "6 post apocalyptic")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Is the story set in a post-apocalyptic world (following a major civilization-collapse event)?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
plt.savefig("./Figures/6 post apocalyptic.png", bbox_inches = 'tight')
#plt.savefig("./Figures/6 post apocalyptic.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/6 post apocalyptic.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 10 - 7 aliens
print("  Making 7 aliens...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure7 = plt.figure(10, figsize = (12, 6))
gs = figure7.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure7.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_7 = ['Yes',
                    'No']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_7 = category_percent_7[category_order_7]

# Define custom colors for each category
custom_colors_7 = ['#AE305D',
                   '#385AC2']

# Bar plot-------------------------------------------
category_percent_7.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_7,
                        width = 1.0,
                        alpha = 1.0,
                        label = "7 aliens")

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
plt.savefig("./Figures/7 aliens.png", bbox_inches = 'tight')
#plt.savefig("./Figures/7 aliens.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/7 aliens.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 11 - 8 aliens are
print("  Making 8 aliens are...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure8 = plt.figure(11, figsize = (12, 6))
gs = figure8.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure8.add_subplot(gs[0])

# Define the desired order of the categories
category_order_8 = ['Bad', 
                    'Leaning bad',
                    'Ambivalent', 
                    'Leaning good',
                    'Good',

                    'Uncertain',
                    'Not applicable']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_8 = category_percent_8[category_order_8]

# Define custom colors for each category
custom_colors_8 = ['#AE305D',
                   '#CF5D5F',
                   '#8B3FCF',
                   '#5580D0',
                   '#385AC2',

                   'green',
                   'lightgrey']

# Bar plot-------------------------------------------
category_percent_8.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_8,
                        width = 1.0,
                        alpha = 1.0,
                        label = "8 aliens are")

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
plt.savefig("./Figures/8 aliens are.png", bbox_inches = 'tight')
#plt.savefig("./Figures/8 aliens are.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/8 aliens are.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 12 - 9 robots and AI
print("  Making 9 robots and AI...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure9 = plt.figure(12, figsize = (12, 6))
gs = figure9.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure9.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_9 = ['Yes', 
                    'No']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_9 = category_percent_9[category_order_9]

# Define custom colors for each category
custom_colors_9 = ['#AE305D',
                   '#385AC2']

# Bar plot-------------------------------------------
category_percent_9.plot(kind = 'bar',
                        stacked = True,
                        ax = ax1,
                        color = custom_colors_9,
                        width = 1.0,
                        alpha = 1.0,
                        label = "9 robots and AI")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Are there any depictions of robots or artificial intelligences in the story? \n(just automatic systems or programs do not count)", fontsize = 14, pad = 5, color = custom_dark_gray)
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
plt.savefig("./Figures/9 robots and AI.png", bbox_inches = 'tight')
#plt.savefig("./Figures/9 robots and AI.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/9 robots and AI.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 13 - 10 robots and AI are
print("  Making 10 robots and AI are...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure10 = plt.figure(13, figsize = (12, 6))
gs = figure10.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure10.add_subplot(gs[0])
# Figure 12 - 10 robots and AI are

#-------------------------------------------
# Define the desired order of the categories
category_order_10 = ['Bad', 
                     'Leaning bad', 
                     'Ambivalent', 
                     'Leaning good',
                     'Good',

                     'Uncertain',
                     'Not applicable']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_10 = category_percent_10[category_order_10]

# Define custom colors for each category
custom_colors_10 = ['#AE305D',
                    '#CF5D5F',
                    '#8B3FCF',
                    '#5580D0',
                    '#385AC2',

                    'green',
                    'lightgrey']

# Bar plot-------------------------------------------
category_percent_10.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_10,
                         width = 1.0,
                         alpha = 1.0,
                         label = "10 robots and AI are")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
#ax1.set_ylabel("Fraction [%]", fontsize = 12, color = custom_dark_gray)
ax1.set_title("How are the robots or artificial intelligences generally depicted?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
plt.savefig("./Figures/10 robots and AI are.png", bbox_inches = 'tight')
#plt.savefig("./Figures/10 robots and AI are.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/10 robots and AI are.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 14 - 11 protagonist
print("  Making 11 protagonist...")
#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure11 = plt.figure(14, figsize = (12, 6))
gs = figure11.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure11.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_11 = ['Yes', 
                     'No']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_11 = category_percent_11[category_order_11]

# Define custom colors for each category
custom_colors_11 = ['#AE305D',
                    '#385AC2']

# Bar plot-------------------------------------------
category_percent_11.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_11,
                         width = 1.0,
                         alpha = 1.0,
                         label = "11 protagonist")

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
plt.savefig("./Figures/11 protagonist.png", bbox_inches = 'tight')
#plt.savefig("./Figures/11 protagonist.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/11 protagonist.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 15 - 12 protagonist is
print("  Making 12 protagonist is...")
#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure12 = plt.figure(15, figsize = (12, 6))
gs = figure12.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure12.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_12 = ['Male', 
                     'Female',

                     'Other',
                     'Non-human',

                     'Not applicable']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_12 = category_percent_12[category_order_12]

# Define custom colors for each category
custom_colors_12 = ['#385AC2',
                    '#AE305D',

                    '#8B3FCF',
                    'green',

                    'lightgrey']

# Bar plot-------------------------------------------
category_percent_12.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_12,
                         width = 1.0,
                         alpha = 1.0,
                         label = "12 protagonist is")

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
plt.savefig("./Figures/12 protagonist is.png", bbox_inches = 'tight')
#plt.savefig("./Figures/12 protagonist is.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/12 protagonist is.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 16 - 13 tech and science
print("  Making 13 tech and science...")
#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure13 = plt.figure(16, figsize = (12, 6))
gs = figure13.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure13.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_13 = ['Bad', 
                     'Leaning bad', 
                     'Ambivalent', 
                     'Leaning good',
                     'Good',

                     'Uncertain']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_13 = category_percent_13[category_order_13]

# Define custom colors for each category
custom_colors_13 = ['#AE305D',
                    '#CF5D5F',
                    '#8B3FCF',
                    '#5580D0',
                    '#385AC2',

                    'lightgrey']

# Bar plot-------------------------------------------
category_percent_13.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_13,
                         width = 1.0,
                         alpha = 1.0,
                         label = "13 tech and science")

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
plt.savefig("./Figures/13 tech and science.png", bbox_inches = 'tight')
#plt.savefig("./Figures/13 tech and science.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/13 tech and science.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 17 - 14 virtual
print("  Making 14 virtual...")
#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure14 = plt.figure(17, figsize = (12, 6))
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
ax1.set_title("Is virtual reality or immersive digital environments \n(e.g., simulations, augmented reality) present in the story?", fontsize = 14, pad = 5, color = custom_dark_gray)
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
# Figure 18 - 15 social issues
print("  Making 15 social issues...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure15 = plt.figure(18, figsize = (12, 6))
gs = figure15.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure15.add_subplot(gs[0])

#-------------------------------------------
# Define the desired order of the categories
category_order_15 = ['Core', 
                     'Major',
                     'Minor',

                     'Absent']

# Reorder the columns in the DataFrame according to the desired category order
category_percent_15 = category_percent_15[category_order_15]

# Define custom colors for each category
custom_colors_15 = ['#385AC2',
                    '#5580D0',
                    '#6CACEB',

                    '#AE305D']

# Bar plot-------------------------------------------
category_percent_15.plot(kind = 'bar',
                         stacked = True,
                         ax = ax1,
                         color = custom_colors_15,
                         width = 1.0,
                         alpha = 1.0,
                         label = "13 social issues")

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
plt.savefig("./Figures/15 social issues.png", bbox_inches = 'tight')
#plt.savefig("./Figures/15 social issues.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/15 social issues.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
# Figure 19 - 16 enviromental
print("  Making 16 enviromental...")

#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure16 = plt.figure(19, figsize = (12, 6))
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
                         label = "16 enviromental")

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
plt.savefig("./Figures/16 enviromental.png", bbox_inches = 'tight')
#plt.savefig("./Figures/16 enviromental.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/16 enviromental.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
print("All done.")

# Showing figures-------------------------------------------------------------------------------------------
plt.show()  # You must call plt.show() to make graphics appear.
