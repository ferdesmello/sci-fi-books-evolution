import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#----------------------------------------------------------------------------------
# reading the data
df_filtered = pd.read_csv("./Data/sci-fi_books_FILTERED.csv", sep=";")
df_200 = pd.read_csv("./Data/top_sci-fi_books_200_PER_DECADE.csv", sep=";")
df_200_AI = pd.read_csv("./Data/AI_answers_to_sci-fi_books.csv", sep=";")

print(df_200_AI.info())
#print(df.head())

#----------------------------------------------------------------------------------
# General information of the FILTERED sample of books
#'title', 'author', 'year', 'decade', 'rate', 'ratings', 'genres', 'synopsis', 'review', 'url'
print("\nFILTERED books.")

book_per_decade = df_filtered['decade'].value_counts()#.sort_values(ascending=False)
mean_per_decade = df_filtered.groupby('decade')[['rate', 'ratings']].mean()
sdv_per_decade = df_filtered.groupby('decade')[['rate', 'ratings']].std()

# Creating a dictionary by passing Series objects as values
frame_1 = {'quantity': book_per_decade,
         'avr rate': mean_per_decade['rate'],
         'std rate': sdv_per_decade['rate'],
         'avr ratings': mean_per_decade['ratings'],
         'std ratings': sdv_per_decade['ratings']}
 
# Creating DataFrame by passing Dictionary
df_all = pd.DataFrame(frame_1)
df_all = (df_all
          .reset_index(drop=False)
          .sort_values(by = ['decade'], ascending=True)
          .reset_index(drop=True))

print(df_all.info())
print(df_all)

#----------------------------------------------------------------------------------
# General information of the 200 PER DECADE sample of books
#'title', 'author', 'year', 'decade', 'rate', 'ratings', 'genres', 'synopsis', 'review', 'url'
print("\n200 PER DECADE books.")

book_per_decade_200 = df_200['decade'].value_counts()#.sort_values(ascending=False)
mean_per_decade_200 = df_200.groupby('decade')[['rate', 'ratings']].mean()
sdv_per_decade_200 = df_200.groupby('decade')[['rate', 'ratings']].std()

# Creating a dictionary by passing Series objects as values
frame_2 = {'quantity': book_per_decade_200,
         'avr rate': mean_per_decade_200['rate'],
         'std rate': sdv_per_decade_200['rate'],
         'avr ratings': mean_per_decade_200['ratings'],
         'std ratings': sdv_per_decade_200['ratings']}
 
# Creating DataFrame by passing Dictionary
df_all_200 = pd.DataFrame(frame_2)
df_all_200 = (df_all_200
              .reset_index(drop=False)
              .sort_values(by = ['decade'], ascending=True)
              .reset_index(drop=True))

print(df_all_200.info())
print(df_all_200)

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Making figures
print("Making the figures...")
# Custom dark gray color
custom_dark_gray = (0.2, 0.2, 0.2)
#-------------------------------------------------------------------------------------
# Figure 1, Quantities
print("  Making figure 1...")

# Creates a figure object with size 12x6 inches
figure1 = plt.figure(1, figsize = (12, 6))
gs = figure1.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure1.add_subplot(gs[0])

# Bar plot
ax1.bar(x = df_all['decade'],
        height = df_all['quantity'], 
        width=9, 
        #bottom=None, 
        align='center', 
        #data=None,
        color=(0.1, 0.1, 0.8, 0.5),
        edgecolor=custom_dark_gray,
        linewidth=0.0,
        # #tick_label=0,
        label = "All sample, quantity")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
ax1.set_ylabel("Quantity", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Books per Decade in the Sample", fontsize = 16, pad = 20, color = custom_dark_gray)
#ax1.grid(True, linestyle = ':', linewidth = '1')

# Axes-------------------------------------------
#ax1.set_xlim(0, 7)
#ax1.set_ylim(260, 380)

ax1.minorticks_on()
ax1.tick_params(which = "major", direction = "out", length = 3, labelsize = 10, color = custom_dark_gray)
ax1.tick_params(which = "minor", direction = "out", length = 3, color = custom_dark_gray)
ax1.tick_params(which = "both", bottom = True, top = True, left = True, right = True, color = custom_dark_gray)
ax1.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False, color = custom_dark_gray)
ax1.spines['right'].set_color(custom_dark_gray)
ax1.spines['top'].set_color(custom_dark_gray)
ax1.spines['left'].set_color(custom_dark_gray)
ax1.spines['bottom'].set_color(custom_dark_gray)
ax1.tick_params(axis = 'both', colors = custom_dark_gray)

# Saving image-------------------------------------------
plt.savefig("./Figures/Quantity_all.png", bbox_inches = 'tight')
#plt.savefig("./Figures/Quantity_all.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/Quantity_all.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Figure 2, Quantities
print("  Making figure 2...")

# Creates a figure object with size 12x6 inches
figure2 = plt.figure(2, figsize = (12, 6))
gs = figure2.add_gridspec(ncols = 2, nrows = 1)

# Create the main plot
ax1 = figure2.add_subplot(gs[0])

# Error bar plot
ax1.errorbar(x = df_all['decade'], 
             y = df_all['avr rate'], 
             #xerr = 0.15,
             yerr = df_all['std rate'],
             marker = "o",
             c = (0.1, 0.1, 0.8, 0.5),
             capsize = 3,
             lw = 2,
             zorder = 1,
             label = "All sample, rate")

# Error bar plot
ax1.errorbar(x = df_all_200['decade'], 
             y = df_all_200['avr rate'], 
             #xerr = 0.15,
             yerr = df_all_200['std rate'],
             marker = "o",
             c = (0.8, 0.1, 0.1, 0.5),
             capsize = 3,
             lw = 2,
             zorder = 2,
             label = "200 sample, rate")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
ax1.set_ylabel("Average rate", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Average Rate per Decade", fontsize = 16, pad = 20, color = custom_dark_gray)
#ax1.grid(True, linestyle = ':', linewidth = '1')

# Axes-------------------------------------------
#ax1.set_xlim(0, 7)
#ax1.set_ylim(260, 380)

ax1.minorticks_on()
ax1.tick_params(which = "major", direction = "out", length = 3, labelsize = 10, color = custom_dark_gray)
ax1.tick_params(which = "minor", direction = "out", length = 3, color = custom_dark_gray)
ax1.tick_params(which = "both", bottom = True, top = True, left = True, right = True, color = custom_dark_gray)
ax1.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False, color = custom_dark_gray)
ax1.spines['right'].set_color(custom_dark_gray)
ax1.spines['top'].set_color(custom_dark_gray)
ax1.spines['left'].set_color(custom_dark_gray)
ax1.spines['bottom'].set_color(custom_dark_gray)
ax1.tick_params(axis = 'both', colors = custom_dark_gray)

#-----------------------------------------
# Create the main plot
ax2 = figure2.add_subplot(gs[1])

# Error bar plot
ax2.errorbar(x = df_all['decade'], 
             y = df_all['avr ratings'], 
             #xerr = 0.15,
             yerr = df_all['std ratings'],
             marker = "o",
             c = (0.1, 0.1, 0.8, 0.5),
             capsize = 3,
             lw = 2,
             zorder = 1,
             label = "All sample, ratings")

# Error bar plot
ax2.errorbar(x = df_all_200['decade'], 
             y = df_all_200['avr ratings'], 
             #xerr = 0.15,
             yerr = df_all_200['std ratings'],
             marker = "o",
             c = (0.8, 0.1, 0.1, 0.5),
             capsize = 3,
             lw = 2,
             zorder = 2,
             label = "200 sample, ratings")

# Design-------------------------------------------
ax2.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
ax2.set_ylabel("Average number of ratings", fontsize = 12, color = custom_dark_gray)
ax2.set_title("Average Number of Ratings per Decade", fontsize = 16, pad = 20, color = custom_dark_gray)
#ax2.grid(True, linestyle = ':', linewidth = '1')

# Axes-------------------------------------------
#ax2.set_xlim(0, 7)
#ax2.set_ylim(260, 380)
ax2.set_yscale("log")

ax2.minorticks_on()
ax2.tick_params(which = "major", direction = "out", length = 3, labelsize = 10, color = custom_dark_gray)
ax2.tick_params(which = "minor", direction = "out", length = 3, color = custom_dark_gray)
ax2.tick_params(which = "both", bottom = True, top = True, left = True, right = True, color = custom_dark_gray)
ax2.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False, color = custom_dark_gray)
ax2.spines['right'].set_color(custom_dark_gray)
ax2.spines['top'].set_color(custom_dark_gray)
ax2.spines['left'].set_color(custom_dark_gray)
ax2.spines['bottom'].set_color(custom_dark_gray)
ax2.tick_params(axis = 'both', colors = custom_dark_gray)

# Saving image-------------------------------------------
plt.savefig("./Figures/Average_rates.png", bbox_inches = 'tight')
#plt.savefig("./Figures/Average_rates.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/Average_rates.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# Figure 3, Quantities
print("  Making figure 3...")

# Creates a figure object with size 12x6 inches
figure3 = plt.figure(3, figsize = (14, 6))
gs = figure3.add_gridspec(ncols = 3, nrows = 1)

#-----------------------------------------
# Create the main plot
ax1 = figure3.add_subplot(gs[0])

# Create the box plot
ax1 = sns.boxplot(data=df_filtered, 
                  x='decade', 
                  y='rate', 
                  color="green")

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
ax1.set_ylabel("Average rate", fontsize = 12, color = custom_dark_gray)
ax1.set_title("Average Rate per Decade", fontsize = 16, pad = 20, color = custom_dark_gray)
#ax1.grid(True, linestyle = ':', linewidth = '1')

# Axes-------------------------------------------
#ax1.set_xlim(0, 7)
#ax1.set_ylim(260, 380)

for tick in ax1.get_xticklabels():
    tick.set_rotation(45)

ax1.minorticks_on()
ax1.tick_params(which = "major", direction = "out", length = 3, labelsize = 10, color = custom_dark_gray)
ax1.tick_params(which = "minor", direction = "out", length = 0, color = custom_dark_gray)
ax1.tick_params(which = "both", bottom = True, top = True, left = True, right = True, color = custom_dark_gray)
ax1.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False, color = custom_dark_gray)
ax1.spines['right'].set_color(custom_dark_gray)
ax1.spines['top'].set_color(custom_dark_gray)
ax1.spines['left'].set_color(custom_dark_gray)
ax1.spines['bottom'].set_color(custom_dark_gray)
ax1.tick_params(axis = 'both', colors = custom_dark_gray)

#----------------------------------------------------------------------------------
# Create the main plot
ax2 = figure3.add_subplot(gs[1])

# Create the box plot
ax2 = sns.swarmplot(data=df_filtered, 
                    x='decade',
                    y='rate',
                    orient='v',
                    size=0.4,
                    color="red")

# Design-------------------------------------------
ax2.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
ax2.set_ylabel("Average rate", fontsize = 12, color = custom_dark_gray)
ax2.set_title("Average Rate per Decade", fontsize = 16, pad = 20, color = custom_dark_gray)
#ax2.grid(True, linestyle = ':', linewidth = '1')

# Axes-------------------------------------------
#ax2.set_xlim(0, 7)
#ax2.set_ylim(260, 380)

ax2.set_xticks(ax2.get_xticks()[::2])

"""ax2.minorticks_on()
ax2.tick_params(which = "major", direction = "out", length = 3, labelsize = 10, color = custom_dark_gray)
ax2.tick_params(which = "minor", direction = "out", length = 3, color = custom_dark_gray)
ax2.tick_params(which = "both", bottom = True, top = True, left = True, right = True, color = custom_dark_gray)
ax2.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False, color = custom_dark_gray)
ax2.spines['right'].set_color(custom_dark_gray)
ax2.spines['top'].set_color(custom_dark_gray)
ax2.spines['left'].set_color(custom_dark_gray)
ax2.spines['bottom'].set_color(custom_dark_gray)
ax2.tick_params(axis = 'both', colors = custom_dark_gray)"""

#----------------------------------------------------------------------------------
# Create the main plot
ax3 = figure3.add_subplot(gs[2])

# Create the box plot
ax3 = sns.violinplot(data=df_filtered,
                     x='decade',
                     y='rate',
                     orient='v',
                     color="blue")

# Design-------------------------------------------
ax3.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
ax3.set_ylabel("Average rate", fontsize = 12, color = custom_dark_gray)
ax3.set_title("Average Rate per Decade", fontsize = 16, pad = 20, color = custom_dark_gray)
#ax3.grid(True, linestyle = ':', linewidth = '1')

# Axes-------------------------------------------
#ax2.set_xlim(0, 7)
#ax2.set_ylim(260, 380)

ax3.set_xticks(ax3.get_xticks()[::2])

# Optional: Rotate the labels for readability
plt.xticks(rotation=45)

"""ax3.minorticks_on()
ax3.tick_params(which = "major", direction = "out", length = 3, labelsize = 10, color = custom_dark_gray)
ax3.tick_params(which = "minor", direction = "out", length = 3, color = custom_dark_gray)
ax3.tick_params(which = "both", bottom = True, top = True, left = True, right = True, color = custom_dark_gray)
ax3.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False, color = custom_dark_gray)
ax3.spines['right'].set_color(custom_dark_gray)
ax3.spines['top'].set_color(custom_dark_gray)
ax3.spines['left'].set_color(custom_dark_gray)
ax3.spines['bottom'].set_color(custom_dark_gray)
ax3.tick_params(axis = 'both', colors = custom_dark_gray)"""

#----------------------------------------------------------------------------------
# Figure 4
print("  Making figure 4...")

# Count the occurrences of each category per decade
#category_counts = pd.crosstab(df_200_AI['decade'], df_200_AI['1 soft hard'])
#category_counts = pd.crosstab(df_200_AI['decade'], df_200_AI['12 protagonist'])
#category_counts = pd.crosstab(df_200_AI['decade'], df_200_AI['3 tone'])
#category_counts = pd.crosstab(df_200_AI['decade'], df_200_AI['5 on Earth'])
#category_counts = pd.crosstab(df_200_AI['decade'], df_200_AI['6 post apocalyptic'])
#category_counts = pd.crosstab(df_200_AI['decade'], df_200_AI['8 aliens are'])
#category_counts = pd.crosstab(df_200_AI['decade'], df_200_AI['10 robots and AI are'])
#category_counts = pd.crosstab(df_200_AI['decade'], df_200_AI['13 social issues'])
#category_counts = pd.crosstab(df_200_AI['decade'], df_200_AI['14 enviromental'])
#category_counts = pd.crosstab(df_200_AI['decade'], df_200_AI['4 setting'])
category_counts = pd.crosstab(df_200_AI['decade'], df_200_AI['2 time'])


# Normalize the counts to get percentages
category_percent = category_counts.div(category_counts.sum(axis=1), axis=0) * 100


# Define the desired order of the categories
category_order = ['distant past', 'far past', 'near past', 'present', 
                  'near future', 'far future', 'distant future', 
                  'multiple timelines', 'uncertain']

# Reorder the columns in the DataFrame according to the desired category order
category_percent = category_percent[category_order]




print(category_percent.info())
print(category_percent)
#-------------------------------------------
# Creates a figure object with size 12x6 inches
figure4 = plt.figure(4, figsize = (12, 6))
gs = figure4.add_gridspec(ncols = 1, nrows = 1)

# Create the main plot
ax1 = figure4.add_subplot(gs[0])

# Define custom colors for each category
custom_colors = ['grey', 'red', 'green']  # Tomato, Steel Blue, Lime Green (adjust as needed)

# Bar plot
category_percent.plot(kind='bar',
                      stacked=True,
                      label="soft-hard",
                      ax=ax1)
                      #color=custom_colors)

# Design-------------------------------------------
ax1.set_xlabel("Decade", fontsize = 12, color = custom_dark_gray)
ax1.set_ylabel("Fraction", fontsize = 12, color = custom_dark_gray)
ax1.set_title("More soft or hard sci-fi?", fontsize = 16, pad = 20, color = custom_dark_gray)
#ax1.grid(True, linestyle = ':', linewidth = '1')

# Axes-------------------------------------------
#ax1.set_xlim(0, 7)
#ax1.set_ylim(260, 380)

ax1.minorticks_on()
ax1.tick_params(which = "major", direction = "out", length = 3, labelsize = 10, color = custom_dark_gray)
ax1.tick_params(which = "minor", direction = "out", length = 0, color = custom_dark_gray)
ax1.tick_params(which = "both", bottom = True, top = False, left = True, right = False, color = custom_dark_gray)
ax1.tick_params(labelbottom = True, labeltop = False, labelleft = True, labelright = False, color = custom_dark_gray)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_color(custom_dark_gray)
ax1.spines['bottom'].set_color(custom_dark_gray)
ax1.tick_params(axis = 'both', colors = custom_dark_gray)

# Saving image-------------------------------------------
plt.savefig("./Figures/Soft_hard.png", bbox_inches = 'tight')
#plt.savefig("./Figures/Quantity_all.eps", transparent = True, bbox_inches = 'tight')
# Transparence will be lost in .eps, save in .svg for transparences
#plt.savefig("./Figures/Quantity_all.svg", format = 'svg', transparent = True, bbox_inches = 'tight')

#----------------------------------------------------------------------------------



#----------------------------------------------------------------------------------



#----------------------------------------------------------------------------------

print("All done.")

# Showing figures-------------------------------------------------------------------------------------------
plt.show()  # You must call plt.show() to make graphics appear.
