import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# Path to your CSV file
file_paths = ['final_methods_mistral7B_fp32.csv', 'final_methods_llama7B_fp32.csv']
titles = ["CATS of Mistral-7B’s MLP", "CATS of Llama2-7B’s MLP"]

for idx in range(2):
    file_path = file_paths[idx]
    # Lists to store the extracted values
    sparsity_list = []
    dense_latency_list = []
    ours_latency_list = []
    optimal_latency_list = []

    # Open the file and read data
    with open(file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)

        # Extract data from each row
        for row in csv_reader:
            if row:  # Check if the row is not empty
                sparsity_list.append(row[0])
                dense_latency_list.append(row[2])
                ours_latency_list.append(row[-2])
                optimal_latency_list.append(row[-1])

        for _list in [sparsity_list, dense_latency_list, optimal_latency_list, ours_latency_list]:
            for idx in range(len(_list)):
                _list[idx] = float(_list[idx])
    print(ours_latency_list)
    # Creating a DataFrame from the lists
    data = {
        'Sparsity': sparsity_list,
        'Dense Latency': dense_latency_list,
        'Ours Latency': ours_latency_list,
        'Optimal Latency': optimal_latency_list
    }
    df = pd.DataFrame(data)


    # Set the line type as scatter plot+line plot
    sns.set_theme()
    # Plot the data
    sns.lineplot(data=df, x='Sparsity', y='Dense Latency', marker="o", errorbar=None)
    # sns.scatterplot(data=df, x='Density', y='Dense')
    sns.lineplot(data=df, x='Sparsity', y='Ours Latency', marker="o", errorbar=None)
    # sns.scatterplot(data=df, x='Density', y='Optimal')
    sns.lineplot(data=df, x='Sparsity', y='Optimal Latency', marker="o", errorbar=None)

    # Set the y-axis name as "Latency (ms)"
    plt.ylabel('Latency(ms)')
    # Set the x-axis name as "Density (%)"
    plt.xlabel('Sparsity')
    # Plot the legend
    plt.legend(['Dense', 'CATS-with-Custom-Kernel', 'Optimal',])

    # sns.move_legend(
    #     ax, "lower center",
    #     bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
    # )
    # Save the plot to a PDF file
    plt.savefig('latency.pdf')
    # Save the plot as an svg
    plt.savefig('latency.svg')
    plt.title(titles[idx])



