import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def line_bar_plot(x, y, title, x_label=None, y_label=None, save_path=None, unit=None):
    '''
    Create a combination line and bar plot.

    :param x: x-axis data
    :param y: y-axis data
    :param title: string
    :param x_label: string
    :param y_label: string
    :param save_path: path to save the plot (optional)
    :param unit: string ('billion'), divides y-values by 1 billion if specified
    :return: plt object or saved file path if save_path is provided
    '''

    if unit == 'billion':
        y = [i / 10**9 for i in y]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Bar plot
    plt.bar(x, y, color='skyblue', label='Bar Data', alpha=0.5, width=0.4)

    # Line plot (plotted on the same y-axis)
    plt.plot(x, y, color='r', label='Line Data', linewidth=2.5, marker='o')

    # Titles and labels
    plt.title(title, fontsize=16, fontweight='bold')
    if x_label:
        plt.xlabel(x_label, fontsize=14)
    if y_label:
        plt.ylabel(y_label, fontsize=14)

    # Add a legend
    plt.legend()

    if save_path:
        # Save the plot
        plt.savefig(save_path)
        return save_path
    else:
        # Return plt object for further use
        return plt
def pie_chart(data, labels, title, save_path=None, explode=None, colors=None):
    '''
    Create a pie chart with optional customization.

    :param data: list of values for the pie chart
    :param labels: list of labels corresponding to the data
    :param title: string, the title of the chart
    :param save_path: string, path to save the chart (optional)
    :param explode: list of floats, determines the fraction to offset each slice (optional)
    :param colors: list of colors for the slices (optional)
    :return: plt object or saved file path if save_path is provided
    '''

    # Set figure size
    plt.figure(figsize=(8, 8))

    # Create pie chart
    wedges, texts, autotexts = plt.pie(
        data,
        labels=labels,
        explode=explode,
        colors=colors,
        autopct='%1.1f%%',  # Show percentage with 1 decimal
        startangle=90,  # Start from 90 degrees for better alignment
        textprops={'fontsize': 12}  # Set font size for labels
    )

    # Add a title
    plt.title(title, fontsize=16, fontweight='bold', pad=20)

    # Beautify the chart (e.g., shadow effect)
    plt.axis('equal')  # Ensure pie chart is circular

    if save_path:
        # Save the plot
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        return save_path
    else:
        # Show the plot
        return plt

def scatter_plot(x, y, title, x_label=None, y_label=None, save_path=None, color='blue', size=100, alpha=0.7):
    '''
    Create a scatter plot with optional customization.

    :param x: x-axis data
    :param y: y-axis data
    :param title: string, the title of the chart
    :param x_label: string, label for the x-axis (optional)
    :param y_label: string, label for the y-axis (optional)
    :param save_path: string, path to save the chart (optional)
    :param color: string, color of the scatter points (default is blue)
    :param size: int, size of the scatter points (default is 100)
    :param alpha: float, transparency of the scatter points (default is 0.7)
    :return: plt object or saved file path if save_path is provided
    '''

    # Set seaborn style
    sns.set_style("whitegrid")

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, c=color, s=size, alpha=alpha, edgecolor='k', linewidth=0.5)

    # Titles and labels
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    if x_label:
        plt.xlabel(x_label, fontsize=14, labelpad=10)
    if y_label:
        plt.ylabel(y_label, fontsize=14, labelpad=10)

    # Beautify the axes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Add a grid
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        # Save the plot
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        return save_path
    else:
        # Show the plot
        return plt
    
def plot_event_by_year(results2):
    # 将 'reference_period_start' 转换为 datetime 类型，并提取年份
    results2['reference_period_start'] = pd.to_datetime(results2['reference_period_start'])
    results2['year'] = results2['reference_period_start'].dt.year
    
    # 按年份和地区分组，计算每组的事件总和
    aggregated_data = results2.groupby(['year', 'admin1_name'])['events'].sum().reset_index()

    # 创建绘图
    plt.figure(figsize=(12, 8))

    # 使用 seaborn 绘制条形图
    sns.barplot(data=aggregated_data, x='year', y='events', hue='admin1_name', palette='Set2')

    # 添加标题和标签
    plt.title('Total Events by Year and Region', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Total Events', fontsize=14)
    
    # 添加图例
    plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 显示图形
    plt.tight_layout()
    plt.show()

def bar_chart(data, title, x_label=None, y_label=None, save_path=None, color='skyblue', alpha=0.7, stacked=False):
    '''
    Create a bar chart with optional customization.

    :param data: DataFrame containing the data to be plotted
    :param title: string, the title of the chart
    :param x_label: string, label for the x-axis (optional)
    :param y_label: string, label for the y-axis (optional)
    :param save_path: string, path to save the chart (optional)
    :param color: string, color of the bars (default is skyblue)
    :param alpha: float, transparency of the bars (default is 0.7)
    :param stacked: bool, whether to make the bars stacked (default is False)
    :return: plt object or saved file path if save_path is provided
    '''
    # Set seaborn style
    sns.set_style("whitegrid")

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the data as a bar chart
    ax = data.plot(kind='bar', stacked=stacked, alpha=alpha)

    # Titles and labels
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    if x_label:
        plt.xlabel(x_label, fontsize=14, labelpad=10)
    if y_label:
        plt.ylabel(y_label, fontsize=14, labelpad=10)

    # Beautify the axes
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12)

    # Add a grid
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        # Save the plot
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        return save_path
    else:
        # Show the plot
        return plt