# Choose categories.py
# This script is used to choose the categories from the COCO dataset for model training.

### Selection criteria:
# 1. The category should have at least 900 images in the training set.
# 2. The category should have at least 80 images in the validation set.
## ->> Due to the validation set limitation, I set the threshold of validation set to 80 so many animals can be included.
# 3. The category should not be 'person', since the person category has too many images and too general.
# 4. The categories should be diverse, and consists of different supercategories.
# 5. A supercategory should have at least 2 categories.
##  ->> ideally, every 10 categories shall be made from at least 2-3 different supercategories.

### Author's note:
# Choosing randomly

import pandas as pd
import random

def pick_one_random(category_names):
    return random.choice(category_names)

def make_filtered_df(df):
    """
    This function removes the categories that do not satisfy the following criteria:
    1. The category should have at least 900 images in the training set.
    2. The category should have at least 90 images in the validation set.
    3. The category should not be 'person', since the person category has too many images and too general.
    """
    df_no_person = df[df['Category Name'] != 'person']
    df_filtered = df_no_person[(df_no_person['Number of Training Images'] >= 900) & (df_no_person['Number of Validation Images'] >= 85)]
    return df_filtered

def choose_categories(num_supercategories, num_categories, save_to_csv=False, csv_name='chosen_categories.csv'):

    df = pd.read_csv('dataset_info.csv')
    df_filtered = make_filtered_df(df)
    category_names = df_filtered['Category Name'].tolist()
    supercategory_names = list(set(df_filtered['Supercategory Name'].tolist()))
    available_supercategories = [supercategory for supercategory in supercategory_names if len(df_filtered[df_filtered['Supercategory Name'] == supercategory]) >= 2]

    assert len(available_supercategories) >= num_supercategories, \
        f"There are only {len(available_supercategories)} supercategories that satisfy the selection criteria. Please reduce the number of supercategories or categories."


    # choose 10 categories from 3 available supercategories, make sure each supercategory has at least 2 categories.
    def sample():
        chosen_supercategories = random.sample(available_supercategories, num_supercategories)
        available_categories = [category for category in category_names if df_filtered[df_filtered['Category Name'] == category]['Supercategory Name'].values[0] in chosen_supercategories]
        assert len(available_categories) >= num_categories, \
            f"There are only {len(available_categories)} categories that satisfy the selection criteria. Please reduce the number of categories."
        
        chosen_categories = random.sample(available_categories, num_categories)

        num_supercategories_after_sampling = []
        for category in chosen_categories:
            num_supercategories_after_sampling.append(df_filtered[df_filtered['Category Name'] == category]['Supercategory Name'].value_counts()[0])

        if len(set(num_supercategories_after_sampling)) != 1:
            chosen_categories = sample()

        return chosen_categories
    
    chosen_categories = sample()

    print("Chosen categories:")
    for category in chosen_categories:
        print(f"{df[df['Category Name'] == category]['Supercategory Name'].values[0]} - {category} - {df[df['Category Name'] == category]['Number of Training Images'].values[0]} training images, {df[df['Category Name'] == category]['Number of Validation Images'].values[0]} validation images")

    if save_to_csv:
        df_chosen_categories = pd.DataFrame({
            'Supercategory Name': [df_filtered[df_filtered['Category Name'] == category]['Supercategory Name'].values[0] for category in chosen_categories],
            'Category Name': chosen_categories,
            'Number of Training Images': [df_filtered[df_filtered['Category Name'] == category]['Number of Training Images'].values[0] for category in chosen_categories],
            'Number of Validation Images': [df_filtered[df_filtered['Category Name'] == category]['Number of Validation Images'].values[0] for category in chosen_categories]
        })
        df_chosen_categories.to_csv(csv_name, index=False)

    return chosen_categories



def choose_categories_manually(*category_names, save_to_csv=False, csv_name='chosen_categories.csv'):
    df = pd.read_csv('dataset_info.csv')
    chosen_categories = []
    for category in category_names:
        if category not in chosen_categories:
            chosen_categories.append(category)

    print("Chosen categories:")
    for category in chosen_categories:
        print(f"{df[df['Category Name'] == category]['Supercategory Name'].values[0]} - {category} - {df[df['Category Name'] == category]['Number of Training Images'].values[0]} training images, {df[df['Category Name'] == category]['Number of Validation Images'].values[0]} validation images")

    if save_to_csv:
        df_chosen_categories = pd.DataFrame({
            'Supercategory Name': [df[df['Category Name'] == category]['Supercategory Name'].values[0] for category in chosen_categories],
            'Category Name': chosen_categories,
            'Number of Training Images': [df[df['Category Name'] == category]['Number of Training Images'].values[0] for category in chosen_categories],
            'Number of Validation Images': [df[df['Category Name'] == category]['Number of Validation Images'].values[0] for category in chosen_categories]
        })
        df_chosen_categories.to_csv(csv_name, index=False)

    return chosen_categories

if __name__ == "__main__":
    num_supercategories = 6
    num_categories = 20
    chosen_categories = choose_categories(num_supercategories, num_categories, save_to_csv=True, csv_name=f'chosen_categories_{num_supercategories}_{num_categories}_v3.csv')

    # df_no_person = df[df['Category Name'] != 'person']
    # print(df_no_person.head())
    # print(df_no_person.info())
    # print(df_no_person['Number of Training Images'].mean())
    # print(df_no_person['Number of Validation Images'].mean())

    # chosen_10_categories = ['dog', 'cat', 'bird', 'car', 'motorcycle', 'horse', 'sheep', 'cow', 'elephant', 'bear']
    # chosen_10_categories = choose_categories_manually(*chosen_10_categories, save_to_txt=True, txt_name='chosen_categories_10.txt')



