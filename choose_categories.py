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

def make_filtered_df(df, include_person=False, min_train_count=100, min_val_count=0):
    """
    This function removes the categories that do not satisfy the following criteria:
    1. The category should have at least 900 images in the training set.
    2. The category should have at least 90 images in the validation set.
    3. The category should not be 'person', since the person category has too many images and too general.
    """
    if not include_person:
        df_filtered = df[df['Category Name'] != 'person']
    else:
        df_filtered = df

    df_final = df_filtered[(df_filtered['Number of Training Images'] >= min_train_count) & (df_filtered['Number of Validation Images'] >= min_val_count)]
    return df_final

def choose_categories(num_supercategories, num_categories, singleLabel=False, min_train_count=100, min_val_count=0, save_to_csv=False, csv_name='chosen_categories.csv'):

    if singleLabel:
        df = pd.read_csv('dataset_infos/dataset_info_singleLabel.csv')
    else:
        df = pd.read_csv('dataset_infos/dataset_info.csv')

    df_filtered = make_filtered_df(df, include_person=False, min_train_count=min_train_count, min_val_count=min_val_count)
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
    num_supercategories1 = 7
    num_categories1 = 30

    num_supercategories2 = 6
    num_categories2 = 20

    num_supercategories3 = 3
    num_categories3 = 10

    singleLabel = True
    min_train_count = 100
    min_val_count = 0
    chosen_categories1 = choose_categories(num_supercategories1, 
                                          num_categories1, 
                                          singleLabel=singleLabel,
                                          min_train_count=min_train_count, 
                                          min_val_count=min_val_count, 
                                          save_to_csv=True, 
                                          csv_name=f'singleLabel_chosen_categories_{num_supercategories1}_{num_categories1}.csv')
    
    for i in range(3):
        chosen_categories = choose_categories(num_supercategories2, 
                                          num_categories2, 
                                          singleLabel=singleLabel,
                                          min_train_count=min_train_count, 
                                          min_val_count=min_val_count, 
                                          save_to_csv=True, 
                                          csv_name=f'singleLabel_chosen_categories_{num_supercategories2}_{num_categories2}_v{i}.csv')
        
        chosen_categories = choose_categories(num_supercategories3, 
                            num_categories3, 
                            singleLabel=singleLabel,
                            min_train_count=min_train_count, 
                            min_val_count=min_val_count, 
                            save_to_csv=True, 
                            csv_name=f'singleLabel_chosen_categories_{num_supercategories3}_{num_categories3}_v{i}.csv')

    # df_no_person = df[df['Category Name'] != 'person']
    # print(df_no_person.head())
    # print(df_no_person.info())
    # print(df_no_person['Number of Training Images'].mean())
    # print(df_no_person['Number of Validation Images'].mean())

    # chosen_10_categories = ['dog', 'cat', 'bird', 'car', 'motorcycle', 'horse', 'sheep', 'cow', 'elephant', 'bear']
    # chosen_10_categories = choose_categories_manually(*chosen_10_categories, save_to_txt=True, txt_name='chosen_categories_10.txt')



