import pandas as pd

def get_accuracy(df, mode):
    """
    Retrieves the accuracy of a result.Epoch

    :param df: Result as Pandas DataFrame
    :param mode: Whether gets the train or test accuracy. Fill in "train" or "test".
    :returns accuracy: an array of accuracies (either train or test) at each epoch
    """

    # assert mode == "train" or mode == "test", "invalid mode name, should be either train or test"
    
    if mode == 'train':
        accuracy = df[df['Mode'] == 'train']['Accuracy']

    elif mode == 'test':
        accuracy = df[df['Mode'] == 'test']['Accuracy']

    if accuracy.iloc[0] > 1:
        accuracy = accuracy / 100
    return accuracy

def get_crossentropy(df, mode):
    """
    Retrieves the cross entropy loss of a result.

    :param df: Result as Pandas DataFrame
    :param mode: Whether gets the train or test cross entropy. Fill in "train" or "test".
    :returns accuracy: an array of cross entropy (either train or test) at each epoch
    """

    # assert mode == "train" or mode == "test", "invalid mode name, should be either train or test"
    
    if mode == 'train':
        crossentropy = df[df['Mode'] == 'train']['Cross Entropy']

    elif mode == 'test':
        crossentropy = df[df['Mode'] == 'test']['Cross Entropy']

    return crossentropy

def generate_unimodal_name(df_name):
    encoder = df_name.split("_")[1]
    lr = df_name.split("-")[2]
    num_classes = df_name.split("-")[3]
    num_prototypes = df_name.split("-")[-1].replace(".csv", "")

    model_name = f"Unimodal PBN {encoder} {lr} {num_classes} {num_prototypes}"

    return model_name 

def generate_subplot(ax, result_filename, value_type="Accuracy",row_nr=0,col_nr=0, onerow=False):

    ax_modelname = generate_unimodal_name(result_filename)
    ax_data = pd.read_csv(result_filename)

    if value_type == "Accuracy":
        ax_train_accu = get_accuracy(ax_data, 'train')
        ax_test_accu = get_accuracy(ax_data, 'test')
    elif value_type == "Cross Entropy":
        ax_train_accu = get_crossentropy(ax_data, 'train')
        ax_test_accu = get_crossentropy(ax_data, 'test')
    ax_num_epochs = list(range(1, len(ax_train_accu)+1))

    if onerow:
        ax[col_nr].plot(ax_num_epochs,ax_train_accu,label='Train')
        ax[col_nr].plot(ax_num_epochs, ax_test_accu, label='Test')
        ax[col_nr].set_xlabel('Epoch')
        ax[col_nr].set_ylabel(value_type)
        ax[col_nr].set_title(ax_modelname)
        ax[col_nr].set_xlim(0,70)
        if value_type == "Accuracy":
            ax[col_nr].set_ylim(0,1)
        ax[col_nr].legend()
    else:
        ax[row_nr,col_nr].plot(ax_num_epochs,ax_train_accu,label='Train')
        ax[row_nr,col_nr].plot(ax_num_epochs, ax_test_accu, label='Test')
        ax[row_nr,col_nr].set_xlabel('Epoch')
        ax[row_nr,col_nr].set_ylabel(value_type)
        ax[row_nr,col_nr].set_title(ax_modelname)
        ax[row_nr,col_nr].set_xlim(0,70)
        if value_type == "Accuracy":
            ax[row_nr,col_nr].set_ylim(0,1)
        ax[row_nr,col_nr].legend()
    