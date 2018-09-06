"""
This script is designed take in multiple Deep Sloop models (.hdf5 files) trained, validated, and selected based on
minimum validation loss and produce an ensemble model that capitalizes on their strengths.
"""

#
# Dependencies
#

from keras.models import load_model

import DeepSloop_Utils as DSU

import os


#
# Definitions
#

def model_loader(model_files,
                 model_names):
    """
    This subroutine will take in a list of model .hdf5 file names from your Model_Results, as well as a list of short,
    human readable names for the models as output a list of models.

    Arguments:
    model_files -- A list of .hdf5 file names corresponding to models you wish to ensemble model.
    model_names --  A list of strings representing easily recognizable names for the models, organized in the same
                    order as the model_files list.
    """
    models = []

    # Open each model file and store the actual model with an easily recognizable name
    for i in range(len(model_files)):
        modelTemp = load_model(os.path.join(model_results_path, model_files[i]))
        modelTemp.name = model_names[i]  # change name to be unique
        models.append(modelTemp)

    return models

def gen_ensemble(ensemble_name,
                 models,
                 repository_path=r"../Data/Filtered_90_70_Sloops_Loop_3_22_Seg_20_150_FLOL_split_datasets_0",
                 model_results_path = "../Model_Results",
                 data_type='Va'):
    """
    This routine will take in one or multiple models, and save an ensemble model that averages their predictions on the
    validation data.


    Arguments:
    repository_path -- A string representing the  path to the directory of the data repository you wish to use
    model_results_path -- A string representing the path to the directory containing the weights you wish to access
    data_type -- A string indicating the type of data you would like to predict on:
                    'Va' -- validation set
                    'Te' -- test set
    ensemble_name -- A string representing the desired name of your ensemble model
    models -- If you are using a single model, this will be a single model imported from
    """
    fasta_file_name = repository_path.split('Data/')[-1].split('split_datasets_')[0]
    Val_fasta = os.path.join(repository_path, '{}Va_NS_RNS.fasta'.format(fasta_file_name))

    X_Val_tens, y_Val_tens, num_Val_sloops, max_Val_sloop_len = DSU.fasta_to_tens(Val_fasta, max_sloop_len)

    # Collect outputs of models in a list
    model_yhat_list = [model(X_Val_tens) for model in models]
    # Average outputs
    yAvg = layers.average(model_yhat_list)

    # Build model from Validaiton data and avg output
    ensemble_model = Model(inputs=X_Val_tens, outputs=yAvg, name=ensemble_name)

    ensemble_file_name = ensemble_name + ".hdf5"
    ensemble_model.save(os.path.join(model_results_path, ensemble_file_name))


#
# Main
#

model_results_path = "../Model_Results"

model_files = ["run_2018-09-06_08-29-46_res_blstm128_blstm128_de64_de16_de1_10_0.144_0.945_0.949.hdf5",
               "run_2018-09-05_18-03-33_res_blstm128_blstm128_do5_de1_12_0.145_0.948_0.950.hdf5",
               "run_2018-09-05_19-50-17_res_conv_blstm128_blstm128_do5_de1_5_0.157_0.938_0.943.hdf5"
               ]

model_names = ["Bi_NoDo",
               "Bi_Do",
               "Hybrid"
               ]

models = model_loader(model_files,
                      model_names)


gen_ensemble(ensemble_name="Bi_BiDo_Hyb_1",
             models=models)

# Ensemble_Model=load_model(path2ensModel)
# Ensemble_Model.summary()
# y=Ensemble_Model.predict(x)