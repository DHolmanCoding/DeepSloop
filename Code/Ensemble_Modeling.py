"""
This script is designed take in multiple Deep Sloop models (.hdf5 files) trained, validated, and selected based on
minimum validation loss and produce an ensemble model that capitalizes on their strengths.
"""

#
# Dependencies
#

from keras.models import load_model

from keras.layers import Input

import keras

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
                 model_results_path = "../Model_Results"):
    """
    This routine will take in one or multiple models, and save an ensemble model that averages their predictions on the
    validation data.

    Arguments:
    ensemble_name -- A string representing the desired name of your ensemble model
    models -- If you are using a single model, this will be a single model imported from
    model_results_path -- A string representing the path to the directory containing the weights you wish to access
    """
    # Instantiate a symbollic tensor for input into the ensemble model
    X = Input(shape=models[0].input_shape[1:])

    # Collect outputs of models in a list
    model_yhat_list = [model(X) for model in models]

    # Average outputs
    y_avg = keras.layers.average(model_yhat_list)

    # Build model from Validaiton data and avg output
    ensemble_model = keras.models.Model(inputs=X, outputs=y_avg, name=ensemble_name)

    # Save your model
    ensemble_file_name = ensemble_name + ".hdf5"
    ensemble_path = os.path.join(model_results_path, ensemble_file_name)
    ensemble_model.save(ensemble_path)
    print("Your ensemble model has been saved to {}".format(ensemble_path))

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
