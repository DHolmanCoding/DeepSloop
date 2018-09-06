"""
This script is designed take in multiple Deep Sloop models (.hdf5 files) trained, validated, and selected based on
minimum validation loss and produce an ensemble model that capitalizes on their strengths
"""

#
# Dependencies
#

from keras.models import load_model

#
# Definitions
#

def gen_ensemble(models, model_input):
        # collect outputs of models in a list
        yModels = [model(model_input) for model in models]
        # averaging outputs
        yAvg = layers.average(yModels)
        # build model from same input and avg output
        modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')

        return modelEns
#
# Main
#

models = []
num_models = len(models)

for i in range(numOfModels):
    modelTemp = load_model(path2modelx)  # load model
    modelTemp.name = "aUniqueModelName"  # change name to be unique
    models.append(modelTemp)

model_input = Input(shape=models[0].input_shape[1:]) # c*h*w
modelEns = gen_ensemble(models, model_input)
model.summary()

modelEns=load_model(path2ensModel)
modelEns.summary()
y=modelEns.predict(x)