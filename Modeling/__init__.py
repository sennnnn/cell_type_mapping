from Modeling.autoencoder import AutoEncoder


def construct_model(params):
    if params["MODEL_SELECTION"] == "autoencoder":
        model = AutoEncoder(params["IN_CHANNELS"], params["OUT_CHANNELS"], params["HIDDEN_CHANNELS"])
        if params["CUDA"]:
            model = model.cuda()
        return model
        