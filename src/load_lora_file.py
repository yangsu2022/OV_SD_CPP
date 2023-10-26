from safetensors.numpy import load_file
import numpy as np

def load_safetensors(lora_file):
    return load_file(lora_file)

def get_lora_weights(state_dict, alpha):
    visited = []
    lora_dict = {}
    lora_dict_list = []
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    #flag = 0
    for key in state_dict:
        if ".alpha" in key or key in visited:
            continue
        if "text" in key:
            layer_infos = key.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split(".")[0]
            lora_dict = dict(name=layer_infos)
            lora_dict.update(type="text_encoder")
        else:
            layer_infos = key.split(LORA_PREFIX_UNET + "_")[1].split('.')[0]
            lora_dict = dict(name=layer_infos)
            lora_dict.update(type="unet")
        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

            # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).astype(np.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).astype(np.float32)
            lora_weights = np.expand_dims(np.expand_dims((alpha * np.matmul(weight_up, weight_down)),axis=2),axis=3)
            lora_dict.update(value=lora_weights)
        else:
            weight_up = state_dict[pair_keys[0]].astype(np.float32)
            weight_down = state_dict[pair_keys[1]].astype(np.float32)
            lora_weights = alpha * np.matmul(weight_up, weight_down)
            lora_dict.update(value=lora_weights)
        lora_dict_list.append(lora_dict)
        #print("{}:Shape {}".format(lora_dict["name"],lora_dict["value"].shape))
        # update visited list
        for item in pair_keys:
            visited.append(item)
    return lora_dict_list

def get_encoder_layers(lora_dict_list):
    encoder_layers = []
    for k in lora_dict_list:
        if k["type"] == "text_encoder":
            encoder_layers.append(k["name"])
    return encoder_layers

def get_encoder_weights(lora_dict_list):
    encoder_weights = []
    for k in lora_dict_list:
        if k["type"] == "text_encoder":
            encoder_weights.append(k["value"])
    return encoder_weights

def get_unet_layers(lora_dict_list):
    unet_layers = []
    for k in lora_dict_list:
        if k["type"] == "unet":
            unet_layers.append(k["name"])
    return unet_layers

def get_unet_weights(lora_dict_list):
    unet_weights = []
    for k in lora_dict_list:
        if k["type"] == "unet":
            unet_weights.append(k["value"])
    return unet_weights

# DEBUG: 
# print("begin to load safetensor")
# py_lora_tensor = load_safetensors("./models/soulcard.safetensors")
# print("finished loaded safetensor and begin to get weights")
# py_lora_weights = get_lora_weights(py_lora_tensor, 0.75)