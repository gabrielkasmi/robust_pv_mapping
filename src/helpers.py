# -*- coding: utf-8 -*-

# Libraries
import os
import torch
from PIL import Image, ImageOps
import numpy as np
import json
from torchvision import transforms
import torchvision
from torch.nn import functional as F
from skimage.util import random_noise
import cv2
from piq import ssim, brisque, psnr
from itertools import chain, combinations
import math
import itertools
import tqdm
import pywt



# computation of the confusion matrix

def compute_confusion_matrix(confusion):
    """
    computes the confusion matrix
    in the inputed confusion vector
    

    labels :
    - 0 : TP
    - 1 : TN
    - 2 : FN
    - 3 : FP
    
    returns an array
    """
    
    tp = len(np.where(confusion == 0)[0])
    tn = len(np.where(confusion == 1)[0])
    fn = len(np.where(confusion == 2)[0])
    fp = len(np.where(confusion == 3)[0])
        
    return np.array([
        [tp, fp],
        [fn, tn]
    ])



def compute_wavelet_transform(image, level = 3, wavelet = 'haar'):
    """
    computes the wavelet transform of the input image
    
    returns a (W,H,C) array where for each channel we compute 
    the associated wavelet transform
    returns the slices as well, to facilitate reconstruction
    
    remark: better to stick with haar wavelets as they do not induce 
    shape issues between arrays.
    """
    
    if not isinstance(image, np.ndarray):
        
        image = np.array(image)
        
    
    transform = np.zeros(image.shape)
    
    for i in range(image.shape[2]):
        # compute the transform for each channel
        
        x = image[:,:,i]
        coeffs = pywt.wavedec2(x, wavelet, level=level)
        arr, slices = pywt.coeffs_to_array(coeffs)
        
        transform[:,:,i] = arr
        
    return transform, slices


def perturb_and_invert(mask, slices, transform, wavelet = "haar"):
    """
    computes the perturbed wavelet transform and inverts it to return the 
    transformed image
    
    the mask's size whould match the size of the transform.
    
    returns the rgb image (as an array)
    """

    def NormalizeData(data):
        """helper to normalize in [0,1] for the plots"""
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    perturbed_image = np.zeros(transform.shape)
        
    for i in range(perturbed_image.shape[2]):
        
        # apply a channel wise perturbation
        perturbation = transform[:,:,i] * mask
        
        # using the slices passed as input and the perturbed
        # transform, compute the inverse for this channel
        
        # compute the coeffs
        coeffs = pywt.array_to_coeffs(perturbation, slices, output_format = "wavedec2")
        perturbed_image[:,:,i] = pywt.waverec2(coeffs, wavelet)
        
    return (NormalizeData(perturbed_image) * 255).astype(np.uint8)


def return_perturbations(image, perturbations, wavelet = "haar", levels = 3, steps = 5):
    """
    given the dicitonnary of perturbations and the image, 
    returns a dictionnary of pertubed images
    """

    steps = list(perturbations.keys())

    # initialize the bi-level images
    perturbed_images = {s : {t : None for t in steps} for s in steps}

    # compute the wavelet transform of the source image
    out, slices = compute_wavelet_transform(image, level = levels, wavelet = wavelet)

    # loop over the pertrubation combinations (steps * steps)
    for s in steps:
        for t in steps:

            mask = perturbations[s][t]
            pert = perturb_and_invert(mask, slices, out, wavelet = wavelet)

            perturbed_images[s][t] = pert

    return perturbed_images


# functions to compute the masks applied to the transform

def coord_in_and_out(arr_mask, scale = 3):
    """
    computes the coordinates in and out the mask. 
    returns the homoteth

    input : arr_mask
    output : in_coord, out_coord : two lists with 
    the coordinates 
    """

    # get the coordinates on the original mask

    out = np.zeros(arr_mask.shape)
    
    y, x = np.where(arr_mask == 255)
    coords_in = np.array(list(zip(y,x)))

    if len(coords_in) == 0: # negative case:
        # else: generate a dummy set of coordinates
        # consider a "mask" located randomly 
        size = 25 # arbitrarily set
        dimension = arr_mask.shape[0]
        center_y, center_x = np.random.uniform(dimension - size, size = 2).astype(int)
        coords_in = np.array(
            [np.array([y, x]) for y, x in itertools.product(range(center_y - size, center_y + size), range(center_x - size, center_x + size))]
            )

    for s in range(scale):
        
        offset = int(out.shape[0] / 2 ** (s+1))

        for c in coords_in:

            # diagonal
            y, x = (c / (2 **(s+1))).astype(int)

            out[y + offset, x + offset] = 255

            # horizontal
            out[y + offset, x] = 255

            # vertical
            out[y , x + offset] = 255

            if (s+1) == scale: # add the top left corner if we are at the smallest scale
                out[y,x] = 255

    y, x = np.where(out == 255)
    coords_in = list(zip(y,x))

    y,x = np.where(out == 0)
    coords_out = list(zip(y,x))

    return coords_in, coords_out


def compute_perturbations(arr_mask, steps = 5, seed = 42, level = 3):
    """
    given the array of masks and a number of steps,
    compute a sequence of perturbations both in and out
    returns a dictionnary where each key is a perturbation level
    (two level) and the value is the perturbation mask

    # first level is the perturbation outside the mask
    # second level is the perturbation inside the mask

    returns: perturbation, the dictionnary with the
    steps * steps perturbations

    """

    np.random.seed(seed)

    size = arr_mask.shape[0]

    perturbations = {s + 1 : {k + 1 : np.zeros(arr_mask.shape) for k in range(steps)} for s in range(steps)}

    coords_in, coords_out = coord_in_and_out(arr_mask, scale = level)

    for out_step in perturbations.keys():

        rand_out = np.random.uniform(size = (size,size))
        rand_out = (rand_out <= (out_step) / steps).astype(int)

        for c in coords_in: # set the perturbation to zero inside the box
            rand_out[c] = 0

        for in_step in perturbations[out_step].keys():

            rand_in = np.random.uniform(size = (size,size))
            rand_in = (rand_in <= (in_step)/steps).astype(int)

            for c in coords_out: # set the perturbation to 0 outside the box
                rand_in[c] = 0

            perturbations[out_step][in_step] = rand_in + rand_out
   
    return perturbations

def NormalizeData(data):
    """helper to normalize in [0,1] for the plots"""
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def confusion(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    if isinstance(confusion_vector, np.ndarray):
        false_positives = np.sum(confusion_vector == np.inf).item()
        true_negatives = np.sum(np.isnan(confusion_vector)).item()
        true_positives = np.sum(confusion_vector == 1.).item()
        false_negatives = np.sum(confusion_vector == 0.).item()

    else:

        true_positives = torch.sum(confusion_vector == 1).item()
        false_positives = torch.sum(confusion_vector == float('inf')).item()
        true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives = torch.sum(confusion_vector == 0).item()
    

    return true_positives, false_positives, true_negatives, false_negatives


def generate_circular_masks(b, shape):
    """
    computes a circular mask of bandwidth b 
    with size shape (tuple)
    """
    
    d = int(shape[0]) # size of the input image
    center = int(shape[0] / 2), int(shape[1] / 2)
        
    Ms = {} # dictionnary, returned at the end, with the masks
    
    k = 0
    
    # distance in the diagonal between the center and the corner:
    d_max = np.sqrt(2) * np.sqrt(center[0] ** 2 + center[1] ** 2)
    
    d_current = np.sqrt((k * b) ** 2 + (k * b) ** 2)
    
    while d_current <= d_max:
                
        M = np.zeros((d,d)) # create a mask of zeros
        # define the up radius and down radius
        
        r_down = k * b
        r_up = (k + 1) * b
                
        # main loop that sets to 0 the coordinates
        # whose distance is within the radii

        for x, y in itertools.product(range(center[0]), range(center[1])):

            x_centered = x - center[0]
            y_centered = y - center[1]

            distance = np.sqrt(x ** 2 + y** 2) 

            if r_down <= distance <= r_up:
                

                M[x_centered,y_centered] = 1
                M[- x_centered,y_centered] = 1
                M[- x_centered,-y_centered] = 1
                M[x_centered,-y_centered] = 1
                
        Ms[k] = M
        
        # increment k and update the current distance
        k += 1
        d_current = np.sqrt((k * b) ** 2 + (k * b) ** 2)

    return Ms


def compute_pr_curve(results):
    """computes the PR curve as a function of the keys
    passed as input, typically the detection thresholds
    """

    thresholds = list(results.keys())
    precision, recall = [], []

    for threshold in thresholds:
        tp, fp, fn = results[threshold]['true_positives'], results[threshold]['false_positives'], results[threshold]['false_negatives']
        precision.append(np.divide(tp,(tp + fp)))
        recall.append(np.divide(tp,(tp + fn)))

    return precision, recall

def return_f1(precision, recall):
    f1 = 2 * (np.array(precision) * np.array(recall)) / (np.array(precision) + np.array(recall)) 

    return 2 * (np.array(precision) * np.array(recall)) / (np.array(precision) + np.array(recall))

def confusion_samples(prediction, truth, names):
    """ Computes the confusion matrix and returns a list with
    the TP/FP/TN/FN names
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)
    
    # convert as an aray
    confusion_vector= confusion_vector.cpu().detach().numpy()
    
    true_positives = np.where(confusion_vector == 1)[0]
    false_positives = np.where(confusion_vector == float('inf'))[0]
    true_negatives = np.where(np.isnan(confusion_vector))[0]
    false_negatives = np.where(confusion_vector == 0)[0]
        
    def check_if_empty(index_items, items):
        """returns an empty list if index_items is empty"""
        return [items[i] for i in index_items]   
    
    return check_if_empty(true_positives, names), check_if_empty(false_positives, names), check_if_empty(true_negatives, names), check_if_empty(false_negatives, names)   

def confusion_samples_with_probs(prediction, truth, names, probabilities):
    """ Computes the confusion matrix and returns a list with
    the TP/FP/TN/FN names
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    # convert as an aray
    confusion_vector = confusion_vector.cpu().detach().numpy()
    probabilities = probabilities.cpu().detach().numpy()

    true_positives = np.where(confusion_vector == 1)[0]
    false_positives = np.where(confusion_vector == float('inf'))[0]
    true_negatives = np.where(np.isnan(confusion_vector))[0]
    false_negatives = np.where(confusion_vector == 0)[0]
    
    def check_if_empty(index_items, items_1, items_2):
        """returns an empty list if index_items is empty"""
        return [items_1[i] for i in index_items], [items_2[i] for i in index_items]   

    return check_if_empty(true_positives, names, probabilities), check_if_empty(false_positives, names, probabilities), check_if_empty(true_negatives, names, probabilities), check_if_empty(false_negatives, names, probabilities)   

# Helper functions
"""
A helper that computes the class activation maps. 

Drawn from https://www.kaggle.com/bonhart/inceptionv3-tta-grad-cam-pytorch for inception v3

"""

class SaveFeatures():
    """
    A class that extracts the pretrained activations

    register_forward_hook returns the input and output of a given layer
    during the foward pass.
    """

    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy

    def remove(self):
        self.hook.remove()


def compute_class_activation_map(feature_conv, weight_fc, class_idx):
    """
    core computation of the class activation map

    args : 

    feature_conv : the last convolutional layer (before the GAP step).
    Corresponds to the matrices A1, ..., Ak in Zhang et al (2016)

    weights_fc : the weights of the fully connected layer.
    Corresponds to the weights w1, ..., wk in Zhang et al (2016)

    class_idx : the class id, i.e; the class_id th neuron to activate


    returns : 
    cam_image : the computed class activation map. It's a 2d map that has the size of the 
    last convolutional layer.

    """
    _, nc, h, w = feature_conv.shape # extract the shape of the convolutional layer
    #print(nc, h, w)

    # computation of the cam
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w) # reshape as a 2d matrix

    # numerical stability
    cam = cam - np.min(cam)
    cam_image = cam / np.max(cam)

    return cam_image


def compute_cam(img, model, to_transform = True, device = 'cuda', sigma_val = None):
    """
    computes the cam for a resnet model
    """
    device = torch.device(device)

    # get the layer from the model
    final_layer = model.layer4
    model = model.to(device)
    
    # register the layer in order to retrieve its value
    # after the forward pass.
    activated_features = SaveFeatures(final_layer)

    model.eval()

    # prediction
    if to_transform : 
        if sigma_val is None:
            transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
            ])
        else:
            transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
            torchvision.transforms.GaussianBlur(np.ceil(4 * sigma_val) + 1, sigma = sigma_val),
            ])

        output = model.forward(transforms(img).unsqueeze(0).to(device))
    else:
        output = model.forward(img.unqueeze(0).to(device))
        
    activated_features.remove()
    probs = F.softmax(output, dim = 1) # the model returns the unnormalized probs. Softmax it to get probs   


    # get the values of the weights
    weight_softmax_params = model.fc.state_dict()['0.weight']
    weight_softmax = np.squeeze(weight_softmax_params.cpu().detach().numpy())
    class_idx = np.argmax(probs.detach().cpu().numpy().squeeze())

    return compute_class_activation_map(activated_features.features(), weight_softmax, 1)

def confusion_samples_embeddings(prediction, truth, names, embeddings):
    """ Computes the confusion matrix and returns a list with
    the TP/FP/TN/FN names
    """

    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)
    
    # convert as an aray
    confusion_vector= confusion_vector.cpu().detach().numpy()
    
    true_positives = np.where(confusion_vector == 1)[0]
    false_positives = np.where(confusion_vector == float('inf'))[0]
    true_negatives = np.where(np.isnan(confusion_vector))[0]
    false_negatives = np.where(confusion_vector == 0)[0]
    
    tp_embeddings = embeddings[true_positives,:]
    fp_embeddings = embeddings[false_positives,:]
    tn_embeddings = embeddings[true_negatives,:]
    fn_embeddings = embeddings[false_negatives,:]

    
    def check_if_empty(index_items, items):
        """returns an empty list if index_items is empty"""
        return [items[i] for i in index_items]   
    
    return check_if_empty(true_positives, names), check_if_empty(false_positives, names), check_if_empty(true_negatives, names), check_if_empty(false_negatives, names), tp_embeddings, fp_embeddings, tn_embeddings, fn_embeddings   

def reshape_as_array(results, img_list):
    """
    reshapes the results from the dictionnary `results`
    as an array. The first column of the array encodes the status
    of the image :
        0 : true positive
        1 : false positive
        2 : true negative
        3 : false negative
    
    returns embeddings : a matrix where each row encodes the embedding of the corersponding
    sample.
    """
    
    # split the keys in two parts : one matches the name of the image and the 
    # index in the whole dataset
    # the other matches the index of the image in the batch and its embedding 
    keys_name_batch_index = list(results.keys())[::2]
    keys_batch_index_embedding =list(results.keys())[1::2]
        
    # instantiate the empty array
    embeddings = np.empty((len(img_list), 2048 + 1))
    
    # dictionnary that will code each case as a number
    category = {keys_name_batch_index[i]: i for i in range(len(keys_name_batch_index))}
    
    
    # initialize a list of unsorted embeddings.
    # this list contains tuples where the first element is the image name
    # and the second element is the embedding, so a (2048,) np.array.
    # the third item is the encoding (0, 1, 2, 3) of the image; 
    results_unsorted = []
    
    # loop per case (true positive, false positive, etc)
    for key_name, key_embedding in zip(keys_name_batch_index, keys_batch_index_embedding):
        # for each case, extract in each batch the name of the image and its embedding
        for name_index, index_embedding in zip(results[key_name], results[key_embedding]):
            for i in range(len(index_embedding)): # both lists have the same length (by construction)
                results_unsorted.append((name_index[i], index_embedding[i], category[key_name]))
    
    # now we sort each image.
    for i, img in enumerate(img_list):
        for result in results_unsorted:
            if img == result[0]:
                embeddings[i,0] = result[2] # add the encoding in the first column
                embeddings[i,1:] = result[1] # add the 2048-dimensional encoding.
    
    return embeddings

def confusion_matrix(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    cm = prediction / truth
    
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)
    
    true_positives = torch.tensor([torch.sum(cm[i] == 1).item() for i in range(cm.shape[0])])
    false_positives = torch.tensor([torch.sum(cm[i] == float('inf')).item() for i in range(cm.shape[0])])
    true_negatives = torch.tensor([torch.sum(torch.isnan(cm[i])).item() for i in range(cm.shape[0])])
    false_negatives = torch.tensor([torch.sum(cm[i] == 0).item() for i in range(cm.shape[0])])
    
    return true_positives, false_positives, true_negatives, false_negatives

# save the outputs
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # 👇️ alternatively use str()
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def convert_to_matrices(accuracy, uncertainty, sigmas_n, sigmas_b):
    """
    converts the outputs 
    """

    n_cols = len(sigmas_b)
    n_rows = len(sigmas_n)


    res_matrix = np.zeros((n_rows, n_cols))
    res_uncertainty = np.zeros((n_rows, n_cols))

    for i, sigma_b in enumerate(sigmas_b):
        for j, sigma_n in enumerate(sigmas_n):

            res_matrix[n_rows - j -1,i] = accuracy[sigma_b][sigma_n]
            res_uncertainty[n_rows - j - 1,i] = uncertainty[sigma_b][sigma_n]
            
    return res_matrix, res_uncertainty

def generate_noisy_image(img, sigma_n = 0, seed = 42, sigma_b = None):
    """
    generates a blurred + noisy image if sigma is not None
    takes as input a PIL image, returns a PIL image
    """

    if sigma_b is None: # no blurring applied

        img = random_noise(np.array(img), mode='gaussian', mean=0, var=sigma_n, clip = True, seed = seed)
    else:

        # case of eps
        # blur the image
        kernel_size = np.ceil(4 * sigma_b) + 1

        if kernel_size % 2 == 0:
            kernel_size -= 1

        blurred_image = torchvision.transforms.functional.gaussian_blur(img, int(kernel_size), sigma = sigma_b)
        # add noise
        img = random_noise(np.array(blurred_image), mode='gaussian', mean=0, var=sigma_n, clip = True, seed = seed)

    return Image.fromarray((NormalizeData(img) * 255).astype(np.uint8))


def open_image(image_path, image_name, size = None):
    """
    opens and crops an image if necessary
    """

    if size is None:
        return Image.open(os.path.join(image_path, image_name)).convert('RGB')

    else:
        # open the image as a PIL image
        img = Image.open(os.path.join(image_path, image_name)).convert('RGB')
        
        # resize the image
        max_x, max_y = img.size[0], img.size[1] # get the size of the image
        x_span, y_span = max(0,max_x - size), max(0,max_y - size) # coordinates of the anchor box
        x, y = int(x_span / 2), int(y_span / 2)
        final_image = torchvision.transforms.functional.crop(img, y, x, size, size)

        return final_image

def resize(img, size = None):
    if size is None: 
        return img
    else:
        # resize the image
        max_x, max_y = img.size[0], img.size[1] # get the size of the image
        x_span, y_span = max(0,max_x - size), max(0,max_y - size) # coordinates of the anchor box
        x, y = int(x_span / 2), int(y_span / 2)
        final_image = torchvision.transforms.functional.crop(img, y, x, size, size)

        return final_image

def compute_spectrum(image, grayscale = True):
    """
    computes the image spectrum of the imputed PIL image, given sigma_n and sigma_b
    image : either a PIL image or an array of a channel
    """

    if grayscale:
        img = ImageOps.grayscale(image) 
        array_img = np.array(img)

    else: 
        array_img = image

    img_float32 = np.float32(array_img) 
    dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])

    return magnitude_spectrum

def compute_spectrum_alterations(image_name, parameters):
    """
    computes the alterations to the spectrum for an image induced
    by the alterations 

    j : opt that indicates the image number to pick
    """

    # directories
    # source_dir : where the altered images are located
    # source_images_dir : where the unaltered test images are located.
    source_dir = parameters["source_dir"]
    source_images_dir = parameters["source_images_dir"] 
    size = parameters["size"] # the size of the image, necessary if an image is created from the source folder
    index = parameters['index'] # the index to pick
    source_b, source_n = parameters['source_parameters']
    target_b, target_n = parameters['target_parameters']

    # open the folder containing the images
    # the source folder is by construction the folder with 0.00 and 0.000
    source_folder = '{}-{:0.2f}-{:0.3f}'.format(image_name, source_b, source_b)
    target_folder = '{}-{:0.2f}-{:0.3f}'.format(image_name, target_b, target_n)
    source = os.path.join(source_dir, source_folder)
    target = os.path.join(source_dir, target_folder)

    # if the folder exists, pick one image
    if os.path.exists(source):
        source_img = Image.open(os.path.join(source, '{}.png'.format(index))).convert('RGB')
    else:
        # create an image
        source_b = None if source_b == 0 else source_b 
        img = Image.open(os.path.join(source_images_dir, '{}.png'.format(image_name)))
        source_img = generate_noisy_image(img, seed = 42, sigma_b = source_b, sigma_n = source_n)

    source_image = resize(source_img, size = size)
    
    if os.path.exists(target):
        target_img = Image.open(os.path.join(target, '{}.png'.format(index))).convert('RGB')

    else:
        target_b = None if target_b == 0 else target_b 
        img = Image.open(os.path.join(source_images_dir, '{}.png'.format(image_name)))
        target_img = generate_noisy_image(img, seed = 42, sigma_n = target_n, sigma_b = target_b)

    target_image = resize(target_img, size = size)        
       
    # compute the spectra
    source_spectrum = compute_spectrum(source_image)
    target_spectrum = compute_spectrum(target_image)

    # compute the difference between the two spectra
    difference_spectrum = np.abs(NormalizeData(target_spectrum) - NormalizeData(source_spectrum))

    return source_spectrum, target_spectrum, difference_spectrum

def compute_ssim(image, reference):
    """
    computes the sructural similarity index between a (list) of images
    and a reference

    inputs: 

    reference : a PIL image or a list of images
    image : a list of images or a PIL image.


    """

    transform = transforms.Compose([transforms.PILToTensor()])

    if isinstance(image, Image.Image):
        # case of a single image
        # convert the reference and image as tensors

        images = transform(image).unsqueeze(0)

    if isinstance(reference, Image.Image):
        references = transform(reference).unsqueeze(0)

    if isinstance(image, list):
        # it should be a list of PIL images
        images = torch.stack([transform(img) for img in image])

    if isinstance(reference, list):
        references = torch.stack([transform(img) for img in reference])


    return ssim(images, references, data_range = 255).item()


def compute_brisque(image):
    """
    this time, only one image 
    """

    transform = transforms.Compose([transforms.PILToTensor()])

    return brisque(transform(image).unsqueeze(0), data_range = 255).item()



def compute_psnr(image, reference):
    """
    computes the sructural similarity index between a (list) of images
    and a reference

    inputs: 

    reference : a PIL image or a list of images
    image : a list of images or a PIL image.


    """

    transform = transforms.Compose([transforms.PILToTensor()])

    if isinstance(image, Image.Image):
        # case of a single image
        # convert the reference and image as tensors

        images = transform(image).unsqueeze(0)

    if isinstance(reference, Image.Image):
        references = transform(reference).unsqueeze(0)

    if isinstance(image, list):
        # it should be a list of PIL images
        images = torch.stack([transform(img) for img in image])

    if isinstance(reference, list):
        references = torch.stack([transform(img) for img in reference])


    return psnr(images, references, data_range = 255).item()


def compute_quantitative_quality(image, reference = None, metric = "ssim"):
    """
    computes the quality of an image, given a reference (opt)

    metrics include :

    full reference metrics : a reference image is needed 
        - ssim
    
    no reference : no reference image is needed
    """

    if metric == 'ssim':

        if reference is not None:

            return compute_ssim(image, reference)
        
        else:
            print('Reference image needed.')
            raise ValueError

    if metric == 'psnr':

        if reference is not None:

            return compute_psnr(image, reference)
        
        else:
            print('Reference image needed.')
            raise ValueError

    if metric == 'brisque':
        return compute_brisque(image)


def compute_quality(image_name, parameters, metric = "ssim"):
    """
    computes the quality of an image passed as input name, 
    given parameters and a metric type

    parameters are the same as those used to compute the alteration
    in the frequency content


    in case of a full reference metric : it computes the quality 
    wrt to the source image
    """

    # directories
    # source_dir : where the altered images are located
    # source_images_dir : where the unaltered test images are located.
    source_dir = parameters["source_dir"]
    source_images_dir = parameters["source_images_dir"] 
    size = parameters["size"] # the size of the image, necessary if an image is created from the source folder
    index = parameters['index'] # the index to pick
    source_b, source_n = parameters['source_parameters']
    target_b, target_n = parameters['target_parameters']

    # open the folder containing the images
    # the source folder is by construction the folder with 0.00 and 0.000
    source_folder = '{}-{:0.2f}-{:0.3f}'.format(image_name, source_b, source_b)
    target_folder = '{}-{:0.2f}-{:0.3f}'.format(image_name, target_b, target_n)
    source = os.path.join(source_dir, source_folder)
    target = os.path.join(source_dir, target_folder)

    # if the folder exists, pick one image
    if os.path.exists(source):
        source_img = Image.open(os.path.join(source, '{}.png'.format(index))).convert('RGB')
    else:
        # create an image
        source_b = None if source_b == 0 else source_b 
        img = Image.open(os.path.join(source_images_dir, '{}.png'.format(image_name)))
        source_img = generate_noisy_image(img, seed = 42, sigma_b = source_b, sigma_n = source_n)

    source_image = resize(source_img, size = size)
    
    if os.path.exists(target):
        target_img = Image.open(os.path.join(target, '{}.png'.format(index))).convert('RGB')

    else:
        target_b = None if target_b == 0 else target_b 
        img = Image.open(os.path.join(source_images_dir, '{}.png'.format(image_name)))
        target_img = generate_noisy_image(img, seed = 42, sigma_n = target_n, sigma_b = target_b)

    target_image = resize(target_img, size = size)    


    # now we have our source and target image, we can compute the quality of the target
    # taking the source as a reference

    if metric in ["ssim", 'psnr']:
        return compute_quantitative_quality(target_image, source_image, metric = metric)

    if metric == "brisque":
        return compute_quantitative_quality(target_image, metric = metric)



    

def alter_image(source_image, mask):
    """
    given the PIL image passed as input and a perturbation mask,
    returns a perturbed PIL image 

    takes in a torch.Tensor of the image and returns a torch.Tensor
    output image
    
    perturbation is passed as the mask in input

    """


    if isinstance(source_image, Image.Image):
        array_img = np.float32(source_image)
    img_filtered = np.empty(array_img.shape)

    #if isinstance(source_image, torch.Tensor):
    #    array_img = array_img.detach().cpu().numpy().

    for i in range(3):
        dft = cv2.dft(array_img[:,:,i], flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # apply mask and inverse DFT
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

        img_filtered[:,:,i] = img_back

    img_final = Image.fromarray((NormalizeData(img_filtered) * 255).astype(np.uint8))

    return img_final


def create_image(source_image, Ms, contains_lower = True):
    """
    given the PIL image passed as input and a dictionnary of components
    create a series of altered PIL images. 

    contains_lower : whether the masks should all contain the 
    filter corresponding to the lowest frequency
    """

    altered_images = {}

    if not contains_lower:

        for component in Ms.keys():


            array_img = np.float32(source_image)
            img_filtered = np.empty(array_img.shape)


            for i in range(3):
                dft = cv2.dft(array_img[:,:,i], flags = cv2.DFT_COMPLEX_OUTPUT)
                dft_shift = np.fft.fftshift(dft)

                # apply mask and inverse DFT
                fshift = dft_shift * Ms[component]
                f_ishift = np.fft.ifftshift(fshift)
                img_back = cv2.idft(f_ishift)
                img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

                img_filtered[:,:,i] = img_back

            img_final = Image.fromarray((NormalizeData(img_filtered) * 255).astype(np.uint8))
        
            altered_images[component] = img_final

    else:

        lowest_frequency = list(Ms.keys())[0]

        for component in list(Ms.keys())[1:]:


            array_img = np.float32(source_image)
            img_filtered = np.empty(array_img.shape)


            for i in range(3):
                dft = cv2.dft(array_img[:,:,i], flags = cv2.DFT_COMPLEX_OUTPUT)
                dft_shift = np.fft.fftshift(dft)

                # apply mask and inverse DFT
                fshift = dft_shift * (Ms[lowest_frequency] + Ms[component]) 
                f_ishift = np.fft.ifftshift(fshift)
                img_back = cv2.idft(f_ishift)
                img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

                img_filtered[:,:,i] = img_back

            img_final = Image.fromarray((NormalizeData(img_filtered) * 255).astype(np.uint8))
        
            altered_images[component] = img_final


    return altered_images

def convert_evolution_probs_to_array(evolution, baseline_probs):
    """
    takes as input the dictionnary of evolutions, with the structure

    { pos : {
        img : probs,
        ...
    },
        neg : {
        img : probs
        }
    }

    baseline probs as the structure : 
    {img : prob}

    and returns an array with n_img rows and n_components + 1 columns
    where the 0th column is the baseline prob for the image
    """
    img_list = baseline_probs.keys()
    n_imgs = len(img_list)
    n_components = len(evolution['pos'][img_list[0]].keys()) + 1

    table = np.zeros((n_imgs, n_components))

    for i, img in enumerate(img_list):

        label = 'pos' if img in evolution['pos'].keys() else 'neg'

        baseline_prob = baseline_probs[img]
        probs = evolution[label][img]

        table[i,0] = baseline_prob
        table[i,1:] = probs

    return table

    
    
def powerset(F):
    """
    compute the set of all subsets of F\i
    discards the empty set
    """
    return list(chain.from_iterable(combinations(F, r) for r in range(len(F)+1)))[1:-1] # remove the last element


def composed_image(image, masks, components):
    """
    returns a filtered image with only the components passed as input
    returns a PIL image, takes as input the PIL image
    """

    # compute the aggregted mask
    aggregated_mask = np.sum(1 - masks[c] for c in components) / len(components)
    
    #### 
    array_img = np.float32(image)
    img_filtered = np.empty(array_img.shape)


    for i in range(3):
        dft = cv2.dft(array_img[:,:,i], flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # apply mask and inverse DFT
        fshift = dft_shift * aggregated_mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

        img_filtered[:,:,i] = img_back
        
    return Image.fromarray((NormalizeData(img_filtered) * 255).astype(np.uint8))


def compute_feature_importance(image, feature, model, masks, parameters):
    """
    computes the importance of the feature 

    feature should be a scalar belonging to the keys of the masks.

    takes as input the PIL image
    returns a scalar
    """

    # unwrap the parameters
    transforms = parameters['transforms']
    device = parameters['device']
    size = parameters['size']

    # define the set F \ i
    F = list(masks.keys())
    F.remove(feature)

    # compute the set of subsets of F
    subsets = powerset(F)

    phi = 0

    factorial_f = math.factorial(len(F)) # consider the total number of features

    for element in tqdm.tqdm(subsets): 
            
            targets = list(element)
            targets.append(feature) # add the ith feature
                            
            # compute x_s : image with the ith feature
            x_s = composed_image(image, masks, targets)
            x_s = resize(x_s, size = size)
            
            # compute x_i : image without the ith feature
            x_i = composed_image(image, masks, element)
            x_i = resize(x_i, size = size)
            
            # inference
            output = model.forward(transforms(x_s).unsqueeze(0).to(device, dtype=torch.float))
            y_s = torch.nn.functional.softmax(output, dim = 1)[:,1].detach().cpu().item() #> threshold
            
            # compute the predicted value for x_i 
            output = model.forward(transforms(x_i).unsqueeze(0).to(device, dtype=torch.float))
            y_i = torch.nn.functional.softmax(output, dim = 1)[:,1].detach().cpu().item() #> threshold

            factorial_e = math.factorial(len(element))
            factorial_f_e = math.factorial((len(F) - len(element) - 1))
            
            coeff = (factorial_e * factorial_f_e) / factorial_f
            
            phi += coeff * (y_s - y_i)

    return phi

def compute_shapeley_values(image_name, model, parameters, normalize = False):
    """
    computes the shapeley values for the inputed image name
    also return the masks used for the computatoin, if necessary
    """

    # unpack the parameters
    source_image_dir = parameters['source_path']
    b = parameters['b'] # bandwidth of the masks
    shape = parameters['image_shape']

    # compute the masks
    masks = generate_circular_masks(b, shape = shape)

    # open the image based on the name
    image = Image.open(os.path.join(source_image_dir, os.path.join('img', '{}.png'.format(image_name)))).convert('RGB')

    # define the set of features
    features = masks.keys()
    shapeley_values = {}

    for feature in features: # compute the feature importance
        phi = compute_feature_importance(image, feature, model, masks, parameters)

        shapeley_values[feature] = phi

    if normalize:
        return normalized_shapeley(shapeley_values)
    else:
        return shapeley_values

def normalized_shapeley(shapeley_values):
    """normalizes the dictionnary of shapeley values"""

    total_effect = np.sum([shapeley_values[s] for s in shapeley_values.keys()])
    normalized_shap_values = {k : (shapeley_values[k] / total_effect) * 100 for k in shapeley_values.keys()}   

    return normalized_shap_values


def shapey_dataset(n, model, parameters, normalize = False):
    """
    computes the shapeley values for n images of the dataset
    """



    dataset_dir = parameters['dataset_dir']
    seed = parameters['seed']

    # set up the seed 
    np.random.seed(seed)

    # get the list of images
    img_list = [i[:-4] for i in os.listdir(dataset_dir) if not i[0] == '.']

    subset = np.random.choice(img_list, size=n, replace=False)

    shapeley_dataset = {}

    for image in tqdm.tqdm(subset):

        shapeley_values = compute_shapeley_values(image, model, parameters, normalize = normalize)
        shapeley_dataset[image] = shapeley_values

    return shapeley_dataset



def convert_evolution_probs_to_array(evolution, baseline_probs, n_components):
    """
    takes as input the dictionnary of evolutions, with the structure

    { pos : {
        img : probs,
        ...
    },
        neg : {
        img : probs
        }
    }

    baseline probs as the structure : 
    {img : prob}

    and returns an array with n_img rows and n_components + 1 columns
    where the 0th column is the baseline prob for the image
    """
    img_list = list(baseline_probs.keys())
    n_imgs = len(img_list)
    
    table = np.zeros((n_imgs, n_components + 1))

    for i, img in enumerate(img_list):

        label = 'pos' if img in evolution['pos'].keys() else 'neg'

        baseline_prob = baseline_probs[img]
        probs = evolution[label][img]

        table[i,0] = baseline_prob
        table[i,1:] = probs

    return table

def delta_probs(evolution_array):
    """
    computes the variation in probability wrt to the baseline (column 0)
    for each frequency component
    
    returns a (n_images, n_components) array
    
    """
    
    n_images = evolution_array.shape[0]
    n_components = evolution_array.shape[1] - 1
    
    out = np.zeros((n_images, n_components))
    
    for i in range(n_images):
        
        baseline_prob = evolution_array[i,0]
        probs = evolution_array[i,1:]
        
        out[i,:] = probs - baseline_prob
        
    return out