# -*- coding: utf-8 -*-

# libraries
import os
import numpy as np
import torch
from torch.nn import functional as F
import tqdm
from PIL import Image
import torchvision
import skimage
from scipy.stats import entropy
import random
from sklearn.metrics import f1_score
from PIL import ImageOps
import cv2




def spectral_redundancy(explanation, spatial_cam, size, grid_size, threshold = 0.95, levels = 3):
    """
    computes the spectral redundancy, which simulateneously summarizes
    the spread in the spatio-scale space

    the measure is defined as follows:
    1) computation of the local maxima in the nyquist square
    2) computation of the distance wrt the upper left corner (characterizes the scale spread) = alpha
    3) computation of the relative relative localizations (i.e. between 0 and 1 in each cell of the 
    wavelet transform)
    4) summation of the localizations on the spatial grid : 
        each cell (i,j) is += 1 / alpha if there is a maxima in the spot (i,j)
        and alpha is the inverse of the distance to the ul corner 
    at this step, we have a localization map weighted by the distance of the hotpots

    5) computation of the spatial spread (takes values in [1, dist_barycenter])
    6) aggregation : spectral redundancy = 1 / spatial_spread \times sum(coeffs(i,j))

    returns a scalar
    """

    # parameters
    width = grid_size // 2 ** (levels) # dimensions of the spatial map
    bins = [(i + 1) / width for i in range(width)] # bins to be used with the relative coordinates
    max_dist = np.sqrt(2) * np.sqrt(size ** 2 )

    # computation of the local maxima in the nyquist sqare
    maxima = compute_spatial_spread(explanation, threshold, scalar = False)
    # compute the spatial spread
    spatial_spread = compute_spatial_spread(spatial_cam)
    spatial_spread = max(1, spatial_spread) # rescale values equal to 0 to 1
    spatial_spread /= max_dist # normalize by the maximum spread

    # computation of the distances wrt the ul corner
    # of the wcam
    if maxima.shape[0] > 0:
        distances = np.linalg.norm(maxima, axis =1)

        # split the space into regions 
        regions = define_regions(size, levels)

        # redundancy map
        redundancy = np.zeros((width, width))

        for i, m in enumerate(maxima):

            # localize the local maxima
            value = regions[m[0], m[1]]

            # find the boundaries of the region
            coordinates = np.argwhere(regions == value)
            x_max, y_max = np.max(coordinates[:,0]), np.max(coordinates[:,1])

            # extract the coordinates of the 
            # local maxima 
            x, y = m[0], m[1]

            # normalize the coordinates
            x_norm = x / x_max
            y_norm = y / y_max

            # discretize the coordinates
            x_bin = np.digitize(x_norm, bins = bins)
            y_bin = np.digitize(y_norm, bins = bins)

            # increase the redundancy map by the inverse of the distance
            redundancy[x_bin, y_bin] += (1 / distances[i])

            # return the value of the redundancy index
            return (1 / spatial_spread) * sum(redundancy.flatten())


def spectral_spread(explanations, threshold = 0.95):
    """
    computes the spectral spread based on the WCAM
    and the quantile thresholding

    the spectral spread is defined as the distance between 
    the upper left corner and the barycenter + the distance between 
    the barycenter and its farthest point

    returns the scalar corresponding to the spread
    """
    # retrieve the maxima
    maxima = compute_spatial_spread(explanations, threshold, scalar = False)

    # compute the barycenter
    barycenter = np.mean(maxima, axis = 0)

    # compute the distance between the barycenter and the upper left 
    # corner of the image
    spread = np.sqrt(
        barycenter[0]**2 + barycenter[1]**2
    )

    # compute the distance between the barycenter and the other points
    points = len(maxima)
    barycenter = np.vstack([barycenter] * points)

    # compute the distance
    # consider the maximum distance and add it to the distance

    d = np.linalg.norm(barycenter-maxima, axis = 1)
    farthest = np.max(d)
    spread += farthest

    return spread

def confusion_matrix_index(preds,truth):
    """
    return the indices of the status of the samples
    """

    confusion = np.array(preds) / np.array(truth)

    # status depends on the outcome
    tn = np.where(np.isnan(confusion))[0]
    tp = np.where(confusion == 1)[0]
    fn = np.where(confusion == 0)[0]
    fp = np.where(np.isinf(confusion))[0]

    return tn, tp, fn, fp

def compute_spatial_spread(map, quantile = 0.9, scalar = True):
    """
    computes the spatial spread of the spatial WCAM
    it corresponds to the distance between the two farthest maxima
    otherwise it is set to 0
    based on the quantile thresholding

    if scalar is false, it will return the coordinates of 
    the centers
    """
    maxima = get_maxima_map(map, quantile)

    if scalar:
        return spatial_spread(maxima)
    
    else: 
        return maxima

    

def return_local_maxima(contours, map):
    """
    returns the coordinates of the local maxima given 
    the contours passed as input.
    """

    maxima = []

    # fill the 
    for contour in contours:
        mask = np.zeros(map.shape)
        cv2.fillPoly(mask, pts =[contour], color=(255,255,255))
        mask /= 255.

        filtered = mask * map
        x, y = np.unravel_index(filtered.argmax(), map.shape)

        maxima.append(np.array([x,y]))
    

    return np.array(maxima)

def return_maxima_of_slice(slice, map):
    """
    given a slice, computes the contours and 
    returns the different local maxima
    """

    # compute the contours
    contours, _ = cv2.findContours(slice, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    return return_local_maxima(contours, map)

def slice_map(map, quantile):
    """
    returns the sliced mask of the map
    given the quantile passed as input
    """
    binaries = np.zeros(map.shape, dtype = np.uint8)
    q = np.quantile(map.flatten(), quantile)
    x, y = np.where(map > q)
    binaries[x,y] = 1

    return binaries

def get_maxima_map(map, quantile):
    """
    returns the coordinates of the maxima
    of the map, after quantile thresholding
    according to the quantile passed as input

    returns : a np.ndarray of coordinates
    """

    # threshold the map
    binary = slice_map(map, quantile)

    return return_maxima_of_slice(binary, map)


def spatial_spread(maxima):
    """
    given the array of maxima, 
    computes the center
    and the distance
    """

    center = np.mean(maxima, axis = 0)

    points = len(maxima)


    # if only one maxima, then the distance is 0
    if points == 0:
        return 0.
    else:
        # reshape the center to match the 
        # size of the maxima
        center = np.vstack([center] * points)

        # compute the distance
        d = np.linalg.norm(center-maxima, axis = 1)
        d = np.sort(d)[::-1]
        # return the cumulated distance of the two 
        # points that are the farthest from the center
        return np.sum(d[:2])

def add_lines(size, levels, ax):
    """
    add white lines to the ax where the 
    WCAM is plotted
    """
    h, v = plot_wavelet_regions(size, levels)

    ax.set_xlim(0,size)
    ax.set_ylim(size,0)

    for k in range(levels):
        ax.plot(h[k][:,0], h[k][:,1], c = 'w')
        ax.plot(v[k][:,0], v[k][:,1], c = 'w')

    return None

def compute_perturbations(image, perturbation = "Gaussian", batch_size = 256, model = None, p = None):
    """
    Generates a batch of perturbed images according to the perturbation passed
    as input
    input : a pil image 
    p : a dictionnary of parameters

    output : a list of perturbed pil images. we generate batch_size perturbed images
    """

    perturbed_images = []

    if perturbation == "Gaussian":

        if p is not None:
            upper_noise = p['sigma_noise']
            upper_blur = p['sigma_blur']
        else:
            upper_noise = 0.01
            upper_blur = 3

        image = np.array(image)
        noise_var = np.random.uniform(0, upper_noise, size = batch_size) 
        blur_var = np.random.uniform(0.01, upper_blur, size = batch_size)
        # apply the noise
        for nv, bv in zip(noise_var, blur_var):
            # apply noise and blur
            img = skimage.filters.gaussian(image, sigma = bv)
            img = skimage.util.random_noise(img, mode='gaussian', mean=0, var=nv, clip = True, seed = 42)

            # convert back the image as PIL image
            img = Image.fromarray((NormalizeData(img) * 255).astype(np.uint8))

            perturbed_images.append(img)

        return perturbed_images
 

def define_regions(size, levels):
    """
    helper that splits the nyquist square into h,v,d regions
    at each level depending on the size of the input
    and the number of levels

    by convention, the approximation coefficients 
    have the highest value, which is then decreasing down 
    to the highest frequency detail coefficients
    """

    # define the number of regions : 3 per level
    # and 1 for the last approximation coefficients
    n_regions = (3 * levels) + 1

    # define the mask. Each region will be labelled
    # 1, ..., n_regions
    mask = n_regions * np.ones((size, size))

    # loop over the levels
    for l in range(levels):

        # define the labels for each detail coefficient
        # from each level
        offset = l * levels

        h, v, d = offset + 1, offset + 2, offset + 3

        # regions in the input that must be labelled
        start = int(size / 2 ** (l + 1))
        end = int(size / (2 ** l))

        # label the regions 
        mask[:end,start:end] = h
        mask[start:end,:end] = v
        mask[start:end, start:end] = d

    return mask

def extract_importance_coefficients(wcams, size ,levels = 3):
    """
    plots the histogram of the importance of the different regions
    of the wcam
    """

    # label the regions in the nysquist square

    mask = define_regions(size, levels)

    # number of regions
    n_regions = np.max(mask.flatten()).astype(int)

    values = []

    # loop over the regions 
    for r in range(1, n_regions + 1):

        local_mask = (mask == r).astype(int)

        n_coeffs = np.sum(local_mask)
        value = np.sum(local_mask * wcams)

        values.append(value / n_coeffs)

    return np.array(values)

# set of auxiliary functions
def plot_wavelet_regions(size,levels):
    """
    returns the dictonnaries with the
    coordinates of the lines for the plots
    """

    center = size // 2
    h, v = {}, {} # dictionnaries that will store the lines
    # initialize the first level
    h[0] = np.array([
        [0, center],
        [size,center],
    ])
    v[0] = np.array([
        [center,size],
        [center,0],
    ])
    # define the horizontal and vertical lines at each level
    for i in range(1, levels):
        h[i] = h[i-1] // 2
        h[i][:,1]
        v[i] = v[i-1] // 2
        v[i][:,1] 
        
    return h, v        


def ml_f1_score(targets, preds):
    """
    taken from https://github.com/qiqi-helloworld/ABSGD/blob/main/wilds-competition/utils.py
    """
    return f1_score(targets, preds, average=None).mean()


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

        fp_index = np.where(confusion_vector == np.inf)[0]
        fn_index = np.where(confusion_vector == 0)[0]

        tp_index = np.where(confusion_vector == 1.)[0]
        tn_index = np.where(np.isnan(confusion_vector))[0]

    else:

        true_positives = torch.sum(confusion_vector == 1).item()
        false_positives = torch.sum(confusion_vector == float('inf')).item()
        true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives = torch.sum(confusion_vector == 0).item()

        # convert the confusion vector as an array 
        confusion_vector = confusion_vector.detach().cpu().numpy()

        fp_index = np.where(confusion_vector == np.inf)[0]
        fn_index = np.where(confusion_vector == 0)[0]

        tp_index = np.where(confusion_vector == 1.)[0]
        tn_index = np.where(np.isnan(confusion_vector))[0]
    
    indices = {
        'tp' : tp_index,
        'fp' : fp_index,
        'tn' : tn_index,
        "fn" : fn_index

    }

    return true_positives, false_positives, true_negatives, false_negatives, indices 


def return_f1(precision, recall):
    """
    Given an array of precisions and of recalls
    computes the F1 score
    """
    return 2 * (np.array(precision) * np.array(recall)) / (np.array(precision) + np.array(recall))

def evaluate_model_on_samples(x, model, batch_size):
    """
    inference loop of a model on a set of samples whose size
    can exceed the batch size.

    returns the vector of predicted classes
    """

    # retrieve the device
    device = next(model.parameters()).device
    
    # predictions vector
    y = np.empty(len(x))

    # nb batch
    nb_batch = int(np.ceil(len(x) / batch_size))

    model.eval()

    with torch.no_grad():

        for batch_index in range(nb_batch):
            # retrieve masks of the current batch
            start_index = batch_index * batch_size
            end_index = min(len(x), (batch_index+1)*batch_size)

            # batch samples
            batch_x = x[start_index : end_index, :,:,:].to(device)

            # predictions
            preds = model(batch_x)
            batch_y =  F.softmax(preds, dim=1).detach().cpu().numpy()[:,1]

            # store the results
            y[start_index:end_index] = batch_y

    return y


def evaluate(model, data, device, threshold):
    """
    evaluates a model on a dataset data
    returns the F1 and the confusion matrix

    args: 
    - model : a torch model
    - data : a Dataloader
    - device : str the cuda device on which inference is done

    """

    # send the model on the device
    model = model.to(device)
    model.eval()

    # Forward pass on the validation dataset and accumulate the probabilities 
    # in a single vector.

    probabilities = []
    all_labels = []

    names = []

    with torch.no_grad():

        for data in tqdm.tqdm(data):

            images, labels, ids = data


            # move the images to the device
            images = images.to(device)

            labels = labels.detach().cpu().numpy()
            all_labels.append(list(labels))

            # calculate outputs by running images through the network and computing the prediction using the threshold
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).detach().cpu().numpy() # the model returns the unnormalized probs. Softmax it to get probs
            probabilities.append(list(probs[:,1]))

            names.append(ids)

    # Convert the probabilities and labels as arrays
    probabilities = sum(probabilities, [])
    probabilities = np.array(probabilities)

    labels = sum(all_labels, [])
    labels = np.array(labels)

    # Compute the precision and recall for the best threshold
    predicted = np.array(probabilities > threshold, dtype = int)

    # compute the confusion matrix
    tp, fp, tn, fn, indices = confusion(predicted, labels)


    # flatten the list of names
    names = list(sum(names, ()))

    # retrieve the name of the images that are missclassified
    wrong_predictions = {}

    for case in indices.keys():

        wrong_predictions[case] = [names[k] for k in indices[case]]


    # return the values
    precision = np.divide(tp, (tp + fp))
    recall = np.divide(tp, (tp + fn))
    f1 = np.divide(2 * precision * recall, precision + recall)

    return f1, tp, fp, tn, fn, wrong_predictions, probabilities, labels



def compute_grayscale_spectrum(image):
    """
    computes the image spectrum of the imputed PIL image, given sigma_n and sigma_b
    """
    
    img = ImageOps.grayscale(image) 
    
    array_img = np.array(img)
    img_float32 = np.float32(array_img) 
    dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1])

    return magnitude_spectrum


def load_image(directory, name, p = {}):
    """
    loads an image an returns it as a PIL image
    applied the transforms specified in the parameters dictionnary p
    """

    image = Image.open(os.path.join(directory,name)).convert('RGB')

    if 'downsample' in p.keys(): # case where the image needs downsampling
        out_size = p['downsample']
        image = image.resize((out_size, out_size))

    if 'noise' in p.keys(): # noise is not featured in the torchvision transforms
        var = p['noise']
        img = random_noise(np.array(image), mode='gaussian', mean=0, var=var, clip = True, seed = 42)
        img = Image.fromarray((NormalizeData(img) * 255).astype(np.uint8))

    if 'resize' in p.keys():

        size = p['resize']
        image, x, y = apply_crop(image, size)

    if 'transforms' in p.keys(): # case where the image needs to be transformed
        transforms = p['transforms']
        image = transforms(image)     


    if 'masks_directory' in p.keys(): # return the label if specified
        masks_directory = p['masks_directory']
        label = np.array([os.path.exists(os.path.join(masks_directory, name))], dtype = int)
        
        if 'resize' in p.keys():

            label = update_label(x, y, name, size, masks_directory)
        
        return image, label
    
    else: 
        return image


def apply_crop(image, size):
    """
    crops the image by randomly picking a point
    in an anchor corner
    returns the transformed image and the x, y anchor
    """            
    max_x, max_y = image.size[0], image.size[1] # get the size of the image
    x_span, y_span = max(0,max_x - size), max(0,max_y - size) # coordinates of the anchor box

    # otherwise, the center of the image is considered
    x, y = int(x_span / 2), int(y_span / 2)
    
    return torchvision.transforms.functional.crop(image, y, x, size, size), x, y

def update_label(x, y, name, size, mask_dir):
    """
    checks on the mask that the label is unchanged.
    """

    # open the mask
    mask = Image.open(os.path.join(mask_dir, name)).convert("RGB")
    mask_cropped = torchvision.transforms.functional.crop(mask, y, x, size, size)

    # check whether the panel is still on the mask
    # by counting the number of activated pixels on the cropped mask.
    return int(torch.sum(mask_cropped) > 5)    


def prediction_stability(model, image, label, parameters, batch_size = 128):
    """
    returns the entropy of the vector of probabilities of the model 
    for a sample perturbed according to the perturbation dictionnary

    perturbations consist in random combinations of gaussian noise and blur

    inputs 
    - model : a torch model
    - image : a pil image
    - label : the label of the image. can be the predicted label
    - parameters : the dictionnary that contains the perturbations and the transforms 
    - batch_size : the batch size
    
    returns: 
    the entropy of the output probability vector
    """

    # compute the perturbations of the image
    # for simplicity, we do one forward of perturbations

    parameters['count'] = batch_size 
    perturbed_images = perturb_image(image, parameters)

    # send the perturbed images to the 
    transforms = parameters['transforms']
    perturbed_images = torch.stack([transforms(im) for im in perturbed_images]).cuda()

    outputs = model(perturbed_images)
    outputs = outputs[:, label].cpu().detach().numpy()

    return entropy(outputs)


def perturb_image(image, perturbations):
    """
    perturbs an PIL image passed as input
    perturbation is done accordingly to the perturbations dictionnary
    """
    random.seed(42)

    if not 'blur' in perturbations.keys():
        raise ValueError ; 'Blur is required in the perturbations'

    if not 'noise' in perturbations.keys():
        raise ValueError ; 'noise is required in the perturbations'

    count = perturbations['count'] # number of perturbations to do

    var = p['noise']
    sigma = p['blur']

    img = np.array(image) # convert as an array

    images = []
    for _ in range(count):
        img = np.array(image) # convert the input image as an array

        img = random_noise(img, mode='gaussian', mean=0, var=var, clip = True, seed = 42)

        sig = np.random.uniform(1, sigma)
        img = gaussian(img, sigma=sig)

        # convert back the image as a PIL image
        img = Image.fromarray((NormalizeData(img) * 255).astype(np.uint8))
        images.append(img)

    return images