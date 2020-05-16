import glob
import os
import nibabel as nib
import numpy as np
import sys
import pandas as pd
import pdb
from tqdm import tqdm


########################################################################################################################
# Code for preprocessing all of the scans and storing them in numpy arrays. Done so that preprocessing wouldn't be
# repeated during training, thereby speeding up the training process. Assumes the data is all in the specified folder
# and not within sub folders in it (ie <Brats_ID> folder that stores the scans for that Brats_ID is in that folder and
# not a subfolder). Preprocessing first crops each image to keep the brain region only if specified after which the pixel
# values are standardized and scaled to the range 0 and 1.

# use the samples(50) added in Brats2019 to be the validation set of Brats2018


# INPUT arguments:
#   arg1: path to where the raw scans to preprocess are stored
#   arg2: path where to save the preprocessed scans
#   arg3: Specify 1 if to crop images to exclude zero regions outside brain. 0 if pixel intensity scaling only.
#   arg4: Specify 1 if to preprocess training data (segmentation labels as well) or 0 if test data only.
#
# OUTPUT:
#   Preprocessed data stored in compressed npz files in the save_preprocessed_path
########################################################################################################################


def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    # 有点像“去掉最低分去掉最高分”的意思,使得数据集更加“公平”
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)  # 限定范围numpy.clip(a, a_min, a_max, out=None)

    # 除了黑色背景外的区域要进行标准化
    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9  # 黑色背景区域
        return tmp


def compare_site_names(one, two):
    df1 = pd.read_csv(one, header=None)
    df2 = pd.read_csv(two, header=None)
    data1 = df1.to_dict(orient='list')[0]
    data2 = df2.to_dict(orient='list')[0]
    diff1 = set(data1).difference(data2)
    diff2 = set(data2).difference(data1)
    return list(diff2)


if __name__ == '__main__':

    save_preprocessed_data_path = '/data/xwj_work/Brats2018/val2019/'
    crop_data = True
    train_data = True

    # Create the folder to store preprocessed data in, exit if folder already exists.
    if not os.path.isdir(save_preprocessed_data_path):
        os.mkdir(save_preprocessed_data_path)

    hgg_diff = compare_site_names("18hgg.csv", "19hgg.csv")
    lgg_diff = compare_site_names("18lgg.csv", "19lgg.csv")
    # Brats2019路径
    Brats2019 = '/data/xwj_work/Brats2019/raw/MICCAI_BraTS_2019_Data_Training'
    extralist = []
    for idx in range(len(hgg_diff)):
        mystr = "HGG/BraTS19" + hgg_diff[idx]
        extralist.append(mystr)

    for idx in range(len(lgg_diff)):
        mystr = "LGG/BraTS19" + lgg_diff[idx]
        extralist.append(mystr)
    print ('2019 has %d sample more than 2018')
    for extraitem in tqdm(extralist):
        path = Brats2019 + '/' + extraitem
        print(path)
        temp = glob.glob(path + "/*_t1.nii.gz")[0]

        # Load in the the different modalities
        img_t1 = nib.load(glob.glob(path + "/*_t1.nii.gz")[0]).get_fdata()
        img_t1ce = nib.load(glob.glob(path + "/*_t1ce.nii.gz")[0]).get_fdata()
        img_t2 = nib.load(glob.glob(path + "/*_t2.nii.gz")[0]).get_fdata()
        img_flair = nib.load(glob.glob(path + "/*_flair.nii.gz")[0]).get_fdata()

        # If preprocessing training data, load in the segmentation label image too
        if train_data:
            img_seg = nib.load(glob.glob(path + "/*_seg.nii.gz")[0]).get_fdata().astype('long')
        img_seg[img_seg == 4] = 3  # Replace label 4 with label 3

        if crop_data:
            # Crop the images to only keep bounding box area of the brain, with the bounding box atleast 128 length in each dimension
            r = np.any(img_t1, axis=(1, 2))
            c = np.any(img_t1, axis=(0, 2))
            z = np.any(img_t1, axis=(0, 1))
            rmin, rmax = np.where(r)[0][[0, -1]]
            cmin, cmax = np.where(c)[0][[0, -1]]
            zmin, zmax = np.where(z)[0][[0, -1]]

            if (rmax - rmin < 127):
                diff = 127 - (rmax - rmin)
                pad_left = int(diff / 2)
                pad_right = diff - pad_left
                rmin = rmin - pad_left
                rmax = rmax + pad_right

            if (cmax - cmin < 127):
                diff = 127 - (cmax - cmin)
                pad_left = int(diff / 2)
                pad_right = diff - pad_left
                cmin = cmin - pad_left
                cmax = cmax + pad_right

            if (zmax - zmin < 127):
                diff = 127 - (zmax - zmin)
                pad_left = int(diff / 2)
                pad_right = diff - pad_left
                zmin = zmin - pad_left
                zmax = zmax + pad_right

            img_t1 = img_t1[rmin: rmax + 1, cmin: cmax + 1, zmin: zmax + 1]
            img_t1ce = img_t1ce[rmin: rmax + 1, cmin: cmax + 1, zmin: zmax + 1]
            img_t2 = img_t2[rmin: rmax + 1, cmin: cmax + 1, zmin: zmax + 1]
            img_flair = img_flair[rmin: rmax + 1, cmin: cmax + 1, zmin: zmax + 1]
            img_seg = img_seg[rmin: rmax + 1, cmin: cmax + 1, zmin: zmax + 1]

        # Standardize and scale pixel intensity values and store all the modalities in the same array
        X = []
        for modality in [img_t1, img_t1ce, img_t2, img_flair]:
            brain_region = modality > 0  # Get region of brain to only manipulate those voxels
            up = np.percentile(modality[brain_region], 99)
            down = np.percentile(modality[brain_region], 1)
            modality[brain_region] = np.clip(modality[brain_region], down,
                                             up)  # 限定范围numpy.clip(a, a_min, a_max, out=None)
            mean = np.mean(modality[brain_region])
            stdev = np.std(modality[brain_region])
            new_img = np.zeros(img_t1.shape)
            new_img[brain_region] = (modality[brain_region] - mean) / stdev  # Standardize by mean and stdev
            # new_img[new_img > 5] = 5  # Clip outliers
            # new_img[new_img < -5] = -5
            Maximum = np.max(new_img)
            Minimum = np.min(new_img[brain_region])
            Range = Maximum - Minimum
            new_img[brain_region] = (new_img[brain_region] - Minimum) / Range  # Scale to be between 0 and 1
            X.append(new_img.astype('float32'))
        save_scans = save_preprocessed_data_path + extraitem[3:] + '_scans'
        np.savez_compressed(save_scans, data=X)
        if train_data:
            save_mask = save_preprocessed_data_path + extraitem[3:] + '_mask'
            np.savez_compressed(save_scans, data=img_seg)
