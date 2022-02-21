#!/usr/bin/env python3

from nibabel.processing import resample_from_to
import scipy
import scipy.spatial
import nibabel as nib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle as pkl
import inspect
from nilearn.image import math_img
from statistics import mean
import scipy.stats as stat
from scipy.signal import resample
import os
import pandas as pd
import itertools


############# COMPUTE DIFF.

def compute_rel(f1, file_name=None, f2=None):
    # Compute relative diff in BT
    if f2 is not None:
        f1 = nib.load(f1)
        f1_data = f1.get_fdata()
        if type(f2) == str : f2 = nib.load(f2)
        f2_data = f2.get_fdata()
        img_rel = (f1_data - f2_data)
        # activated regions have nonzero values
        img_rel = np.where((f1_data == 0) & (f2_data == 0), np.nan, img_rel)
        nft_img = nib.Nifti1Image(img_rel, f1.affine, header=f1.header)
        nib.save(nft_img, os.path.join('data/diff/', '{}-rel.nii.gz'.format(file_name)))

    # Compute relative diff in WT
    else:
        sample_list = []
        flag_ = False
        for i in range(1, 11):
            file_ = f1.replace('MCA', str(i))
            file_ = nib.load(file_)
            file_d = file_.get_fdata()
            sample_list.append(file_d)
            if flag_:
                mask = np.where((file_d == 0) & (tmp == 0), np.nan, 1)
            flag_ = True
            tmp = file_d
        diff_list = []
        for pair_ in list(itertools.combinations(sample_list, 2)):
            f1_data = pair_[0]
            f2_data = pair_[1]
            diff_list.append((f1_data - f2_data))
        diff_avg = np.mean(diff_list, axis=0)
        img_rel = np.where((mask == 1), diff_avg, np.nan)
        nft_img_rel = nib.Nifti1Image(img_rel, file_.affine, header=file_.header)
        nib.save(nft_img_rel, os.path.join('data/diff/', '{}-rel.nii.gz'.format(file_name)))
        return nft_img_rel


def combine_diffs(f1, f2, meta_, file_name):
    var_f2_res = resample_from_to(f2, f1, order=0)
    # to combine two image variances, we use: var(x+y) = var(x) + var(y) + 2*cov(x,y)
    # and since the correlation between two arrays are so weak, we droped `2*cov(x,y)` from the formula
    f1t = f1.get_fdata()
    f2t = var_f2_res.get_fdata()
    wt_diff = (np.nan_to_num(f1.get_fdata()) + np.nan_to_num(var_f2_res.get_fdata()))
    # nan bg images
    wt_diff = np.where((np.isnan(f1t)) | (np.isnan(f2t)), np.nan, wt_diff)
    nft_img = nib.Nifti1Image(wt_diff, meta_.affine, header=meta_.header)
    nib.save(nft_img, os.path.join('data/diff/', '{}-rel.nii.gz'.format(file_name)))


def var_between_tool(tool_results):
    for type_ in ['unthresh', 'thresh']:
        f = 'stat_file'
        if type_ == 'thresh': f = 'act_deact'

        fsl_ = tool_results['fsl'][f]
        afni_ = tool_results['afni'][f]
        spm_ = tool_results['spm'][f]
        # resampling first image on second image
        spm_res = resample_from_to(nib.load(spm_), nib.load(fsl_), order=0)
        afni_res = resample_from_to(nib.load(afni_), nib.load(fsl_), order=0)
        # compute abs diff
        compute_rel(fsl_, 'fsl-afni-{}'.format(type_), f2=afni_res)
        compute_rel(fsl_, 'fsl-spm-{}'.format(type_), f2=spm_res)
        spm_res = resample_from_to(nib.load(spm_), nib.load(afni_), order=0)
        compute_rel(afni_, 'afni-spm-{}'.format(type_), f2=spm_res)

    ## unthresholded subject-level
    for i in range(1, 17):
        fsl_ = tool_results['fsl']['SBJ'].replace('NUM', '%.2d' % i )
        afni_ = tool_results['afni']['SBJ'].replace('NUM', '%.2d' % i )
        spm_ = tool_results['spm']['SBJ'].replace('NUM', '%.2d' % i )
        # resampling first image on second image
        spm_res = resample_from_to(nib.load(spm_), nib.load(fsl_), order=0)
        afni_res = resample_from_to(nib.load(afni_), nib.load(fsl_), order=0)
        # compute abd diff
        compute_rel(fsl_, 'sbj{}-fsl-afni-unthresh'.format('%.2d' % i), f2=afni_res)
        compute_rel(fsl_, 'sbj{}-fsl-spm-unthresh'.format('%.2d' % i), f2=spm_res)
        spm_res = resample_from_to(nib.load(spm_), nib.load(afni_), order=0)
        compute_rel(afni_, 'sbj{}-afni-spm-unthresh'.format('%.2d' % i), f2=spm_res)


def var_between_fuzzy(mca_results):
    for type_ in ['thresh', 'unthresh']:
        f = 'stat_file'
        if type_ == 'thresh': f = 'act_deact'
        # compute abs diff
        fsl_abs = compute_rel(mca_results['fsl'][f], 'fuzzy-fsl-{}'.format(type_), f2=None)
        afni_abs = compute_rel(mca_results['afni'][f], 'fuzzy-afni-{}'.format(type_), f2=None)
        spm_abs = compute_rel(mca_results['spm'][f], 'fuzzy-spm-{}'.format(type_), f2=None)
        combine_diffs(fsl_abs, afni_abs, nib.load(mca_results['fsl'][f].replace('MCA', '1')),
                      'fuzzy-fsl-afni-{}'.format(type_))
        combine_diffs(fsl_abs, spm_abs, nib.load(mca_results['fsl'][f].replace('MCA', '1')),
                      'fuzzy-fsl-spm-{}'.format(type_))
        combine_diffs(afni_abs, spm_abs, nib.load(mca_results['afni'][f].replace('MCA', '1')),
                      'fuzzy-afni-spm-{}'.format(type_))

    # unthresholded subject-level
    for i in range(1, 17):
        fsl_ = mca_results['fsl']['SBJ'].replace('NUM', '%.2d' % i )
        afni_ = mca_results['afni']['SBJ'].replace('NUM', '%.2d' % i )
        spm_ = mca_results['spm']['SBJ'].replace('NUM', '%.2d' % i )
        # compute abs diff
        fsl_abs = compute_rel(fsl_, 'sbj{}-fuzzy-fsl-unthresh'.format('%.2d' % i), f2=None)
        afni_abs = compute_rel(afni_, 'sbj{}-fuzzy-afni-unthresh'.format('%.2d' % i), f2=None)
        spm_abs = compute_rel(spm_, 'sbj{}-fuzzy-spm-unthresh'.format('%.2d' % i), f2=None)
        combine_diffs(fsl_abs, afni_abs, nib.load(fsl_.replace("MCA", '1')),
                      'sbj{}-fuzzy-fsl-afni-unthresh'.format('%.2d' % i))
        combine_diffs(fsl_abs, spm_abs, nib.load(fsl_.replace("MCA", '1')),
                      'sbj{}-fuzzy-fsl-spm-unthresh'.format('%.2d' % i))
        combine_diffs(afni_abs, spm_abs, nib.load(afni_.replace("MCA", '1')),
                      'sbj{}-fuzzy-afni-spm-unthresh'.format('%.2d' % i))


############# PLOT DIFF.

def check_intensities(bt_, wt_, type_, ax, tool1):
    # Correlated maps
    if type(tool1) == str: img1 = nib.load(tool1)
    else: img1 = tool1
    img_data1 = img1.get_fdata()
    corr_map = np.where((0.99<bt_/wt_) & (bt_/wt_< 1.01) , img_data1, np.nan)#
    other_voxel = np.where((0.99<bt_/wt_) & (bt_/wt_< 1.01), np.nan, img_data1)#
    # print(f"bt other: min: {np.nanmin(other_voxel)} max: {np.nanmax(other_voxel)} mean {np.nanmean(other_voxel)} ")
    h_, bins = np.histogram(np.nan_to_num(other_voxel))
    ax.hist(np.reshape(other_voxel, -1), bins, log=True, label="Others")
    # print(f"bt corr: min: {np.nanmin(corr_map)} max: {np.nanmax(corr_map)} mean {np.nanmean(corr_map)} ")
    # h_, bins = np.histogram(np.nan_to_num(corr_map))
    ax.hist(np.reshape(corr_map, -1), bins, log=True, label="Correlated voxels")
    ax.legend()
    ax.set_title(type_)
    #plt.show()


def plot_diff_corr_group(sbj=False):
    ### Plot correlation of differences between BT and WT
    num=2
    if sbj == True: num=17
    for ind1, type_ in enumerate(['unthresh']):
        for i in range(1, num):
            fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(24, 12))
            for ind_, pair_ in enumerate(['fsl-spm', 'fsl-afni', 'afni-spm']):
                # read BT
                if sbj == True: btool = nib.load('./data/diff/sbj{}-{}-{}-rel.nii.gz'.format('%.2d' % i, pair_, type_))
                else: btool = nib.load('./data/diff/{}-{}-rel.nii.gz'.format(pair_, type_))
                bt_ = btool.get_fdata()
                data2 = np.reshape(bt_, -1)
                data2 = np.nan_to_num(data2)

                # read WT
                for ind_2, tool_ in enumerate(pair_.split('-')):
                    if sbj == True: bfuzzy = nib.load('./data/diff/sbj{}-fuzzy-{}-{}-rel.nii.gz'.format('%.2d' % i, pair_, type_))
                    else: bfuzzy = nib.load('./data/diff/fuzzy-{}-{}-rel.nii.gz'.format(tool_, type_))
                    wt_ = bfuzzy.get_fdata()
                    if btool.shape != bfuzzy.shape:
                        res_wt = resample_from_to(bfuzzy, btool, order=0)
                        wt_ = res_wt.get_fdata()
                    data1 = np.reshape(wt_, -1)
                    data1 = np.nan_to_num(data1)

                    # r, p = scipy.stats.pearsonr(data1, data2)
                    # print("P-value: {} and R: {}".format(p, r))
                    slope, intercept, r2, p2, stderr = scipy.stats.linregress(data1, data2)
                    line = 'Regression line'
                    y = intercept + slope * data1
                    label_r = "Regression line\np={:.3f}, r={:.3f}".format(p2, r2)

                    hist = ax[ind_2][ind_].hist2d(data1, data2, bins=20, norm=mpl.colors.LogNorm(), cmap=plt.cm.Reds)
                    ax[ind_2][ind_].plot(data1, y, color='black', label=label_r)

                    ax[ind_2][ind_].set_title('')
                    ax[ind_2][ind_].set_xlabel('')
                    # if ind_ == 0: 
                    ax[ind_2][ind_].set_ylabel(f'BT({pair_.upper()})', fontsize=14)
                    ax[ind_2][ind_].set_xlabel(f'WT({tool_.upper()})', fontsize=14)
                    # ax[ind_2][ind_].set_ylim([-7.5, 7.5])
                    # ax[ind_2][ind_].set_xlim([-3.5, 3.5])
                    # #ax[ind_].set_xticklabels(fontsize=14)
                    # #ax[ind_].set_yticklabels(fontsize=14)
                    ax[ind_2][ind_].legend(facecolor='white', loc='upper right', fontsize=12)
                    ax[ind_2][ind_].tick_params(axis='both', labelsize=12)
                    #ax[ind_].set_xscale('log')
                    plt.subplots_adjust(wspace=0.3)                    
            if sbj == True: fig.savefig('./paper/figures/abs/corr/sbj{}-rel-corr-{}-plot.png'.format('%.2d' % i, type_), bbox_inches='tight')
            else: fig.savefig('./paper/figures/abs/corr/rel-corr-{}-plot.png'.format(type_), bbox_inches='tight')

############# COMPUTE and PLOT DICES

def dump_variable(var_, name_):
    with open('./data/{}.pkl'.format(str(name_)),'wb') as f:
        pkl.dump(var_, f)


def load_variable(name_):
    with open('./data/{}.pkl'.format(name_),'rb') as fr:
        return pkl.load(fr)


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def compute_dice(data1_file, data2_file):

    # Load nifti images
    data1_img = nib.load(data1_file)
    data2_img = nib.load(data2_file)
    # Load data from images
    data2 = data2_img.get_data()
    data1 = data1_img.get_data()
    # Get asbolute values (positive and negative blobs are of interest)
    data2 = np.absolute(data2)
    data1 = np.absolute(data1)

    # Resample data2 on data1 using nearest neighbours
    data2_resl_img = resample_from_to(data2_img, data1_img, order=0)
    data2_res = data2_resl_img.get_data()
    data2_res = np.absolute(data2_res)

    # Masking (compute Dice using intersection of both masks)
    # bg_ = np.logical_or(np.isnan(data1), np.isnan(data2_res))

    # Binarizing data
    data1 = np.nan_to_num(data1)
    data1[data1>0] = 1
    data2_res = np.nan_to_num(data2_res)
    data2_res[data2_res>0] = 1

    num_activated_1 = np.sum(data1 > 0)
    num_activated_res_2 = np.sum(data2_res>0)

    # Masking intersections
    # data1[bg_] = 0
    # data2_res[bg_] = 0

    # Vectorize
    data1 = np.reshape(data1, -1)
    data2_res = np.reshape(data2_res, -1)
    # similarity = 1.0 - dissimilarity
    dice_score = 1-scipy.spatial.distance.dice(data1>0, data2_res>0)

    return (dice_score, num_activated_1, num_activated_res_2)


def keep_roi(stat_img, reg_, image_parc, filename=None):
    parc_img = nib.load(image_parc)
    data_img = nib.load(stat_img)
    # Resample parcellation on data1 using nearest neighbours
    parc_img_res = resample_from_to(parc_img, data_img, order=0)
    parc_data_res = parc_img_res.get_fdata(dtype=np.float32)
    colls = np.where(parc_data_res != reg_)
    parc_data_res[colls] = np.nan
    # data_img_nan = nib.Nifti1Image(parc_data_res, img_.affine, header=img_.header)
    # nib.save(data_img_nan, 'parce_region{}.nii.gz'.format(reg_))

    data_orig = data_img.get_data()
    # If there are NaNs in data_file remove them (to mask using parcelled data only)
    data_orig = np.nan_to_num(data_orig)
    # Replace background by NaNs
    data_nan = data_orig.astype(float)
    data_nan[np.isnan(parc_data_res)] = np.nan

    data_orig[np.isnan(parc_data_res)] = 0.0
    if np.all(data_orig == 0.0):
        print("There is not activation in R")
        s = 0
    else:
        print("There is activated voxel in R")
        s = 1

    # Save as image
    data_img_nan = nib.Nifti1Image(data_nan, data_img.get_affine())
    if filename is None:
        filename = stat_img.replace('.nii', '_nan.nii')
    nib.save(data_img_nan, filename)

    return(filename, s)


def get_dice_values(regions_txt, image_parc, tool_results, mca_results):
    # read parcellation data
    parc_img = nib.load(image_parc)
    parc_data = parc_img.get_fdata(dtype=np.float32)

    regions = {}
    with open(regions_txt) as f:
        for line in f:
            (key, val) = line.split()
            regions[int(key)+180] = val
            regions[int(key)] = 'R_' +  '_'.join(val.split('_')[1:])

    dices_ = {}
    file_cols = ["Region", "FSL IEEE", "SPM IEEE", "AFNI IEEE",
                 "FSL MCA1", "FSL MCA2", "FSL MCA3", "FSL MCA4", "FSL MCA5", "FSL MCA6",
                 "FSL MCA7","FSL MCA8", "FSL MCA9", "FSL MCA10",
                 "SPM MCA1", "SPM MCA2", "SPM MCA3", "SPM MCA4", "SPM MCA5", "SPM MCA6",
                 "SPM MCA7", "SPM MCA8", "SPM MCA9", "SPM MCA10",
                 "AFNI MCA1", "AFNI MCA2", "AFNI MCA3", "AFNI MCA4", "AFNI MCA5", "AFNI MCA6",
                 "AFNI MCA7", "AFNI MCA8", "AFNI MCA9", "AFNI MCA10"]
    dframe = pd.DataFrame(np.array([['R', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                                     28, 29, 30, 31, 32, 33]]), columns=file_cols)
    #for act_ in ['exc_set_file', 'exc_set_file_neg', 'act_deact', 'stat_file']:
    for act_ in ['act_deact']:
        ### Print global Dice values 
        bt1 = compute_dice(tool_results['fsl'][act_], tool_results['afni'][act_])[0]
        bt2 = compute_dice(tool_results['fsl'][act_], tool_results['spm'][act_])[0]
        bt3 = compute_dice(tool_results['afni'][act_], tool_results['spm'][act_])[0]
        print("Global Dice in BT for FSL-AFNI {}, FSL-SPM {}, and AFNI-SPM {}".format(bt1, bt2, bt3))

        for tool_ in ['fsl', 'spm', 'afni']:
            dice_wt_list = []
            for pair_ in list(itertools.combinations(range(1, 11), 2)):
                f1_data = mca_results[tool_][act_].replace('MCA', str(pair_[0]))
                f2_data = mca_results[tool_][act_].replace('MCA', str(pair_[1]))
                dice_wt_list.append(compute_dice(f1_data, f2_data)[0])
                # wt2 = compute_dice(mca_results[tool_][1][act_], mca_results[tool_][3][act_])[0]
                # wt3 = compute_dice(mca_results[tool_][2][act_], mca_results[tool_][3][act_])[0]
            print("Global Dice in WT for {} {}".format(tool_, np.mean(dice_wt_list)))

        masked_regions = {}
        for r in regions.keys():
            act_bin = []
            act_bin.append(regions[r])
            colls = np.where(parc_data == r)
            if regions[r] not in dices_.keys():
                dices_[regions[r]] = {}
                dices_[regions[r]]['size'] = len(colls[0])
                dices_[regions[r]][act_] = {}
                dices_[regions[r]][act_]['tool'] = {}
                dices_[regions[r]][act_]['mca'] = {}
            else:
                dices_[regions[r]][act_] = {}
                dices_[regions[r]][act_]['tool'] = {}
                dices_[regions[r]][act_]['mca'] = {}

            masked_regions['fsl'], s = keep_roi(tool_results['fsl'][act_], r, image_parc)
            act_bin.append(s)
            masked_regions['spm'], s = keep_roi(tool_results['spm'][act_], r, image_parc)
            act_bin.append(s)
            masked_regions['afni'], s = keep_roi(tool_results['afni'][act_], r, image_parc)
            act_bin.append(s)

            dices_[regions[r]][act_]['tool']['fsl-afni'] = compute_dice(masked_regions['fsl'], masked_regions['afni'])[0]
            dices_[regions[r]][act_]['tool']['fsl-spm'] = compute_dice(masked_regions['fsl'], masked_regions['spm'])[0]
            dices_[regions[r]][act_]['tool']['afni-spm'] = compute_dice(masked_regions['afni'], masked_regions['spm'])[0]
            # dices = (dice_res_1, dark_dice_1[1], dark_dice_2[1], num_activated_1, num_activated_2)

            for tool_ in mca_results.keys():
                for i in range(1, 11):
                    masked_regions['{}{}'.format(tool_, i)], s = keep_roi(mca_results[tool_][act_].replace('MCA', str(i)), r, image_parc)#, '{}1'.format(tool_))
                    act_bin.append(s)
                # masked_regions['{}2'.format(tool_)], s = keep_roi(mca_results[tool_][2][act_], r, image_parc)#, '{}2'.format(tool_))
                # act_bin.append(s)
                # masked_regions['{}3'.format(tool_)], s = keep_roi(mca_results[tool_][3][act_], r, image_parc)#, '{}3'.format(tool_))
                # act_bin.append(s)

                # dices_[regions[r]][act_]['mca']['{}1'.format(tool_)] = compute_dice(masked_regions['{}1'.format(tool_)], masked_regions['{}2'.format(tool_)])[0]
                # dices_[regions[r]][act_]['mca']['{}2'.format(tool_)] = compute_dice(masked_regions['{}1'.format(tool_)], masked_regions['{}3'.format(tool_)])[0]
                # dices_[regions[r]][act_]['mca']['{}3'.format(tool_)] = compute_dice(masked_regions['{}2'.format(tool_)], masked_regions['{}3'.format(tool_)])[0]

            dframe2 = pd.DataFrame(np.array([act_bin]),
                                   columns=file_cols)
            dframe = dframe.append(dframe2, ignore_index=True)

    dframe.to_csv('./data/active_regions.csv')
    dump_variable(dices_, retrieve_name(dices_)[0])
    return dices_


def plot_dices(dices_):
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(16, 12))
    marker_ = ['o', '*', 'v', '>', '1', '2', '3', '4', 'P']
    colors = ['red', 'blue', 'green']
    for ind1_, act_ in enumerate(['act_deact']):
        for ind_, tool_ in enumerate(['fsl-afni', 'fsl-spm', 'afni-spm']):
            tool_list = []
            mca_list = []
            for reg_ in dices_.keys():
                tool_list.append(dices_[reg_][act_]['tool'][tool_])
                f1, f2 = tool_.split('-')
                mca_mean = []
                for i in [1, 2, 3]:
                    mca_mean.append(dices_[reg_][act_]['mca']['{}{}'.format(f1, i)])
                    mca_mean.append(dices_[reg_][act_]['mca']['{}{}'.format(f2, i)])
                mca_list.append(mean(mca_mean))

            # Get region sizes
            # tool_list = np.nan_to_num(tool_list)
            r_sizes = []
            r_name = []
            for i in range(len(tool_list)):
                l_ = list(dices_.keys())
                s = dices_[l_[i]]['size']
                r_sizes.append(s)
                r_name.append(l_[i])
    
            # Plot Dice values by region size
            tool_dice = np.array(tool_list)
            mca_dice = np.array(mca_list)
            tool_dice_nonz = [ tool_dice[i] for i in range(len(tool_dice)) if not (np.isnan(tool_dice[i]) or np.isnan(mca_dice[i]))]
            mca_dice_nonz = [ mca_dice[i] for i in range(len(tool_dice)) if not (np.isnan(tool_dice[i]) or np.isnan(mca_dice[i]))]
            sizes_nonz = [ r_sizes[i] for i in range(len(tool_dice)) if not (np.isnan(tool_dice[i]) or np.isnan(mca_dice[i]))]
            tools_nonz = [ tool_list[i] for i in range(len(tool_dice)) if tool_dice[i] != 0 and mca_dice[i] != 0]
            tool_dice = tool_dice_nonz
            mca_dice = mca_dice_nonz
            # sizes_nonz = r_sizes
            print(len(tool_dice), len(mca_dice))

            # Is there a correlation between tool_dice and region size?
            _, _, r, p, _ = scipy.stats.linregress(tool_dice, sizes_nonz)
            print(f'BT Dice and region size are correlated (p={p})')
            # Is there a correlation between mca_dice and region size?
            _, _, r, p, _ = scipy.stats.linregress(mca_dice, sizes_nonz)
            print(f'WT Dice and region size are correlated (p={p})')

            slope, intercept, r, p, stderr = scipy.stats.linregress(mca_dice, tool_dice)
            line = f'Regression line: y={intercept:.2f}+{slope:.2f}x (r={r:.2f}, p={p})'
            y = intercept + slope * np.array(mca_dice)
            print(sorted(mca_dice))
            
            ax.plot(mca_dice, tool_dice, linewidth=0, alpha=.5, color=colors[ind_], marker='o', label='{}'.format(tool_.upper()))
            ax.plot(mca_dice, y, color=colors[ind_], alpha=.7, label=line)
            ax.set_xlabel('WT Dice', fontsize=22)
            ax.set_ylabel('BT Dice', fontsize=22)
            ax.legend(fontsize=16)
            ax.tick_params(axis='both', labelsize=18)
    #plt.show()
    name = './paper/figures/dices_corr.png'
    print(f'Dice plot saved in {name}')
    plt.savefig(name, bbox_inches='tight')


############# PRINT STATS

def print_gl_stats(path_):
    # Compute stats of variabilities (ignore NaNs as bg)    
    for type_ in ['unthresh', 'thresh']:
        for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
            bt_ = nib.load('{}{}-{}-rel.nii.gz'.format(path_, pair_, type_))
            bt_abs_data = bt_.get_fdata()
            bt_abs_mean = np.nanmean(bt_abs_data)
            bt_abs_std = np.nanstd(bt_abs_data)
            print('BT variability of tstats in {} {}:\nMean {}\nStd. {}'.format(pair_, type_, bt_abs_mean, bt_abs_std))

            # one-sample t-test for zero centering
            bg_ = np.where((np.isnan(bt_abs_data)) , False, True)
            bt_img = np.nan_to_num(bt_abs_data)[bg_]
            zt_stat1, zp_val1 = stat.ttest_1samp(resample(bt_img, int(1e4)), 0.01)
            print("Zero centering one-sample t-test in BT({}) is {} and p-value {}".format(pair_, zt_stat1, zp_val1))

        for tool_ in ['fsl', 'spm', 'afni']:
            wt2_ = nib.load('{}fuzzy-{}-{}-rel.nii.gz'.format(path_, tool_, type_))
            wt2_abs_data = wt2_.get_fdata()
            wt2_abs_mean = np.nanmean(wt2_abs_data)
            wt2_abs_std = np.nanstd(wt2_abs_data)
            print('WT variability of tstats in {} {}:\nMean {}\nStd. {}'.format(tool_, type_, wt2_abs_mean, wt2_abs_std))

            # one-sample t-test for zero centering
            bg_ = np.where((np.isnan(wt2_abs_data)) , False, True)
            wt_img = np.nan_to_num(wt2_abs_data)[bg_]
            zt_stat1, zp_val1 = stat.ttest_1samp(resample(wt_img, int(1e4)), 0.01)
            print("Zero centering one-sample t-test in WT({}) is {} and p-value {}".format(tool_, zt_stat1, zp_val1))

        
        # FSL stats at global virtual precision t=17
        # if type_ == 'unthresh':
        #     img_ = './data/abs/FL-FSL/p17_fsl_unthresh_abs.nii.gz'
        #     wt2_ = nib.load(img_)
        #     wt2_abs_data = wt2_.get_fdata()
        #     wt2_abs_mean = np.nanmean(wt2_abs_data)
        #     wt2_abs_std = np.nanstd(wt2_abs_data)
        #     print('WT variability of tstats in FSL {} at precision t=17 bits:\nMean {}\nStd. {}'.format(type_, wt2_abs_mean, wt2_abs_std))
        print("stop")


def print_sl_stats(path_):
    # Compute Mean of std over subjects (ignore NaNs as bg)
    min_sbj = 1000
    max_sbj = 0
    all_wt_list = {}
    all_bt_list = {}
    for i in range(1, 17):
        wt_list = []
        bt_list = []
        for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
            bt_ = nib.load(os.path.join(path_, 'sbj{}-{}-unthresh-rel.nii.gz'.format('%.2d' % i, pair_)))
            bt_abs_data = bt_.get_fdata()

            # one-sample t-test for zero centering
            bg_ = np.where((np.isnan(bt_abs_data)) , False, True)
            bt_img = np.nan_to_num(bt_abs_data)[bg_]
            zt_stat1, zp_val1 = stat.ttest_1samp(resample(bt_img, int(1e4)), 0.01)
            print("Sbj {}, Zero centering one-sample t-test in BT({}) is {} and p-value {}".format(i, pair_, zt_stat1, zp_val1))

            bt_abs_mean = (np.nanmean(bt_abs_data))
            # print("Sbj {}, Mean of diff {} is {}".format(i, pair_, bt_abs_mean))
            bt_abs_std = np.nanstd(bt_abs_data)
            bt_list.append(bt_abs_std)
            # print("Sbj {}, STD of diff {} is {}".format(i, pair_, bt_abs_std))
            if pair_ in all_bt_list.keys():
                tmp = all_bt_list[pair_]['mean']
                all_bt_list[pair_]['mean'] = bt_abs_mean + tmp
                tmp = all_bt_list[pair_]['std']
                all_bt_list[pair_]['std'] = bt_abs_std + tmp
            else:
                all_bt_list[pair_] = {}
                all_bt_list[pair_]['mean'] = bt_abs_mean
                all_bt_list[pair_]['std'] = bt_abs_std

        for tool_ in ['fsl', 'spm', 'afni']:
            wt_ = nib.load(os.path.join(path_, 'sbj{}-fuzzy-{}-unthresh-rel.nii.gz'.format('%.2d' % i, tool_)))
            wt_abs_data = wt_.get_fdata()
            # one-sample t-test for zero centering
            bg_ = np.where((np.isnan(wt_abs_data)) , False, True)
            wt_img = np.nan_to_num(wt_abs_data)[bg_]
            zt_stat1, zp_val1 = stat.ttest_1samp(resample(wt_img, int(1e4)), 0.01)
            print("Sbj {}, Zero centering one-sample t-test in WT({}) is {} and p-value {}".format(i, tool_, zt_stat1, zp_val1))

            wt_abs_mean = (np.nanmean(wt_abs_data))
            # print("Sbj {}, Mean of diff {} is {}".format(i, tool_, wt_abs_mean))
            wt_abs_std = np.nanstd(wt_abs_data)
            wt_list.append(wt_abs_std)
            # print("Sbj {}, STD of diff {} is {}".format(i, tool_, wt_abs_std))
            if tool_ in all_wt_list.keys():
                tmp = all_wt_list[tool_]['mean']
                all_wt_list[tool_]['mean'] = wt_abs_mean + tmp
                tmp = all_wt_list[tool_]['std']
                all_wt_list[tool_]['std'] = wt_abs_std + tmp
            else:
                all_wt_list[tool_] = {}
                all_wt_list[tool_]['mean'] = wt_abs_mean
                all_wt_list[tool_]['std'] = wt_abs_std

        if mean(wt_list) > max_sbj:
            wt_list_fff = wt_list
            max_sbj = mean(wt_list)
            i_max = i
            max_sbj_bt = mean(bt_list)

    print('Subject {} has the highest WT variability with average differences of {}\n BT avg diff is {}'.format(i_max, max_sbj, max_sbj_bt))
    
    for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
        print('BT variability of tstats in {} unthresholded:\nMean {}\nStd. {}'.format(pair_, all_bt_list[pair_]['mean']/16, all_bt_list[pair_]['std']/16))
    for tool_ in ['fsl', 'spm', 'afni']:
        print('WT variability of tstats in {} unthresholded:\nMean {}\nStd. {}'.format(tool_, all_wt_list[tool_]['mean']/16, all_wt_list[tool_]['std']/16))

    print('stop')


def compute_stat_test(path_):
    for type_ in ['unthresh', 'thresh']:
        for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
            gvp = False
            num_sample = int(1e4)

            bt_ = nib.load('{}{}-{}-rel.nii.gz'.format(path_, pair_, type_))
            bt_abs_data = bt_.get_fdata()
            bg_ = np.where((np.isnan(bt_.get_fdata())) , False, True)
            bt_img = np.nan_to_num(bt_abs_data)[bg_]

            t1, t2 = pair_.split('-')
            wt1_ = nib.load('{}fuzzy-{}-{}-rel.nii.gz'.format(path_, t1, type_))
            wt1_abs_data = wt1_.get_fdata()
            bg_ = np.where((np.isnan(wt1_.get_fdata())) , False, True)
            img1 = np.nan_to_num(wt1_abs_data)[bg_]

            wt2_ = nib.load('{}fuzzy-{}-{}-rel.nii.gz'.format(path_, t2, type_))
            wt2_abs_data = wt2_.get_fdata()
            bg_ = np.where((np.isnan(wt2_.get_fdata())) , False, True)
            img2 = np.nan_to_num(wt2_abs_data)[bg_]

            res_bt = resample(bt_img, num_sample)
            res1_ = resample(img1, num_sample)
            res2_ = resample(img2, num_sample)

            t_stat1, p_val1 = stat.wilcoxon(res_bt, res1_)
            t_stat2, p_val2 = stat.wilcoxon(res_bt, res2_)
            print("wilcoxon-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t1, t_stat1, p_val1))
            print("wilcoxon-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t2, t_stat2, p_val2))
            t_stat1, p_val1 = stat.ttest_ind(res_bt, res1_)
            t_stat2, p_val2 = stat.ttest_ind(res_bt, res2_)
            print("t-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t1, t_stat1, p_val1))
            print("t-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t2, t_stat2, p_val2))

            print('stop')


def compute_sbj_stat_test(path_):
    for i in range(1, 17):
        print(f"Subject{'%.2d' % i}:")
        for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
            num_sample = int(1e4)
            bt_ = nib.load('{}sbj{}-{}-unthresh-rel.nii.gz'.format(path_, '%.2d' % i, pair_))
            bt_abs_data = bt_.get_fdata()
            bg_ = np.where((np.isnan(bt_.get_fdata())) , False, True)
            bt_img = np.nan_to_num(bt_abs_data)[bg_]

            t1, t2 = pair_.split('-')
            wt1_ = nib.load('{}sbj{}-fuzzy-{}-unthresh-rel.nii.gz'.format(path_, '%.2d' % i, t1))
            wt1_abs_data = wt1_.get_fdata()
            bg_ = np.where((np.isnan(wt1_.get_fdata())) , False, True)
            img1 = np.nan_to_num(wt1_abs_data)[bg_]

            wt2_ = nib.load('{}sbj{}-fuzzy-{}-unthresh-rel.nii.gz'.format(path_, '%.2d' % i, t2))
            wt2_abs_data = wt2_.get_fdata()
            bg_ = np.where((np.isnan(wt2_.get_fdata())) , False, True)
            img2 = np.nan_to_num(wt2_abs_data)[bg_]

            # print('WT img1 mean {} and img2 mean {} and bt_img mean {} '.format(mean(img1), mean(img2), mean(bt_img)))
            res_bt = resample(bt_img, num_sample)
            res1_ = resample(img1, num_sample)
            res2_ = resample(img2, num_sample)

            t_stat1, p_val1 = stat.wilcoxon(res_bt, res1_)
            t_stat2, p_val2 = stat.wilcoxon(res_bt, res2_)
            print("wilcoxon-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t1, t_stat1, p_val1))
            print("wilcoxon-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t2, t_stat2, p_val2))

            t_stat1, p_val1 = stat.ttest_ind(res_bt, res1_)
            t_stat2, p_val2 = stat.ttest_ind(res_bt, res2_)

            print("t-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t1, t_stat1, p_val1))
            print("t-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t2, t_stat2, p_val2))
            print('stop')


def combine_thresh(tool_results, mca_results):
    ## Combine activation and deactivation of thresholded maps ###
    # between tool
    for i in tool_results.keys():
        path_ = os.path.dirname(tool_results[i]['exc_set_file'])
        n = nib.load(tool_results[i]['exc_set_file'])
        d = n.get_data()
        exc_set_nonan = nib.Nifti1Image(np.nan_to_num(d), n.affine, header=n.header)
        n = nib.load(tool_results[i]['exc_set_file_neg'])
        d = n.get_data()

        exc_set_neg_nonan = nib.Nifti1Image(np.nan_to_num(d), n.affine, header=n.header)
        to_display = math_img("img1-img2", img1=exc_set_nonan, img2=exc_set_neg_nonan)
        nib.save(to_display, os.path.join(path_, '{}.nii.gz'.format(i)))

    # within tool
    # fsl_ = tool_results['fsl']['SBJ'].replace('NUM', '%.2d' % i )
    for tool_ in mca_results.keys():
        dic_ = mca_results[tool_]
        for i in range(1, 11):
            stat_file = dic_['exc_set_file'].replace('MCA', str(i))
            path_ = os.path.dirname(stat_file)
            n = nib.load(stat_file)
            d = n.get_data()
            exc_set_nonan = nib.Nifti1Image(np.nan_to_num(d), n.affine, header=n.header)
            n = nib.load(dic_['exc_set_file_neg'].replace('MCA', str(i)))
            d = n.get_data()
            exc_set_neg_nonan = nib.Nifti1Image(np.nan_to_num(d), n.affine, header=n.header)
            to_display = math_img("img1-img2", img1=exc_set_nonan, img2=exc_set_neg_nonan)
            nib.save(to_display, os.path.join(path_, '{}_s{}.nii.gz'.format(tool_, i)))


def main(args=None):

    tool_results = {}
    tool_results['fsl'] = {}
    tool_results['fsl']['exc_set_file'] = './results/ds000001/tools/FSL/thresh_zstat1.nii.gz'
    tool_results['fsl']['exc_set_file_neg'] = './results/ds000001/tools/FSL/thresh_zstat2.nii.gz'
    tool_results['fsl']['stat_file'] = './results/ds000001/tools/FSL/tstat1.nii.gz'
    tool_results['fsl']['act_deact'] = './results/ds000001/tools/FSL/fsl.nii.gz'
    tool_results['fsl']['SBJ'] = './results/ds000001/tools/FSL/subject_level/sbjNUM_tstat1.nii.gz' # NUM will replace with the sbj number
    tool_results['fsl']['SBJ-AR'] = 'results/ds000001/tools/FSL/subject_level/LEVEL1/sub-NUM/run-01.feat/reg/highres2standard.nii.gz' # anatomical registration image
    tool_results['fsl']['SBJ-FR'] = 'results/ds000001/tools/FSL/subject_level/LEVEL1/sub-NUM/run-01.feat/reg/example_func2standard.nii.gz' # functional registration image
    tool_results['spm'] = {}
    tool_results['spm']['exc_set_file'] = './results/ds000001/tools/SPM/Octave/spm_exc_set.nii.gz'
    tool_results['spm']['exc_set_file_neg'] = './results/ds000001/tools/SPM/Octave/spm_exc_set_neg.nii.gz'
    tool_results['spm']['stat_file'] = './results/ds000001/tools/SPM/Octave/spm_stat.nii.gz'
    # tool_results['spm']['exc_set_file'] = './results/ds000001/tools/SPM/Matlab2016b/spm_exc_set.nii.gz'
    # tool_results['spm']['exc_set_file_neg'] = './results/ds000001/tools/SPM/Matlab2016b/spm_exc_set_neg.nii.gz'
    # tool_results['spm']['stat_file'] = './results/ds000001/tools/SPM/Matlab2016b/spm_stat.nii.gz'
    tool_results['spm']['act_deact'] = './results/ds000001/tools/SPM/Octave/spm.nii.gz'
    tool_results['spm']['SBJ'] = './results/ds000001/tools/SPM/Octave/subject_level/sub-NUM/spm_stat.nii.gz' # NUM will replace with the sbj number
    tool_results['afni'] = {}
    tool_results['afni']['exc_set_file'] = './results/ds000001/tools/AFNI/Positive_clustered_t_stat.nii.gz'
    tool_results['afni']['exc_set_file_neg'] = './results/ds000001/tools/AFNI/Negative_clustered_t_stat.nii.gz'
    tool_results['afni']['stat_file'] = './results/ds000001/tools/AFNI/3dMEMA_result_t_stat_masked.nii.gz'
    tool_results['afni']['act_deact'] = './results/ds000001/tools/AFNI/afni.nii.gz'
    tool_results['afni']['SBJ'] = './results/ds000001/tools/AFNI/subject_level/tstats/sbjNUM_result_t_stat_masked.nii.gz' # NUM will replace with the sbj number
    tool_results['afni']['SBJ-AR'] = './results/original/AFNI/subject_level/registrations/anatQQ_images/anatQQ.sub-NUM.nii'

    fsl_mca = {}
    fsl_mca['exc_set_file'] = "./results/ds000001/fuzzy/p53/FSL/runMCA/thresh_zstat1_53_runMCA.nii.gz"
    fsl_mca['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/FSL/runMCA/thresh_zstat2_53_runMCA.nii.gz"
    fsl_mca['stat_file'] = "./results/ds000001/fuzzy/p53/FSL/runMCA/tstat1_53_runMCA.nii.gz"
    fsl_mca['act_deact'] = "./results/ds000001/fuzzy/p53/FSL/runMCA/fsl_sMCA.nii.gz"
    fsl_mca['SBJ'] = './results/ds000001/fuzzy/p53/FSL/subject_level/runMCA/sbjNUM_tstat1.nii.gz'

    spm_mca = {}
    spm_mca['exc_set_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/runMCA/spm_exc_set.nii.gz"
    spm_mca['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/SPM-Octace/runMCA/spm_exc_set_neg.nii.gz"
    spm_mca['stat_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/runMCA/spm_stat.nii.gz"
    spm_mca['act_deact'] = "./results/ds000001/fuzzy/p53/SPM-Octace/runMCA/spm_sMCA.nii.gz"
    spm_mca['SBJ'] = './results/ds000001/fuzzy/p53/SPM-Octace/subject_level/runMCA/sub-NUM/spm_stat.nii.gz'

    afni_mca = {}
    afni_mca['exc_set_file'] = "./results/ds000001/fuzzy/p53/AFNI/runMCA/Positive_clustered_t_stat.nii.gz"
    afni_mca['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/AFNI/runMCA/Negative_clustered_t_stat.nii.gz"
    afni_mca['stat_file'] = "./results/ds000001/fuzzy/p53/AFNI/runMCA/3dMEMA_result_t_stat_masked.nii.gz"
    afni_mca['act_deact'] = "./results/ds000001/fuzzy/p53/AFNI/runMCA/afni_sMCA.nii.gz"
    afni_mca['SBJ'] = './results/ds000001/fuzzy/p53/AFNI/subject_level/tstats-runMCA/sbjNUM_result_t_stat_masked.nii.gz'

    mca_results = {}
    mca_results['fsl'] = {}
    mca_results['fsl'] = fsl_mca
    mca_results['spm'] = {}
    mca_results['spm'] = spm_mca
    mca_results['afni'] = {}
    mca_results['afni'] = afni_mca
    
    rel_path = 'data/diff/'
    #joint histogram
    ### Combine activation and deactivation maps
    # combine_thresh(tool_results, mca_results)

    ### Create abs diff images
    # var_between_tool(tool_results) #BT
    # var_between_fuzzy(mca_results) #WT

    ### Print stats 
    # print_gl_stats(rel_path)
    # print_sl_stats(rel_path)
    # compute_stat_test(rel_path)
    # compute_sbj_stat_test(rel_path)

    ### Plot correlation of variances between BT and FL (Fig 4)  
    # plot_diff_corr_group(sbj=False)  
    # plot_diff_corr_group(sbj=True)

    ### Compute Dice scores and then plot
    # image_parc = './data/MNI-parcellation/HCPMMP1_on_MNI152_ICBM2009a_nlin_resampled.splitLR.nii.gz'
    # regions_txt = './data/MNI-parcellation/HCP-MMP1_on_MNI152_ICBM2009a_nlin.txt'
    # if os.path.exists('./data/dices2_.pkl'):
    #     dices_ = load_variable('dices_')
    #     plot_dices(dices_)
    # else:
    #     dices_ = get_dice_values(regions_txt, image_parc, tool_results, mca_results)
    #     plot_dices(dices_)

    ### abs diff in different precisions in WT
    # path_ = './results/ds000001/fuzzy/'
    # compute_abs_WT(path_)
    ### Global nearest precision
    # p_nearest, all_rmse = global_nearest_precision()
    # print(p_nearest)
    # plot_rmse_nearest(all_rmse)


if __name__ == '__main__':
    main()