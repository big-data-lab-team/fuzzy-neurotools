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


############# Global nearest precision

def calculate_rmse(im1, im2):
    """Computing the RMSE between two images."""
    img1 = np.nan_to_num(im1.get_fdata())
    img2 = np.nan_to_num(im2.get_fdata())

    if img1.shape != img2.shape:
        im1 = resample_from_to(im1, im2, order=0)
        img1 = np.nan_to_num(im1.get_fdata())

    bg_ = np.where((np.isnan(im1.get_fdata())) & (np.isnan(im2.get_fdata())), False, True)
    img1 = img1[bg_]
    img2 = img2[bg_]

    return np.sqrt(np.mean((img1-img2)**2, dtype=np.float64), dtype=np.float64)


def global_nearest_precision():
    rmse_ = {}
    all_rmse = {}
    global_average_rmse = []
    for p in range(11, 54, 2):
        rmse_[p] = 0
        all_rmse[str(p)] = []
        for s in ['fsl-afni', 'fsl-spm', 'afni-spm']:
            bt_ = nib.load('./data/abs/{}-unthresh-abs.nii.gz'.format(s))
            # bt_ = bt_.get_fdata()
            wt_fsl = nib.load('./data/abs/FL-FSL/p{}_fsl_unthresh_abs.nii.gz'.format(p))
            # wt_fsl = wt_fsl.get_fdata()
            rmse_value = calculate_rmse(bt_, wt_fsl)
            all_rmse[str(p)].append(rmse_value)
            rmse_[p] += rmse_value

    all_rmse = np.transpose(np.array(list(all_rmse.values())))
    return min(rmse_, key=rmse_.get), all_rmse


def plot_rmse_nearest(all_rmse):
    import matplotlib.cm as cm
    fig = plt.figure(figsize=(15,10))
    colors = ['red', 'green', 'blue']
    tool_pair = ['fsl-afni', 'fsl-spm', 'afni-spm']
    for s, rmse_ in enumerate(all_rmse):
        #type_ = ""
        #if s+1 in i2T1w: type_ += "*"
        #if s+1 in i2T2w: type_ += "$\dag$"
        plt.plot(range(11,54, 2), rmse_, marker='x', color=colors[s], alpha=0.8, label="({})".format(tool_pair[s].replace('-', ', ').upper()))

    average = np.mean(all_rmse, axis=0, dtype=np.float64)
    p1 = plt.plot(range(11,54, 2), average, marker='o', linewidth=2, color='black', label='Average') #, color='grey', alpha=0.5)
    plt.xticks(range(11,54, 2), fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Virtual precision (bits)', fontsize=22)
    plt.ylabel('RMSE', fontsize=22)
    plt.legend(fontsize=16)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # order = color_sbj_order
    # legend2 = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=12, bbox_to_anchor=(1, .96), title="Subjects")
    # legend1 = plt.legend(p1, ["Average"], bbox_to_anchor=(1.105,1))# bbox_to_anchor=(1.105,0.25) #loc=1
    # plt.gca().add_artist(legend1)
    # plt.gca().add_artist(legend2)
    plt.savefig('./paper/figures/rmse-precisions-abs.png', bbox_inches='tight', facecolor='w', transparent=False)


############# COMPUTE ABS. DIFF.

def compute_abs_WT(path_):
    for p in range (11, 54, 2):
        s1 = os.path.join(path_, 'p{}/FSL/run1/tstat1.nii.gz'.format(p))
        s2 = os.path.join(path_, 'p{}/FSL/run2/tstat1.nii.gz'.format(p))
        s3 = os.path.join(path_, 'p{}/FSL/run3/tstat1.nii.gz'.format(p))
        f1 = nib.load(s1)
        f1_data = f1.get_fdata()
        f2 = nib.load(s2)
        f2_data = f2.get_fdata()
        f3 = nib.load(s3)
        f3_data = f3.get_fdata()
        img_abs = (np.absolute(f1_data - f2_data) + np.absolute(f1_data - f3_data) + np.absolute(f2_data - f3_data))/3
        img_abs = np.where((f1_data == 0) & (f2_data == 0) & (f3_data == 0), np.nan, img_abs)
        nft_img_abs = nib.Nifti1Image(img_abs, f1.affine, header=f1.header)
        nib.save(nft_img_abs, os.path.join('data/abs/FL-FSL/', 'p{}_fsl_unthresh_abs.nii.gz'.format(p)))


def compute_abs(f1, f2, file_name=None, f3=None):
    # Compute absolute diff in BT
    f1 = nib.load(f1)
    f1_data = f1.get_fdata()
    if type(f2) == str : f2 = nib.load(f2)
    f2_data = f2.get_fdata()
    if f3 is None:
        img_abs = np.absolute(f1_data - f2_data)
        # activated regions have nonzero values
        img_abs = np.where((f1_data == 0) & (f2_data == 0), np.nan, img_abs)
        nft_img = nib.Nifti1Image(img_abs, f1.affine, header=f1.header)
        nib.save(nft_img, os.path.join('data/abs/', '{}-abs.nii.gz'.format(file_name)))

    # Compute absolute diff in WT
    else:
        if type(f3) == str : f3 = nib.load(f3)
        f3_data = f3.get_fdata()
        img_abs = (np.absolute(f1_data - f2_data) + np.absolute(f1_data - f3_data) + np.absolute(f2_data - f3_data))/3
        img_abs = np.where((f1_data == 0) & (f2_data == 0) & (f3_data == 0), np.nan, img_abs)
        nft_img_abs = nib.Nifti1Image(img_abs, f1.affine, header=f1.header)
        nib.save(nft_img_abs, os.path.join('data/abs/', '{}-abs.nii.gz'.format(file_name)))
        return nft_img_abs


def combine_abs(f1, f2, meta_, file_name):
    var_f2_res = resample_from_to(f2, f1, order=0)
    # to combine two image variances, we use: var(x+y) = var(x) + var(y) + 2*cov(x,y)
    # and since the correlation between two arrays are so weak, we droped `2*cov(x,y)` from the formula
    f1t = f1.get_fdata()
    f2t = var_f2_res.get_fdata()
    average_abs = (np.nan_to_num(f1.get_fdata()) + np.nan_to_num(var_f2_res.get_fdata()))/2
    # nan bg images
    average_abs = np.where((np.isnan(f1t)) & (np.isnan(f2t)), np.nan, average_abs)
    nft_img = nib.Nifti1Image(average_abs, meta_.affine, header=meta_.header)
    nib.save(nft_img, os.path.join('data/abs/', '{}-abs.nii.gz'.format(file_name)))


def var_between_tool(tool_results):
    for type_ in ['thresh', 'unthresh']:
        f = 'stat_file'
        if type_ == 'thresh': f = 'act_deact'

        fsl_ = tool_results['fsl'][f]
        afni_ = tool_results['afni'][f]
        spm_ = tool_results['spm'][f]
        # resampling first image on second image
        spm_res = resample_from_to(nib.load(spm_), nib.load(fsl_), order=0)
        afni_res = resample_from_to(nib.load(afni_), nib.load(fsl_), order=0)
        # compute abs diff
        compute_abs(fsl_, afni_res, 'fsl-afni-{}'.format(type_), f3=None)
        compute_abs(fsl_, spm_res, 'fsl-spm-{}'.format(type_), f3=None)
        spm_res = resample_from_to(nib.load(spm_), nib.load(afni_), order=0)
        compute_abs(afni_, spm_res, 'afni-spm-{}'.format(type_), f3=None)

    ## unthresholded subject-level
    for i in range(1, 17):
        fsl_ = tool_results['fsl']['SBJ'].replace('NUM', '%.2d' % i )
        afni_ = tool_results['afni']['SBJ'].replace('NUM', '%.2d' % i )
        spm_ = tool_results['spm']['SBJ'].replace('NUM', '%.2d' % i )
        # resampling first image on second image
        spm_res = resample_from_to(nib.load(spm_), nib.load(fsl_), order=0)
        afni_res = resample_from_to(nib.load(afni_), nib.load(fsl_), order=0)
        # compute abd diff
        compute_abs(fsl_, afni_res, 'sbj{}-fsl-afni-unthresh'.format('%.2d' % i), f3=None)
        compute_abs(fsl_, spm_res, 'sbj{}-fsl-spm-unthresh'.format('%.2d' % i), f3=None)
        spm_res = resample_from_to(nib.load(spm_), nib.load(afni_), order=0)
        compute_abs(afni_, spm_res, 'sbj{}-afni-spm-unthresh'.format('%.2d' % i), f3=None)


def var_between_fuzzy(mca_results):
    for type_ in ['thresh', 'unthresh']:
        f = 'stat_file'
        if type_ == 'thresh': f = 'act_deact'
        # compute abs diff
        fsl_abs = compute_abs(mca_results['fsl'][1][f], mca_results['fsl'][2][f], 'fuzzy-fsl-{}'.format(type_), f3=mca_results['fsl'][3][f])
        afni_abs = compute_abs(mca_results['afni'][1][f], mca_results['afni'][2][f], 'fuzzy-afni-{}'.format(type_), f3=mca_results['afni'][3][f])
        spm_abs = compute_abs(mca_results['spm'][1][f], mca_results['spm'][2][f], 'fuzzy-spm-{}'.format(type_), f3=mca_results['spm'][3][f])
        combine_abs(fsl_abs, afni_abs, nib.load(mca_results['fsl'][1][f]), 'fuzzy-fsl-afni-{}'.format(type_))
        combine_abs(fsl_abs, spm_abs, nib.load(mca_results['fsl'][1][f]), 'fuzzy-fsl-spm-{}'.format(type_))
        combine_abs(afni_abs, spm_abs, nib.load(mca_results['afni'][1][f]), 'fuzzy-afni-spm-{}'.format(type_))

    # unthresholded subject-level
    for i in range(1, 17):
        fsl_1 = mca_results['fsl'][1]['SBJ'].replace('NUM', '%.2d' % i )
        fsl_2 = mca_results['fsl'][2]['SBJ'].replace('NUM', '%.2d' % i )
        fsl_3 = mca_results['fsl'][3]['SBJ'].replace('NUM', '%.2d' % i )
        afni_1 = mca_results['afni'][1]['SBJ'].replace('NUM', '%.2d' % i )
        afni_2 = mca_results['afni'][2]['SBJ'].replace('NUM', '%.2d' % i )
        afni_3 = mca_results['afni'][3]['SBJ'].replace('NUM', '%.2d' % i )
        spm_1 = mca_results['spm'][1]['SBJ'].replace('NUM', '%.2d' % i )
        spm_2 = mca_results['spm'][2]['SBJ'].replace('NUM', '%.2d' % i )
        spm_3 = mca_results['spm'][3]['SBJ'].replace('NUM', '%.2d' % i )
        # compute abs diff
        fsl_abs = compute_abs(fsl_1, fsl_2, 'sbj{}-fuzzy-fsl-unthresh'.format('%.2d' % i), f3=fsl_3)
        afni_abs = compute_abs(afni_1, afni_2, 'sbj{}-fuzzy-afni-unthresh'.format('%.2d' % i), f3=afni_3)
        spm_abs = compute_abs(spm_1, spm_2, 'sbj{}-fuzzy-spm-unthresh'.format('%.2d' % i), f3=spm_3)
        combine_abs(fsl_abs, afni_abs, nib.load(fsl_1), 'sbj{}-fuzzy-fsl-afni-unthresh'.format('%.2d' % i))
        combine_abs(fsl_abs, spm_abs, nib.load(fsl_1), 'sbj{}-fuzzy-fsl-spm-unthresh'.format('%.2d' % i))
        combine_abs(afni_abs, spm_abs, nib.load(afni_1), 'sbj{}-fuzzy-afni-spm-unthresh'.format('%.2d' % i))


def get_diff(path_):
    # Group-level
    for type_ in ['thresh', 'unthresh']:
        for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
            img1_ = nib.load('{}{}-{}-abs.nii.gz'.format(path_, pair_, type_))
            img2_ = nib.load('{}fuzzy-{}-{}-abs.nii.gz'.format(path_, pair_, type_))
            img1_data = np.nan_to_num(img1_.get_fdata())
            img2_data = np.nan_to_num(img2_.get_fdata())
            diff_ = img1_data - img2_data
            nft_img = nib.Nifti1Image(diff_, img1_.affine, header=img1_.header)
            nib.save(nft_img, os.path.join(path_, 'diffs', 'btMwt-{}-{}.nii.gz'.format(pair_, type_)))
    
    # Subject-level
    for i in range(1, 17):
        for type_ in ['unthresh']:
            for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
                img1_ = nib.load('{}/sbj{}-{}-{}-abs.nii.gz'.format(path_, '%.2d' % i, pair_, type_))
                img2_ = nib.load('{}/sbj{}-fuzzy-{}-{}-abs.nii.gz'.format(path_, '%.2d' % i, pair_, type_))
                img1_data = np.nan_to_num(img1_.get_fdata())
                img2_data = np.nan_to_num(img2_.get_fdata())
                diff_ = img1_data - img2_data
                nft_img = nib.Nifti1Image(diff_, img1_.affine, header=img1_.header)
                nib.save(nft_img, os.path.join(path_, 'diffs', 'sbj{}-btMwt-{}-{}.nii.gz'.format('%.2d' % i, pair_, type_)))


############# PLOT ABS. DIFF.

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


def plot_corr_variances_group(tool_results):
    ### Plot correlation of variances between BT and WT
    for ind1, type_ in enumerate(['unthresh']):
        fig, ax = plt.subplots(nrows=1,ncols=3,figsize=(24, 7))
        fig2, ax2 = plt.subplots(nrows=1,ncols=3,figsize=(24, 7))
        #fig.suptitle("Correlation of variances between tool-variability vs. numerical-variability")
        for ind_, pair_ in enumerate(['fsl-spm', 'fsl-afni', 'afni-spm']):
            bfuzzy = './data/abs/fuzzy-{}-{}-abs.nii.gz'.format(pair_, type_)
            bfuzzy = nib.load(bfuzzy)
            btool = './data/abs/{}-{}-abs.nii.gz'.format(pair_, type_)
            btool = nib.load(btool)
            if btool.shape != bfuzzy.shape:
                raise NameError('Images from BT and WT are from different dimensions!')

            wt_ = bfuzzy.get_fdata()
            bt_ = btool.get_fdata()

            # check intensity of correlated area
            t1, t2 = pair_.split('-')
            check_intensities(bt_, wt_, f'Group-level {type_} {pair_}', ax2[ind_],
                              tool_results[t1]['stat_file'])

            data1 = np.reshape(wt_, -1)
            data1 = np.nan_to_num(data1)
            data2 = np.reshape(bt_, -1)
            data2 = np.nan_to_num(data2)

            r, p = scipy.stats.pearsonr(data1, data2)
            # print("P-value: {} and R: {}".format(p, r))
            slope, intercept, r2, p2, stderr = scipy.stats.linregress(data1, data2)
            #line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
            line = 'Regression line'
            y = intercept + slope * data1

            t1, t2 = pair_.split('-')
            label_ = "({}, {}) \np={}, r={:.4f}".format(t1.upper(), t2.upper(), p, r) #'{}'.format(pair_.upper())
            # if single: label_ = '{}'.format(pair_.upper())

            ax[ind_].plot(data1, data2, linewidth=0, marker='o', alpha=.5, label=label_)
            # ax[ind_].plot([0, 2.5], [0, 2.5], color='black', linestyle='dashed', label='Identity line')

            ax[ind_].plot(data1, y, color='darkred', label=line)

            # ci = 1.96 * np.std(y)/np.mean(y)
            # ax.fill_between(x, (y-ci), (y+ci), color='g', alpha=.9)
            ax[ind_].set_title('')
            ax[ind_].set_xlabel('')
            if ind_ == 0: ax[ind_].set_ylabel('BT variability', fontsize=14)
            ax[ind_].set_xlabel('WT variability', fontsize=14)
            ax[ind_].set_ylim([-0.15, 7.9])
            ax[ind_].set_xlim([-0.07, 2.6])
            # ax[ind_].set_xticklabels(fontsize=14)
            # ax[ind_].set_yticklabels(fontsize=14)
            ax[ind_].legend(facecolor='white', loc='upper right', fontsize=12)
            ax[ind_].tick_params(axis='both', labelsize=12)
            #ax[ind_].set_xscale('log')

            if ind_ == 0: ax2[ind_].set_ylabel('#voxels', fontsize=14)
            ax2[ind_].set_xlabel('T-statistics', fontsize=14)

        fig.savefig('./paper/figures/abs/corr/abs-corr-{}-plot.png'.format(type_), bbox_inches='tight')
        fig2.savefig('./paper/figures/abs/corr/hist/hist-corr-{}-plot.png'.format(type_), bbox_inches='tight')


def plot_corr_variances_gvp(tool_results):
    ### Plot correlation of variances between BT and WT at (global virtual precision)
    for ind1, type_ in enumerate(['unthresh']):
        fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(24, 7))
        fig2, ax2 = plt.subplots(nrows=1,ncols=2,figsize=(24, 7))
        #fig.suptitle("Correlation of variances between tool-variability vs. numerical-variability")
        for ind_, pair_ in enumerate(['fsl-spm', 'fsl-afni']):
            bfuzzy = './data/abs/FL-FSL/p17_fsl_unthresh_abs.nii.gz'
            bfuzzy = nib.load(bfuzzy)
            btool = './data/abs/{}-{}-abs.nii.gz'.format(pair_, type_)
            btool = nib.load(btool)
            if btool.shape != bfuzzy.shape:
                raise NameError('Images from BT and WT are from different dimensions!')

            wt_ = bfuzzy.get_fdata()
            bt_ = btool.get_fdata()

            # check intensity of correlated area
            t1, t2 = pair_.split('-')
            check_intensities(bt_, wt_, f'Group level at global t: {type_} {pair_}', ax2[ind_],
                              tool_results[t1]['stat_file'])

            data1 = np.reshape(wt_, -1)
            data1 = np.nan_to_num(data1)
            data2 = np.reshape(bt_, -1)
            data2 = np.nan_to_num(data2)

            r, p = scipy.stats.pearsonr(data1, data2)
            t1, t2 = pair_.split('-')
            label_ = "({}, {}) \np={}, r={:.4f}".format(t1.upper(), t2.upper(), p, r) #'{}'.format(pair_.upper())
            slope, intercept, r2, p2, stderr = scipy.stats.linregress(data1, data2)
            line = 'Regression line'
            y = intercept + slope * data1

            ax[ind_].plot(data1, data2, linewidth=0, marker='o', alpha=.5, label=label_)
            # ax[ind_].plot([0, 3.7], [0, 3.7], color='black', linestyle='dashed', label='Identity line')
            ax[ind_].plot(data1, y, color='darkred', label=line)

            ax[ind_].set_title('')
            ax[ind_].set_xlabel('')
            if ind_ == 0: ax[ind_].set_ylabel('BT variability', fontsize=20)
            ax[ind_].set_xlabel('WT variability', fontsize=20)
            ax[ind_].set_ylim([-0.15, 7.9])
            ax[ind_].set_xlim([-0.07, 3.8])
            ax[ind_].legend(facecolor='white', loc='upper right', fontsize=16)
            ax[ind_].tick_params(axis='both', labelsize=16)

            if ind_ == 0: ax2[ind_].set_ylabel('#voxels', fontsize=20)
            ax2[ind_].set_xlabel('T-statistics', fontsize=20)

        fig.savefig('./paper/figures/abs/corr/abs-corr-{}-vp17-plot.png'.format(type_), bbox_inches='tight')
        fig2.savefig('./paper/figures/abs/corr/hist/hist-corr-{}-vp17-plot.png'.format(type_), bbox_inches='tight')


def plot_corr_variances_sbj(tool_results):
    ### Plot correlation of variances between BT and WT
    for ind1, type_ in enumerate(['unthresh']):
        fig, ax = plt.subplots(nrows=16,ncols=3,figsize=(28, 64))
        fig2, ax2 = plt.subplots(nrows=16,ncols=3,figsize=(28, 64))
        #fig.suptitle("Correlation of variances between tool-variability vs. numerical-variability")
        for i in range (1, 17):
            for ind_, pair_ in enumerate(['fsl-spm', 'fsl-afni', 'afni-spm']):
                # bfuzzy = './data/std/fuzzy-{}-{}-std.nii.gz'.format(pair_, type_)
                bfuzzy = './data/abs/sbj{}-fuzzy-{}-{}-abs.nii.gz'.format('%.2d' % i, pair_, type_)
                bfuzzy = nib.load(bfuzzy)
                # btool = './data/std/{}-{}-std.nii.gz'.format(pair_, type_)
                btool = './data/abs/sbj{}-{}-{}-abs.nii.gz'.format('%.2d' % i, pair_, type_)
                btool = nib.load(btool)
                if btool.shape != bfuzzy.shape:
                    raise NameError('Images from BT and WT are from different dimensions!')

                wt_ = bfuzzy.get_fdata()
                bt_ = btool.get_fdata()

                # check intensity of correlated area
                t1, t2 = pair_.split('-')
                img_type = 'SBJ-FR' #'SBJ-AR', 'SBJ'
                if img_type in tool_results[t1].keys():
                    tool1 = tool_results[t1][img_type].replace('NUM', '%.2d' % i )
                    tool1 = resample_from_to(nib.load(tool1), btool, order=0)
                    check_intensities(bt_, wt_, f"subject{'%.2d' % i}: {type_} {pair_}", ax2[i-1][ind_], tool1)

                data1 = np.reshape(wt_, -1)
                data1 = np.nan_to_num(data1)
                data2 = np.reshape(bt_, -1)
                data2 = np.nan_to_num(data2)

                r, p = scipy.stats.pearsonr(data1, data2)
                print("P-value: {} and R: {}".format(p, r))
                slope, intercept, r2, p2, stderr = scipy.stats.linregress(data1, data2)
                line = 'Regression line'
                y = intercept + slope * data1

                t1, t2 = pair_.split('-')
                label_ = "({}, {}) \np={}, r={:.4f}".format(t1.upper(), t2.upper(), p, r)

                ax[i-1][ind_].plot(data1, data2, linewidth=0, marker='o', alpha=.5, label=label_)
                # ax[i-1][ind_].plot([0, 2.5], [0, 2.5], color='black', linestyle='dashed', label='Identity line')
                ax[i-1][ind_].plot(data1, y, color='darkred', label=line)

                ax[i-1][ind_].set_title('')
                ax[i-1][ind_].set_xlabel('')
                if ind_ == 0: ax[i-1][ind_].set_ylabel('BT variability', fontsize=14)
                if i == 16: ax[i-1][ind_].set_xlabel('WT variability', fontsize=14)
                ax[i-1][ind_].set_ylim([-0.15, 7.9])
                ax[i-1][ind_].set_xlim([-0.07, 2.6])
                ax[i-1][ind_].legend(facecolor='white', loc='upper right', fontsize=12)
                ax[i-1][ind_].tick_params(axis='both', labelsize=12)

                if ind_ == 0: ax2[i-1][ind_].set_ylabel('#voxels', fontsize=14)
                if i == 16: ax2[i-1][ind_].set_xlabel('Intensities (func on MNI)', fontsize=14)

        fig.savefig('./paper/figures/abs/corr/sbj-abs-corr-{}-plot.png'.format(type_), bbox_inches='tight')
        fig2.savefig('./paper/figures/abs/corr/hist/his-{}-corr-{}-plot.png'.format(img_type, type_), bbox_inches='tight')


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
            # if val in ["L_V1_ROI", "L_V2_ROI", "L_V3_ROI", "L_V4_ROI", "L_LO1_ROI", "L_LO2_ROI"]:
            #            non_bg = np.where(parc_data != 0)
            #            L_colls = np.where(parc_data == int(key))
            #            R_colls = np.where(parc_data == int(key) + 180)
            #            total = len(L_colls[0]) + len(R_colls[0])
            #            perc_ = (total / non_bg[0].size  )*100
            #            print(f"Region name: {'_'.join(val.split('_')[1:])} and percentage of voxels: {perc_}")

    dices_ = {}
    file_cols = ["Region", "FSL IEEE", "SPM IEEE", "AFNI IEEE", "FSL MCA1", "FSL MCA2", "FSL MCA3",
                 "SPM MCA1", "SPM MCA2", "SPM MCA3", "AFNI MCA1", "AFNI MCA2", "AFNI MCA3"]
    dframe = pd.DataFrame(np.array([['R', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]),
                          columns=file_cols)
    #for act_ in ['exc_set_file', 'exc_set_file_neg', 'act_deact', 'stat_file']:
    for act_ in ['act_deact']:
        ### Print global Dice values 
        bt1 = compute_dice(tool_results['fsl'][act_], tool_results['afni'][act_])[0]
        bt2 = compute_dice(tool_results['fsl'][act_], tool_results['spm'][act_])[0]
        bt3 = compute_dice(tool_results['afni'][act_], tool_results['spm'][act_])[0]
        print("Global Dice in BT for FSL-AFNI {}, FSL-SPM {}, and AFNI-SPM {}".format(bt1, bt2, bt3))

        for tool_ in ['fsl', 'spm', 'afni']:            
            wt1 = compute_dice(mca_results[tool_][1][act_], mca_results[tool_][2][act_])[0]
            wt2 = compute_dice(mca_results[tool_][1][act_], mca_results[tool_][3][act_])[0]
            wt3 = compute_dice(mca_results[tool_][2][act_], mca_results[tool_][3][act_])[0]
            print("Global Dice in WT for {} {}".format(tool_, (wt1+wt2+wt3)/3))

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
                masked_regions['{}1'.format(tool_)], s = keep_roi(mca_results[tool_][1][act_], r, image_parc)#, '{}1'.format(tool_))
                act_bin.append(s)
                masked_regions['{}2'.format(tool_)], s = keep_roi(mca_results[tool_][2][act_], r, image_parc)#, '{}2'.format(tool_))
                act_bin.append(s)
                masked_regions['{}3'.format(tool_)], s = keep_roi(mca_results[tool_][3][act_], r, image_parc)#, '{}3'.format(tool_))
                act_bin.append(s)

                dices_[regions[r]][act_]['mca']['{}1'.format(tool_)] = compute_dice(masked_regions['{}1'.format(tool_)], masked_regions['{}2'.format(tool_)])[0]
                dices_[regions[r]][act_]['mca']['{}2'.format(tool_)] = compute_dice(masked_regions['{}1'.format(tool_)], masked_regions['{}3'.format(tool_)])[0]
                dices_[regions[r]][act_]['mca']['{}3'.format(tool_)] = compute_dice(masked_regions['{}2'.format(tool_)], masked_regions['{}3'.format(tool_)])[0]

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
    for type_ in ['thresh', 'unthresh']:
        for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
            bt_ = nib.load('{}{}-{}-abs.nii.gz'.format(path_, pair_, type_))
            bt_abs_data = bt_.get_fdata()
            bt_abs_mean = np.nanmean(bt_abs_data)
            bt_abs_std = np.nanstd(bt_abs_data)
            print('BT variability of tstats in {} {}:\nMean {}\nStd. {}'.format(pair_, type_, bt_abs_mean, bt_abs_std))

        for tool_ in ['fsl', 'spm', 'afni']:
            wt2_ = nib.load('{}fuzzy-{}-{}-abs.nii.gz'.format(path_, tool_, type_))
            wt2_abs_data = wt2_.get_fdata()
            wt2_abs_mean = np.nanmean(wt2_abs_data)
            wt2_abs_std = np.nanstd(wt2_abs_data)
            print('WT variability of tstats in {} {}:\nMean {}\nStd. {}'.format(tool_, type_, wt2_abs_mean, wt2_abs_std))
        
        # FSL stats at global virtual precision t=17
        if type_ == 'unthresh':
            img_ = './data/abs/FL-FSL/p17_fsl_unthresh_abs.nii.gz'
            wt2_ = nib.load(img_)
            wt2_abs_data = wt2_.get_fdata()
            wt2_abs_mean = np.nanmean(wt2_abs_data)
            wt2_abs_std = np.nanstd(wt2_abs_data)
            print('WT variability of tstats in FSL {} at precision t=17 bits:\nMean {}\nStd. {}'.format(type_, wt2_abs_mean, wt2_abs_std))
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
            bt_ = nib.load(os.path.join(path_, 'sbj{}-{}-unthresh-abs.nii.gz'.format('%.2d' % i, pair_)))
            bt_abs_data = bt_.get_fdata()
            bt_abs_mean = np.nanmean(bt_abs_data)
            bt_list.append(bt_abs_mean)
            bt_abs_std = np.nanstd(bt_abs_data)
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
            wt_ = nib.load(os.path.join(path_, 'sbj{}-fuzzy-{}-unthresh-abs.nii.gz'.format('%.2d' % i, tool_)))
            wt_abs_data = wt_.get_fdata()
            wt_abs_mean = np.nanmean(wt_abs_data)
            wt_list.append(wt_abs_mean)
            wt_abs_std = np.nanstd(wt_abs_data)
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
            max_sbj = mean(wt_list)
            i_max = i
            max_sbj_bt = mean(bt_list)

    print('Subject {} has the highest WT variability with average absolute differences of {}\n BT avg abs diff is {}'.format(i_max, max_sbj, max_sbj_bt))
    
    for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
        print('BT variability of tstats in {} unthresholded:\nMean {}\nStd. {}'.format(pair_, all_bt_list[pair_]['mean']/16, all_bt_list[pair_]['std']/16))
    for tool_ in ['fsl', 'spm', 'afni']:
        print('WT variability of tstats in {} unthresholded:\nMean {}\nStd. {}'.format(tool_, all_wt_list[tool_]['mean']/16, all_wt_list[tool_]['std']/16))

    print('stop')


def compute_stat_test(path_):
    for type_ in ['unthresh', 'thresh']:
        for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
            gvp = False
            num_sample = int(1e3)

            bt_ = nib.load('{}{}-{}-abs.nii.gz'.format(path_, pair_, type_))
            bt_abs_data = bt_.get_fdata()
            bg_ = np.where((np.isnan(bt_.get_fdata())) , False, True)
            bt_img = np.nan_to_num(bt_abs_data)[bg_]

            t1, t2 = pair_.split('-')
            wt1_ = nib.load('{}fuzzy-{}-{}-abs.nii.gz'.format(path_, t1, type_))
            wt1_abs_data = wt1_.get_fdata()
            bg_ = np.where((np.isnan(wt1_.get_fdata())) , False, True)
            img1 = np.nan_to_num(wt1_abs_data)[bg_]

            wt2_ = nib.load('{}fuzzy-{}-{}-abs.nii.gz'.format(path_, t2, type_))
            wt2_abs_data = wt2_.get_fdata()
            bg_ = np.where((np.isnan(wt2_.get_fdata())) , False, True)
            img2 = np.nan_to_num(wt2_abs_data)[bg_]

            if type_ == 'unthresh' and t1 == 'fsl':
                gvp = True
                wtGvp = nib.load('./data/abs/FL-FSL/p17_fsl_unthresh_abs.nii.gz')
                wtGvp_abs_data = wtGvp.get_fdata()
                bg_ = np.where((np.isnan(wtGvp.get_fdata())) , False, True)
                imgGvp = np.nan_to_num(wtGvp_abs_data)[bg_]
                resGvp = resample(imgGvp, num_sample)

            res_bt = resample(bt_img, num_sample)
            res1_ = resample(img1, num_sample)
            res2_ = resample(img2, num_sample)

            t_stat1, p_val1 = stat.wilcoxon(res_bt, res1_)
            t_stat2, p_val2 = stat.wilcoxon(res_bt, res2_)
            print("wilcoxon-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t1, t_stat1, p_val1*6))
            print("wilcoxon-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t2, t_stat2, p_val2*6))

            t_stat1, p_val1 = stat.ttest_ind(res_bt, res1_)
            t_stat2, p_val2 = stat.ttest_ind(res_bt, res2_)
            print("t-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t1, t_stat1, p_val1*6))
            print("t-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t2, t_stat2, p_val2*6))

            if gvp:
                t_statGvp, p_valGvp = stat.wilcoxon(res_bt, resGvp)
                print("wilcoxon-test between BT({}) and WT({}) at GVP t=17 bits is {} and p-value {}".format(pair_, 'FSL', t_statGvp, p_valGvp*6))
                t_statGvp, p_valGvp = stat.ttest_ind(res_bt, resGvp)
                print("t-test between BT({}) and WT({}) at GVP t=17 bits is {} and p-value {}".format(pair_, 'FSL', t_statGvp, p_valGvp*6))

            print('stop')


def compute_sbj_stat_test(path_):
    for i in range(1, 17):
        print(f"Subject{'%.2d' % i}:")
        for pair_ in ['fsl-spm', 'fsl-afni', 'afni-spm']:
            num_sample = int(1e3)
            bt_ = nib.load('{}sbj{}-{}-unthresh-abs.nii.gz'.format(path_, '%.2d' % i, pair_))
            bt_abs_data = bt_.get_fdata()
            bg_ = np.where((np.isnan(bt_.get_fdata())) , False, True)
            bt_img = np.nan_to_num(bt_abs_data)[bg_]

            t1, t2 = pair_.split('-')
            wt1_ = nib.load('{}sbj{}-fuzzy-{}-unthresh-abs.nii.gz'.format(path_, '%.2d' % i, t1))
            wt1_abs_data = wt1_.get_fdata()
            bg_ = np.where((np.isnan(wt1_.get_fdata())) , False, True)
            img1 = np.nan_to_num(wt1_abs_data)[bg_]

            wt2_ = nib.load('{}sbj{}-fuzzy-{}-unthresh-abs.nii.gz'.format(path_, '%.2d' % i, t2))
            wt2_abs_data = wt2_.get_fdata()
            bg_ = np.where((np.isnan(wt2_.get_fdata())) , False, True)
            img2 = np.nan_to_num(wt2_abs_data)[bg_]

            # print('WT img1 mean {} and img2 mean {} and bt_img mean {} '.format(mean(img1), mean(img2), mean(bt_img)))
            res_bt = resample(bt_img, num_sample)
            res1_ = resample(img1, num_sample)
            res2_ = resample(img2, num_sample)

            t_stat1, p_val1 = stat.wilcoxon(res_bt, res1_)
            t_stat2, p_val2 = stat.wilcoxon(res_bt, res2_)
            print("wilcoxon-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t1, t_stat1, p_val1*6))
            print("wilcoxon-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t2, t_stat2, p_val2*6))

            t_stat1, p_val1 = stat.ttest_ind(res_bt, res1_)
            t_stat2, p_val2 = stat.ttest_ind(res_bt, res2_)

            print("t-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t1, t_stat1, p_val1*6))
            print("t-test between BT({}) and WT({}) is {} and p-value {}".format(pair_, t2, t_stat2, p_val2*6))
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
    for tool_ in mca_results.keys():
        dic_ = mca_results[tool_]
        for i in dic_.keys():
            path_ = os.path.dirname(dic_[i]['exc_set_file'])
            n = nib.load(dic_[i]['exc_set_file'])
            d = n.get_data()
            exc_set_nonan = nib.Nifti1Image(np.nan_to_num(d), n.affine, header=n.header)
            n = nib.load(dic_[i]['exc_set_file_neg'])
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
    fsl_mca[1] = {}
    fsl_mca[1]['exc_set_file'] = "./results/ds000001/fuzzy/p53/FSL/run1/thresh_zstat1_53_run1.nii.gz"
    fsl_mca[1]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/FSL/run1/thresh_zstat2_53_run1.nii.gz"
    fsl_mca[1]['stat_file'] = "./results/ds000001/fuzzy/p53/FSL/run1/tstat1_53_run1.nii.gz"
    fsl_mca[1]['act_deact'] = "./results/ds000001/fuzzy/p53/FSL/run1/fsl_s1.nii.gz"
    fsl_mca[1]['SBJ'] = './results/ds000001/fuzzy/p53/FSL/subject_level/run1/sbjNUM_tstat1.nii.gz'
    fsl_mca[2] = {}
    fsl_mca[2]['exc_set_file'] = "./results/ds000001/fuzzy/p53/FSL/run2/thresh_zstat1_53_run2.nii.gz"
    fsl_mca[2]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/FSL/run2/thresh_zstat2_53_run2.nii.gz"
    fsl_mca[2]['stat_file'] = "./results/ds000001/fuzzy/p53/FSL/run2/tstat1_53_run2.nii.gz"
    fsl_mca[2]['act_deact'] = "./results/ds000001/fuzzy/p53/FSL/run2/fsl_s2.nii.gz"
    fsl_mca[2]['SBJ'] = './results/ds000001/fuzzy/p53/FSL/subject_level/run2/sbjNUM_tstat1.nii.gz'
    fsl_mca[3] = {}
    fsl_mca[3]['exc_set_file'] = "./results/ds000001/fuzzy/p53/FSL/run3/thresh_zstat1_53_run3.nii.gz"
    fsl_mca[3]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/FSL/run3/thresh_zstat2_53_run3.nii.gz"
    fsl_mca[3]['stat_file'] = "./results/ds000001/fuzzy/p53/FSL/run3/tstat1_53_run3.nii.gz"
    fsl_mca[3]['act_deact'] = "./results/ds000001/fuzzy/p53/FSL/run3/fsl_s3.nii.gz"
    fsl_mca[3]['SBJ'] = './results/ds000001/fuzzy/p53/FSL/subject_level/run3/sbjNUM_tstat1.nii.gz'

    spm_mca = {}
    spm_mca[1] = {}
    spm_mca[1]['exc_set_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run1/spm_exc_set.nii.gz"
    spm_mca[1]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run1/spm_exc_set_neg.nii.gz"
    spm_mca[1]['stat_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run1/spm_stat.nii.gz"
    spm_mca[1]['act_deact'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run1/spm_s1.nii.gz"
    spm_mca[1]['SBJ'] = './results/ds000001/fuzzy/p53/SPM-Octace/subject_level/run1/sub-NUM/spm_stat.nii.gz'
    spm_mca[2] = {}
    spm_mca[2]['exc_set_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run2/spm_exc_set.nii.gz"
    spm_mca[2]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run2/spm_exc_set_neg.nii.gz"
    spm_mca[2]['stat_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run2/spm_stat.nii.gz"
    spm_mca[2]['act_deact'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run2/spm_s2.nii.gz"
    spm_mca[2]['SBJ'] = './results/ds000001/fuzzy/p53/SPM-Octace/subject_level/run2/sub-NUM/spm_stat.nii.gz'
    spm_mca[3] = {}
    spm_mca[3]['exc_set_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run3/spm_exc_set.nii.gz"
    spm_mca[3]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run3/spm_exc_set_neg.nii.gz"
    spm_mca[3]['stat_file'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run3/spm_stat.nii.gz"
    spm_mca[3]['act_deact'] = "./results/ds000001/fuzzy/p53/SPM-Octace/run3/spm_s3.nii.gz"
    spm_mca[3]['SBJ'] = './results/ds000001/fuzzy/p53/SPM-Octace/subject_level/run3/sub-NUM/spm_stat.nii.gz'

    afni_mca = {}
    afni_mca[1] = {}
    afni_mca[1]['exc_set_file'] = "./results/ds000001/fuzzy/p53/AFNI/run1/Positive_clustered_t_stat.nii.gz"
    afni_mca[1]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/AFNI/run1/Negative_clustered_t_stat.nii.gz"
    afni_mca[1]['stat_file'] = "./results/ds000001/fuzzy/p53/AFNI/run1/3dMEMA_result_t_stat_masked.nii.gz"
    afni_mca[1]['act_deact'] = "./results/ds000001/fuzzy/p53/AFNI/run1/afni_s1.nii.gz"
    afni_mca[1]['SBJ'] = './results/ds000001/fuzzy/p53/AFNI/subject_level/tstats-run1/sbjNUM_result_t_stat_masked.nii.gz'
    afni_mca[2] = {}
    afni_mca[2]['exc_set_file'] = "./results/ds000001/fuzzy/p53/AFNI/run2/Positive_clustered_t_stat.nii.gz"
    afni_mca[2]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/AFNI/run2/Negative_clustered_t_stat.nii.gz"
    afni_mca[2]['stat_file'] = "./results/ds000001/fuzzy/p53/AFNI/run2/3dMEMA_result_t_stat_masked.nii.gz"
    afni_mca[2]['act_deact'] = "./results/ds000001/fuzzy/p53/AFNI/run2/afni_s2.nii.gz"
    afni_mca[2]['SBJ'] = './results/ds000001/fuzzy/p53/AFNI/subject_level/tstats-run2/sbjNUM_result_t_stat_masked.nii.gz'
    afni_mca[3] = {}
    afni_mca[3]['exc_set_file'] = "./results/ds000001/fuzzy/p53/AFNI/run3/Positive_clustered_t_stat.nii.gz"
    afni_mca[3]['exc_set_file_neg'] = "./results/ds000001/fuzzy/p53/AFNI/run3/Negative_clustered_t_stat.nii.gz"
    afni_mca[3]['stat_file'] = "./results/ds000001/fuzzy/p53/AFNI/run3/3dMEMA_result_t_stat_masked.nii.gz"
    afni_mca[3]['act_deact'] = "./results/ds000001/fuzzy/p53/AFNI/run3//afni_s3.nii.gz"
    afni_mca[3]['SBJ'] = './results/ds000001/fuzzy/p53/AFNI/subject_level/tstats-run3/sbjNUM_result_t_stat_masked.nii.gz'

    mca_results = {}
    mca_results['fsl'] = {}
    mca_results['fsl'] = fsl_mca
    mca_results['spm'] = {}
    mca_results['spm'] = spm_mca
    mca_results['afni'] = {}
    mca_results['afni'] = afni_mca
    
    abs_path = 'data/abs/'
    ### Combine activation and deactivation maps
    # combine_thresh(tool_results, mca_results)

    ### Create abs diff images
    # var_between_tool(tool_results) #BT
    # var_between_fuzzy(mca_results) #WT
    ### Create diff images between BT and WT abs diff images
    # get_diff(abs_path)

    ### abs diff in different precisions in WT
    # path_ = './results/ds000001/fuzzy/'
    # compute_abs_WT(path_)
    ### Global nearest precision
    # p_nearest, all_rmse = global_nearest_precision()
    # print(p_nearest)
    # plot_rmse_nearest(all_rmse)

    ### Plot correlation of variances between BT and FL (Fig 4)
    # plot_corr_variances_group(tool_results)
    # plot_corr_variances_gvp(tool_results)
    # plot_corr_variances_sbj(tool_results)

    ### Compute Dice scores and then plot
    # image_parc = './data/MNI-parcellation/HCPMMP1_on_MNI152_ICBM2009a_nlin_resampled.splitLR.nii.gz'
    # regions_txt = './data/MNI-parcellation/HCP-MMP1_on_MNI152_ICBM2009a_nlin.txt'
    # if os.path.exists('./data/dices_.pkl'):
    #     dices_ = load_variable('dices_')
    #     plot_dices(dices_)
    # else:
    #     dices_ = get_dice_values(regions_txt, image_parc, tool_results, mca_results)
    #     plot_dices(dices_)

    ### Print stats 
    # print_gl_stats(abs_path)
    # print_sl_stats(abs_path)
    # compute_stat_test(abs_path)
    # compute_sbj_stat_test(abs_path)

if __name__ == '__main__':
    main()