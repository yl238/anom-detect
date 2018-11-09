"""
The script for doing outlier detection using different score
"""
import argparse
from model import VAE
from loss import VAELoss
from dataloader import load_vae_test_datasets, load_vae_train_datasets
import os
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from itertools import product
import numpy as np
import pandas as pd
import scipy.stats as stats

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str)
parser.add_argument('--data', required=True, type=str)
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--kl_weight', type=float, default=1,
                    help="weight on KL term")
parser.add_argument('--out_csv', default='result.csv')
args = parser.parse_args()

# load checkpoint
if not os.path.isfile(args.model_path):
    print('%s is not path to a file' % args.model_path)
    exit()
checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
print("checkpoint loaded!")
print("val loss: {}\tepoch: {}\t".format(checkpoint['val_loss'], checkpoint['epoch']))

# model and criterion
model = VAE(args.image_size)
model.load_state_dict(checkpoint['state_dict'])
criterion = VAELoss(size_average=True, kl_weight=args.kl_weight)

if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()

# load data
test_loader = load_vae_test_datasets(args.image_size, args.data)

############################# ANOMALY SCORE DEF ##########################
def get_vae_score(vae, image, L=5):
    """
    The vae score for a single image, which is basically the loss
    :param image: [1, 3, 256, 256]
    :return (vae loss, KL, reconst_err)
    """
    image_batch = image.expand(L,
                               image.size(1),
                               image.size(2),
                               image.size(3))
    reconst_batch, mu, logvar = vae.forward(image_batch)
    vae_loss, loss_details = criterion(reconst_batch, image_batch, mu, logvar)
    return vae_loss, loss_details['KL'], -loss_details['reconst_logp']

def _log_mean_exp(x, dim):
    """
    A numerical stable version of log(mean(exp(x)))
    :param x: The input
    :param dim: The dimension along which to take mean with
    """
    # m [dim1, 1]
    m, _ = torch.max(x, dim=dim, keepdim=True)

    # x0 [dm1, dim2]
    x0 = x - m

    # m [dim1]
    m = m.squeeze(dim)

    return m + torch.log(torch.mean(torch.exp(x0),
                                    dim=dim))

def get_iwae_score(vae, image, L=5):
    """
    The vae score for a single image, which is basically the loss
    :param image: [1, 3, 256, 256]
    :return scocre: (iwae score, iwae KL, iwae reconst).
    """
    # [L, 3, 256, 256]
    image_batch = image.expand(L,
                               image.size(1),
                               image.size(2),
                               image.size(3))

    # [L, z_dim, 1, 1]
    mu, logvar = vae.encode(image_batch)
    eps = torch.randn_like(mu)
    z = mu + eps * torch.exp(0.5 * logvar)
    kl_weight = criterion.kl_weight
    # [L, 3, 256, 256]
    reconst = vae.decode(z)
    # [L]
    log_p_x_z = -torch.sum((reconst - image_batch).pow(2).reshape(L, -1),
                          dim=1)

    # [L]
    log_p_z = -torch.sum(z.pow(2).reshape(L, -1), dim=1)

    # [L]
    log_q_z = -torch.sum(eps.pow(2).reshape(L, -1), dim=1)

    iwae_score = -_log_mean_exp(log_p_x_z + (log_p_z - log_q_z)*kl_weight, dim=0)
    iwae_KL_score = -_log_mean_exp(log_p_z - log_q_z, dim=0)
    iwae_reconst_score = -_log_mean_exp(log_p_x_z, dim=0)

    return iwae_score, iwae_KL_score, iwae_reconst_score

############################# END OF ANOMALY SCORE ###########################

# Define the number of samples of each score
def compute_all_scores(vae, image):
    """
    Given an image compute all anomaly score
    return (reconst_score, vae_score, iwae_score)
    """
    vae_loss, KL, reconst_err = get_vae_score(vae, image=image, L=15)
    iwae_loss, iwae_KL, iwae_reconst = get_iwae_score(vae, image, L=15)
    result = {'reconst_score': reconst_err.item(),
              'KL_score': KL.item(),
              'vae_score': vae_loss.item(),
              'iwae_score': iwae_loss.item(),
              'iwae_KL_score': iwae_KL.item(),
              'iwae_reconst_score': iwae_reconst.item()}
    return result


# MAIN LOOP
score_names = ['reconst_score', 'KL_score', 'vae_score',
               'iwae_reconst_score', 'iwae_KL_score', 'iwae_score']
classes = test_loader.dataset.classes
scores = {(score_name, cls): [] for (score_name, cls) in product(score_names,
                                                                 classes)}
model.eval()
with torch.no_grad():
    for idx, (image, target) in tqdm(enumerate(test_loader)):
        cls = classes[target.item()]
        if args.cuda:
            image = image.cuda()

        score = compute_all_scores(vae=model, image=image)
        for name in score_names:
            scores[(name, cls)].append(score[name])

# display the mean of scores
means = np.zeros([len(score_names), len(classes)])
for (name, cls) in product(score_names, classes):
    means[score_names.index(name), classes.index(cls)] = sum(scores[(name, cls)]) / len(scores[(name, cls)])
df_mean = pd.DataFrame(means, index=score_names, columns=classes)
print("###################### MEANS #####################")
print(df_mean)


classes.remove('NORMAL')
auc_result = np.zeros([len(score_names), len(classes) + 1])
f1_result = np.zeros([len(score_names), len(classes)+1])
precision_result = np.zeros([len(score_names), len(classes)+1])
recall_result = np.zeros([len(score_names), len(classes)+1])
accuracy_result = np.zeros([len(score_names), len(classes)+1])

# get auc roc for each class
for (name, cls) in product(score_names, classes):
    normal_scores = scores[(name, 'NORMAL')]
    abnormal_scores = scores[(name, cls)]
    y_true = [0]*len(normal_scores) + [1]*len(abnormal_scores)
    y_score = normal_scores + abnormal_scores
    auc_result[score_names.index(name), classes.index(cls)] = roc_auc_score(y_true, y_score)
    f1_result[score_names.index(name), classes.index(cls)] = f1_score(y_true, y_score)
    precision_result[score_names.index(name), classes.index(cls)] = precision_score(y_true, y_score)
    recall_result[score_names.index(name), classes.index(cls)] = recall_score(y_true, y_score)
    accuracy_result[score_names.index(name), classes.index(cls)] = accuracy_score(y_true, y_score)

# add auc roc against all diseases
for name in score_names:
    normal_scores = scores[(name, 'NORMAL')]
    abnormal_scores = np.concatenate([scores[(name, cls)]for cls in classes]).tolist()
    y_true = [0]*len(normal_scores) + [1]*len(abnormal_scores)
    y_score = normal_scores + abnormal_scores
    auc_result[score_names.index(name), -1] = roc_auc_score(y_true, y_score)
    f1_result[score_names.index(name), -1] = f1_score(y_true, y_score)
    precision_result[score_names.index(name), -1] = precision_score(y_true, y_score)
    recall_result[score_names.index(name), -1] = recall_score(y_true, y_score)
    accuracy_result[score_names.index(name), -1] = accuracy_score(y_true, y_score)

df = pd.DataFrame(auc_result, index=score_names, columns=classes + ['ALL'])
# display
print("###################### AUC ROC #####################")
print(df)
print("####################################################")
df.to_csv(args.out_csv)

df = pd.DataFrame(f1_result, index=score_names, columns=classes + ['ALL'])
# display
print("###################### F1 #####################")
print(df)
print("####################################################")
df.to_csv('f1.csv')

df = pd.DataFrame(precision_result, index=score_names, columns=classes + ['ALL'])
# display
print("###################### Precision #####################")
print(df)
print("####################################################")
df.to_csv('precision.csv')

df = pd.DataFrame(recall_result, index=score_names, columns=classes + ['ALL'])
# display
print("###################### Recall #####################")
print(df)
print("####################################################")
df.to_csv('recall.csv')

df = pd.DataFrame(accuracy_result, index=score_names, columns=classes + ['ALL'])
# display
print("###################### Accuracy #####################")
print(df)
print("####################################################")
df.to_csv('accuracy')

# fit a gamma distribution
_, val_loader = load_vae_train_datasets(args.image_size, args.data, 32)
model.eval()
all_reconst_err = []
num_val = len(val_loader.dataset)
with torch.no_grad():
    for img, _ in tqdm(val_loader):
        if args.cuda:
            img = img.cuda()

        # compute output
        recon_batch, mu, logvar = model(img)
        loss, loss_details = criterion.forward_without_reduce(recon_batch, img, mu, logvar)
        reconst_err = -loss_details['reconst_logp']
        all_reconst_err += reconst_err.tolist()

fit_alpha, fit_loc, fit_beta=stats.gamma.fit(all_reconst_err)

# using gamma for outlier detection
# get auc roc for each class
LARGE_NUMBER = 1e30

def get_gamma_score(scores):
    result = -stats.gamma.logpdf(scores, fit_alpha, fit_loc, fit_beta)
    # replace inf in result with largest number
    result[result == np.inf] = LARGE_NUMBER
    return result

auc_gamma_result = np.zeros([1, len(classes)+1])
name = 'reconst_score'
for cls in classes:
    normal_scores = get_gamma_score(scores[(name, 'NORMAL')]).tolist()
    abnormal_scores = get_gamma_score(scores[(name, cls)]).tolist()
    y_true = [0]*len(normal_scores) + [1]*len(abnormal_scores)
    y_score = normal_scores + abnormal_scores
    auc_gamma_result[0, classes.index(cls)] = roc_auc_score(y_true, y_score)

# for all class
normal_scores = get_gamma_score(scores[(name, 'NORMAL')]).tolist()
abnormal_scores = np.concatenate([get_gamma_score(scores[(name, cls)]) for cls in classes]).tolist()
y_true = [0]*len(normal_scores) + [1]*len(abnormal_scores)
y_score = normal_scores + abnormal_scores
auc_gamma_result[0, -1] = roc_auc_score(y_true, y_score)
df = pd.DataFrame(auc_gamma_result, index=['gamma score'], columns=classes + ['ALL'])

# display
print("###################### AUC ROC GAMMA #####################")
print(df)
print("##########################################################")
