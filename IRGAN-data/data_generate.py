import torch
from model import Generator
from config import irgan_config
from data_utils import RecDataset, DataProvider
from scipy.sparse import csr_matrix
from scipy.stats import entropy
import numpy as np

def build_csr_matrix(users_list,items_list,num_users,num_items):
    implicit_rating =[]
    for i in range(len(users_list)):
        implicit_rating.append(1)
    matrix_implicit = csr_matrix((implicit_rating, (users_list, items_list)), shape=(num_users, num_items))
    return matrix_implicit

def get_item_distribution(profiles):
    # [min(max(0, round(i)), 5) for i in a]
    profiles_T = profiles.transpose()
    fn_count = lambda item_vec: np.array(
    [sum([(j == i) for j in item_vec]) for i in range(2)]) / item_vec.shape[0]
    fn_norm = lambda item_vec: np.asarray(item_vec) / np.sum(item_vec)
    item_distribution = np.array(list(map(fn_count, profiles_T)))
    item_distribution = np.array(list(map(fn_norm, item_distribution)))
    return item_distribution

def eval_TVD_JS(P, Q):
    # TVD
    dis_TVD = np.mean(np.sum(np.abs(P - Q) / 2, 1))
    # JS
    fn_KL = lambda p, q: entropy(p, q)
    M = (P + Q) / 2
    js_vec = []
    for iid in range(P.shape[0]):
        p, q, m = P[iid], Q[iid], M[iid]
        js_vec.append((fn_KL(p, m) + fn_KL(q, m)) / 2)
    dis_JS = np.mean(np.array(js_vec))
    return dis_TVD, dis_JS

epochs = irgan_config.epochs
batch_size = irgan_config.batch_size
epochs_d = irgan_config.epochs_d
epochs_g = irgan_config.epochs_g
emb_dim = irgan_config.emb_dim
eta_G = irgan_config.eta_G
eta_D = irgan_config.eta_D
device = irgan_config.device
weight_decay_g = irgan_config.weight_decay_g
weight_decay_d = irgan_config.weight_decay_d
# Hyper-parameters and datset-specific parameters
rec_dataset = RecDataset('./data/ml-100k')
all_users = rec_dataset.get_users()
all_items = rec_dataset.get_items()
num_users = rec_dataset.get_num_users()
num_items = rec_dataset.get_num_items()
bought_mask = rec_dataset.get_bought_mask().to(device)
train_ui = rec_dataset.get_user_item_pairs()
users_list = rec_dataset.get_users_list()
items_list = rec_dataset.get_items_list()
dp = DataProvider(device)
G = Generator(num_users, num_items, emb_dim, bought_mask)
G = G.to(device)

pretrained_model = torch.load("./pretrained_models/ml-100k/pretrained_model_discriminator.pkl",map_location=irgan_config.device)
G.load_state_dict(pretrained_model)
matrix_implicit = build_csr_matrix(users_list,items_list,num_users,num_items)

fake_users,fake_items = dp.geneate_synthesis_data(G, train_ui,  fake_users_num= 10)
syn_users = users_list+fake_users
syn_items = items_list+fake_items
syn_users_num = len(set(syn_users))


matrix_implicit_syn = build_csr_matrix(syn_users,syn_items,syn_users_num,num_items)


real_item_distribution = get_item_distribution(matrix_implicit.toarray())
fake_gan_distribution = get_item_distribution(matrix_implicit_syn.toarray())
dis_TVD, dis_JS = eval_TVD_JS(real_item_distribution, fake_gan_distribution)
print(dis_TVD)
print(dis_JS)

with open('./data/ml-100k-syn10/train1.txt', 'w') as file:
    # 将两个列表的元素逐行写入文件
    for item1, item2 in zip(syn_users, syn_items):
        file.write(f'{item1}\t{item2}\n')



  


