import torch
from model import Generator
from config import irgan_config
from data_utils import RecDataset, DataProvider

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

fake_users,fake_items = dp.geneate_synthesis_data(G, train_ui,  fake_users_num= 30)
syn_users = users_list+fake_users
syn_items = items_list+fake_items


with open('./data/ml-100k-syn30/train1.txt', 'w') as file:
    # 将两个列表的元素逐行写入文件
    for item1, item2 in zip(syn_users, syn_items):
        file.write(f'{item1}\t{item2}\n')



  


